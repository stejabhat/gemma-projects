import io
import os
import re
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import pytesseract
import requests
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pdf2image import convert_from_path

app = FastAPI(title="PDF OCR + Ollama Layout Extraction")

BASE_DIR = Path(__file__).resolve().parent

# Allow local dev; tighten in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

INDEX_HTML = BASE_DIR / "index.html"


@app.get("/")
async def serve_index() -> FileResponse:
    if not INDEX_HTML.is_file():
        raise HTTPException(status_code=404, detail="index.html not found")
    return FileResponse(INDEX_HTML)

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma4:12b")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL_OVERRIDE", OLLAMA_MODEL)
_OLLAMA_MAX_INPUT_CHARS = int(os.environ.get("OLLAMA_MAX_INPUT_CHARS", "12000"))


def _ollama_extract_structured(ocr_text: str, start_page: int, end_page: int) -> str:
    if len(ocr_text) > _OLLAMA_MAX_INPUT_CHARS:
        ocr_text = (
            ocr_text[:_OLLAMA_MAX_INPUT_CHARS]
            + "\n\n[Truncated for model context]"
        )

    prompt = """Structure these OCR pages %d-%d.
Keep headings, lists, tables, paragraph breaks.
Output Markdown headings/tables when clear.
Do not invent content.
If unclear, keep it as-is.
Output only structured text, max 250 words.
""" % (start_page, end_page) + "\n\n" + ocr_text

    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "num_predict": 256,
                    "num_ctx": 2048,
                },
            },
            timeout=None,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"Ollama error {resp.status_code}: {resp.text}")
        data = resp.json()
        return (data.get("response") or data.get("message") or "").strip()
    except requests.Timeout as e:
        raise RuntimeError(
            "Ollama timed out. This host likely cannot run the selected model. "
            "Use a smaller model or fewer pages."
        ) from e
    except Exception as e:
        raise RuntimeError(f"Ollama failed: {e}") from e


@app.post("/api/process")
async def process_pdf(
    file: UploadFile = File(...),
    start_page: int = Form(1),
    end_page: int = Form(1),
    use_llm: bool = Form(True),
):
    if start_page < 1 or end_page < 1:
        raise HTTPException(status_code=400, detail="start_page and end_page must be >= 1")
    if end_page < start_page:
        raise HTTPException(status_code=400, detail="end_page must be >= start_page")

    # Save upload to a temp file for pdf2image
    suffix = os.path.splitext(file.filename or "")[1].lower()
    if suffix not in [".pdf", ""]:
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    content = await file.read()
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        try:
            total_pages = _count_pdf_pages(tmp_path)
        except ValueError as ve:
            raise HTTPException(status_code=400, detail=str(ve)) from ve

        if total_pages <= 0:
            raise HTTPException(status_code=400, detail="Could not determine PDF page count")

        if start_page > total_pages or end_page > total_pages:
            raise HTTPException(
                status_code=400,
                detail=f"Page bounds out of range. PDF has {total_pages} pages.",
            )

        images = _pdf_pages_to_images(tmp_path, start_page=start_page, end_page=end_page)
        if not images:
            raise HTTPException(status_code=400, detail="No pages rendered from PDF")

        page_texts = []
        for idx, img in enumerate(images, start=start_page):
            try:
                t = _ocr_image_to_text(img)
            finally:
                img.close()
            page_texts.append(f"[Page {idx}]\n{t}")

        ocr_text = "\n\n".join(page_texts).strip()

        structured = ""
        if use_llm:
            try:
                structured = _ollama_extract_structured(ocr_text, start_page=start_page, end_page=end_page)
            except RuntimeError as e:
                structured = "[LLM structuring failed: %s]\n\n%s" % (e, ocr_text)

        return {
            "total_pages": total_pages,
            "start_page": start_page,
            "end_page": end_page,
            "ocr_text": ocr_text,
            "structured_content": structured,
        }
    finally:
        if tmp_path is not None:
            try:
                os.remove(tmp_path)
            except Exception:
                pass

