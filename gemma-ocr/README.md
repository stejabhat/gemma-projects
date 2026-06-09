# PDF OCR + Structured Extraction

Scanned PDF OCR pipeline using FastAPI, Tesseract, and Ollama.

**Workflow**
1. Upload a scanned PDF
2. Select a page range (1-based)
3. OCR the selected pages with Tesseract
4. Send OCR text to Ollama to produce structured Markdown output preserving layout, headings, lists, and tables

**Output**
- `ocr_text`: raw reconstructed text
- `structured_content`: LLM-refined Markdown structure

## Prerequisites

**System packages**

```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr poppler-utils
```

**Ollama**

- Ollama must be running on `http://localhost:11434`
- Use a small CPU-friendly model for reliable performance on low-RAM hosts:

```bash
export OLLAMA_MODEL_OVERRIDE=gemma3:1b
ollama pull gemma3:1b
```

If `gemma3:1b` is unavailable, alternatives like `qwen2.5:1.5b` or `phi3:mini` also work well.

The 12B-tier models are not recommended for constrained hosts; generation may be very slow or stall.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
export OLLAMA_MODEL_OVERRIDE=gemma3:1b
uvicorn server:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000/` and use the web UI.

**Do not serve `index.html` with `python3 -m http.server`.** The UI submits `POST /api/process` to the same origin. If the static server handles the UI, that POST hits SimpleHTTP and returns `501`.

## API

`POST /api/process`

| Field        | Type  | Notes                          |
|--------------|-------|--------------------------------|
| `file`       | file  | PDF document                   |
| `start_page` | int   | 1-based inclusive start        |
| `end_page`   | int   | 1-based inclusive end          |

Response

```json
{
  "total_pages": 10,
  "start_page": 1,
  "end_page": 2,
  "ocr_text": "...",
  "structured_content": "..."
}
```

## Notes

- Processing time depends on page count, image quality, OCR speed, and model size; the server uses no request timeout so Ollama can run as long as needed.
- Image memory is freed after each page; temp files are always cleaned up.
- CORS allows `*` for local development. Tighten before exposing to the internet.
