"""HTTP API.  Run:  uvicorn visrag.api:app --port 8000 --env-file .env   (needs GROQ_API_KEY)"""
import io
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image

from . import __version__
from .pipeline import get_client, run

MAX_BYTES = 8 * 1024 * 1024
UI_PAGE = Path(__file__).parent / "static" / "index.html"

app = FastAPI(title="VisLoc-RAG", version=__version__,
              description="Photo in, cited and fact-checked facts out.")


@app.get("/", include_in_schema=False)
def ui() -> FileResponse:
    return FileResponse(UI_PAGE)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": __version__}


@app.post("/facts")
def facts(file: UploadFile = File(...), location: str = Form("")) -> dict:
    data = file.file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "Image larger than 8 MB")
    try:
        Image.open(io.BytesIO(data)).verify()
    except Exception:
        raise HTTPException(400, "Not a readable image") from None
    try:
        client = get_client()
    except RuntimeError as e:
        raise HTTPException(503, str(e)) from None
    result = run(data, location, client=client)
    return {"subject": result["subject"], "facts": result["shown_facts"],
            "facts_checked": len(result["facts"]),
            "sources": [{k: s[k] for k in ("id", "title", "url")} for s in result["sources"]],
            "timing": result["timing"]}
