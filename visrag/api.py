"""HTTP API.  Run:  uvicorn visrag.api:app --port 8000   (needs GROQ_API_KEY)"""
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from . import __version__
from .pipeline import run

app = FastAPI(title="VisLoc-RAG", version=__version__)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": __version__}


@app.post("/facts")
def facts(file: UploadFile = File(...), location: str = Form("")) -> dict:
    data = file.file.read(8 * 1024 * 1024 + 1)
    if len(data) > 8 * 1024 * 1024:
        raise HTTPException(413, "Image larger than 8 MB")
    result = run(data, location)
    return {"subject": result["subject"], "facts": result["shown_facts"],
            "sources": [{k: s[k] for k in ("id", "title", "url")} for s in result["sources"]],
            "timing": result["timing"]}
