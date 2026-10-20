"""API tests: no internet or API key needed (the pipeline is replaced with a stub)."""
import io

from fastapi.testclient import TestClient
from PIL import Image

import visrag.api as api

client = TestClient(api.app)


def _jpeg() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), "gray").save(buf, format="JPEG")
    return buf.getvalue()


def test_health():
    assert client.get("/health").json()["status"] == "ok"


def test_rejects_non_image():
    r = client.post("/facts", files={"file": ("x.jpg", b"not an image", "image/jpeg")})
    assert r.status_code == 400


def test_rejects_large_upload():
    r = client.post("/facts", files={"file": ("x.jpg", b"0" * (api.MAX_BYTES + 1), "image/jpeg")})
    assert r.status_code == 413


def test_missing_key_is_503(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    r = client.post("/facts", files={"file": ("x.jpg", _jpeg(), "image/jpeg")})
    assert r.status_code == 503 and "GROQ_API_KEY" in r.json()["detail"]


def test_returns_only_supported_facts_without_source_text(monkeypatch):
    shown = [{"fact": "Built in 1609.", "sources": ["S1"], "supported": True}]
    monkeypatch.setattr(api, "get_client", lambda: object())
    monkeypatch.setattr(api, "run", lambda data, location, client: {
        "subject": {"name": "Himeji Castle"},
        "facts": shown + [{"fact": "It is gold.", "sources": ["S1"], "supported": False}],
        "shown_facts": shown,
        "sources": [{"id": "S1", "title": "Himeji Castle (en)", "url": "u", "text": "long text"}],
        "timing": {"total_s": 1.0}})
    r = client.post("/facts", files={"file": ("x.jpg", _jpeg(), "image/jpeg")}, data={"location": "Himeji"})
    body = r.json()
    assert r.status_code == 200
    assert body["facts"] == shown
    assert body["sources"] == [{"id": "S1", "title": "Himeji Castle (en)", "url": "u"}]
