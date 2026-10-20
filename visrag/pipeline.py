"""Photo -> identify the subject -> fetch evidence -> write cited facts -> check each fact.

- Evidence comes from Wikipedia (English + Japanese) as numbered sources; Japanese text is passed
  to the model directly instead of being machine-translated.
- Facts must cite source ids, temperature is low, and a separate call checks each fact against
  the sources it cites. Only supported facts are shown by default.
- The API key is read from the environment; network calls have timeouts and retries.
- Every run records token usage and per-stage timing for evaluation.
"""
from __future__ import annotations

import base64
import io
import json
import os
import re
import time
from urllib.parse import quote

import requests
from PIL import Image

VISION_MODEL = os.getenv("VISRAG_VISION_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
TEXT_MODEL = os.getenv("VISRAG_TEXT_MODEL", "llama-3.3-70b-versatile")
HEADERS = {"User-Agent": "visloc-rag/0.2 (portfolio project; contact via GitHub)"}


def get_client():
    from groq import Groq
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise RuntimeError("Set the GROQ_API_KEY environment variable. Never put keys in code.")
    return Groq(api_key=key, timeout=30, max_retries=2)


def parse_json(text: str):
    """Pull the first JSON object or array out of a model reply (handles ```json fences)."""
    m = re.search(r"(\{.*\}|\[.*\])", text, re.S)
    if not m:
        raise ValueError(f"No JSON in reply: {text[:200]!r}")
    return json.loads(m.group(1))


def image_to_data_url(image_bytes: bytes, max_side: int = 1024) -> str:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _chat(client, model: str, messages: list, usage: dict, **kw) -> str:
    resp = client.chat.completions.create(model=model, messages=messages, **kw)
    u = getattr(resp, "usage", None)
    if u:
        usage["prompt_tokens"] = usage.get("prompt_tokens", 0) + (u.prompt_tokens or 0)
        usage["completion_tokens"] = usage.get("completion_tokens", 0) + (u.completion_tokens or 0)
    return resp.choices[0].message.content


IDENTIFY_PROMPT = """Identify the main subject of this photo.
The user's current location (may or may not be related): {location}

Return ONLY JSON:
{{"name": "most specific name, e.g. 'Himeji Castle'",
  "category": "landmark | building | nature | food | art | object | animal | street",
  "search_query": "short query to look this up in an encyclopedia",
  "confidence": "high | medium | low"}}"""


def identify(client, image_bytes: bytes, location: str, usage: dict) -> dict:
    reply = _chat(client, VISION_MODEL, [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": image_to_data_url(image_bytes)}},
        {"type": "text", "text": IDENTIFY_PROMPT.format(location=location or "unknown")},
    ]}], usage, max_tokens=200, temperature=0)
    return parse_json(reply)


def wikipedia_sources(query: str, langs=("en", "ja"), per_lang: int = 2) -> list[dict]:
    """Top Wikipedia articles for the query, as numbered sources with their summary text."""
    sources = []
    for lang in langs:
        try:
            r = requests.get(f"https://{lang}.wikipedia.org/w/api.php", headers=HEADERS, timeout=10,
                             params={"action": "query", "list": "search", "srsearch": query,
                                     "srlimit": per_lang, "format": "json"})
            titles = [h["title"] for h in r.json()["query"]["search"]]
        except (requests.RequestException, KeyError, ValueError):
            continue
        for title in titles:
            try:
                s = requests.get(f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/"
                                 + quote(title.replace(" ", "_")), headers=HEADERS, timeout=10).json()
            except (requests.RequestException, ValueError):
                continue
            if s.get("extract"):
                sources.append({"id": f"S{len(sources) + 1}", "title": f"{title} ({lang})",
                                "url": s.get("content_urls", {}).get("desktop", {}).get("page", ""),
                                "text": s["extract"][:1500]})
    return sources


FACTS_PROMPT = """Subject: {name} ({category}). User location: {location}

Sources (some may be in Japanese):
{sources}

Write up to 5 interesting, specific facts about the subject using ONLY these sources.
- Every fact must list the ids of the sources that support it.
- Prefer concrete numbers, dates and names. No general knowledge that is not in the sources.
- If the sources are about something else, return an empty list.
Write the facts in English. Return ONLY JSON:
[{{"fact": "...", "sources": ["S1"]}}]"""

CHECK_PROMPT = """Is the claim fully supported by the source text? Answer only "yes" or "no".

Source text:
{evidence}

Claim: {fact}"""


def write_facts(client, subject: dict, sources: list[dict], location: str, usage: dict) -> list[dict]:
    if not sources:
        return []
    src = "\n\n".join(f"[{s['id']}] {s['title']}: {s['text']}" for s in sources)
    reply = _chat(client, TEXT_MODEL, [{"role": "user", "content": FACTS_PROMPT.format(
        name=subject.get("name", ""), category=subject.get("category", ""),
        location=location or "unknown", sources=src)}], usage, max_tokens=800, temperature=0.2)
    facts = parse_json(reply)
    return [f for f in facts if isinstance(f, dict) and f.get("fact")][:5]


def check_facts(client, facts: list[dict], sources: list[dict], usage: dict) -> list[dict]:
    """A separate model call per fact: is it supported by the sources it cites?"""
    by_id = {s["id"]: s for s in sources}
    for f in facts:
        cited = [by_id[i] for i in f.get("sources", []) if i in by_id]
        if not cited:
            f["supported"] = False
            continue
        evidence = "\n\n".join(s["text"] for s in cited)
        reply = _chat(client, TEXT_MODEL, [{"role": "user", "content": CHECK_PROMPT.format(
            evidence=evidence, fact=f["fact"])}], usage, max_tokens=5, temperature=0)
        f["supported"] = reply.strip().lower().startswith("yes")
    return facts


def run(image_bytes: bytes, location: str = "", client=None, search=wikipedia_sources) -> dict:
    client = client or get_client()
    usage: dict = {}
    t = {}
    t0 = time.perf_counter()
    subject = identify(client, image_bytes, location, usage)
    t["identify_s"] = time.perf_counter() - t0
    t1 = time.perf_counter()
    sources = search(subject.get("search_query") or subject.get("name", ""))
    t["search_s"] = time.perf_counter() - t1
    t2 = time.perf_counter()
    facts = write_facts(client, subject, sources, location, usage)
    t["facts_s"] = time.perf_counter() - t2
    t3 = time.perf_counter()
    facts = check_facts(client, facts, sources, usage)
    t["check_s"] = time.perf_counter() - t3
    t["total_s"] = time.perf_counter() - t0
    return {"subject": subject, "sources": sources, "facts": facts,
            "shown_facts": [f for f in facts if f.get("supported")],
            "timing": {k: round(v, 2) for k, v in t.items()}, "usage": usage}
