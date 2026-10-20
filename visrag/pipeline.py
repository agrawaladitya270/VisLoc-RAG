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

import requests
from PIL import Image, ImageOps

from . import __version__

VISION_MODEL = os.getenv("VISRAG_VISION_MODEL", "qwen/qwen3.8-27b")
TEXT_MODEL = os.getenv("VISRAG_TEXT_MODEL", "openai/gpt-oss-120b")
# The checker is a different model family from the writer, so they are less likely to share mistakes.
CHECK_MODEL = os.getenv("VISRAG_CHECK_MODEL", "qwen/qwen3.8-27b")
HEADERS = {"User-Agent": f"visloc-rag/{__version__} (https://github.com/agrawaladitya270/VisLoc-RAG)"}


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
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(image_bytes))).convert("RGB")  # phone rotation
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
    return resp.choices[0].message.content or ""


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


class WikipediaBusy(RuntimeError):
    """Wikipedia answered HTTP 429: too many requests."""


def _wiki(lang: str, **params) -> dict:
    r = requests.get(f"https://{lang}.wikipedia.org/w/api.php", headers=HEADERS, timeout=10,
                     params={"action": "query", "format": "json", "formatversion": 2, **params})
    if r.status_code == 429:
        raise WikipediaBusy("Wikipedia is limiting requests right now. Try again in "
                            f"{r.headers.get('retry-after', 'a few')} seconds.")
    r.raise_for_status()
    return r.json()["query"]


def _pages(lang: str, full_text: bool = False, **params) -> list[dict]:
    """Pages with plain text, URL and Japanese language link, in search-rank order.

    Without `full_text` only the lead section is returned; the API gives full text for one page per request.
    """
    intro = {} if full_text else {"exintro": 1}   # MediaWiki treats any value, even 0, as true
    try:
        query = _wiki(lang, prop="extracts|info|langlinks", explaintext=1, inprop="url", lllang="ja",
                      redirects=1, **intro, **params)
    except (requests.RequestException, KeyError, ValueError):
        return []
    return sorted(query.get("pages", []), key=lambda p: p.get("index", 0))


def wikipedia_sources(query: str, per_lang: int = 2, max_chars: int = 3000) -> list[dict]:
    """Top English articles for the query plus the Japanese version of the best one, as numbered sources.

    The Japanese article is found through Wikipedia's language link, which is more precise than
    searching Japanese Wikipedia with an English query; that search is only the fallback.
    Three API requests: an English search (lead sections and language links), then the full text of
    the best English match and of its Japanese version. Few requests keeps us inside rate limits.
    """
    sources = []

    def add(lang: str, pages: list[dict]) -> None:
        for page in pages:
            text = (page.get("extract") or "").strip()
            if text:
                sources.append({"id": f"S{len(sources) + 1}", "title": f"{page['title']} ({lang})",
                                "url": page.get("fullurl", ""), "text": text[:max_chars]})

    en = _pages("en", generator="search", gsrsearch=query, gsrlimit=per_lang)
    if en:
        en[0] = (_pages("en", full_text=True, titles=en[0]["title"]) or en)[0]
    add("en", en)
    ja_title = next((link["title"] for page in en[:1] for link in page.get("langlinks", [])), None)
    if ja_title:
        add("ja", _pages("ja", full_text=True, titles=ja_title))
    else:
        add("ja", _pages("ja", generator="search", gsrsearch=query, gsrlimit=1))
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
        location=location or "unknown", sources=src)}], usage, max_tokens=4000, temperature=0.2)
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
        # room for reasoning models, which think before answering
        reply = _chat(client, CHECK_MODEL, [{"role": "user", "content": CHECK_PROMPT.format(
            evidence=evidence, fact=f["fact"])}], usage, max_tokens=400, temperature=0)
        f["supported"] = reply.strip().lower().startswith("yes")
    return facts


def run(image_bytes: bytes, location: str = "", client=None, search=wikipedia_sources,
        name: str = "") -> dict:
    """With `name`, the user has said what the photo shows, so the vision step is skipped."""
    client = client or get_client()
    usage: dict = {}
    t = {}
    t0 = time.perf_counter()
    if name:
        subject = {"name": name, "search_query": name, "named_by_user": True}
    else:
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
