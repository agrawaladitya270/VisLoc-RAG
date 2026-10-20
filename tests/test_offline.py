"""Runs without internet or API keys, using a fake model client."""
import io
import json
from types import SimpleNamespace

import pytest
from PIL import Image

import visrag.pipeline as pipeline
from eval.run_eval import matches
from visrag.pipeline import parse_json, run, wikipedia_sources


class FakeClient:
    """Answers the three kinds of calls the pipeline makes, in order."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        usage = SimpleNamespace(prompt_tokens=100, completion_tokens=20)
        msg = SimpleNamespace(content=self.replies.pop(0))
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=usage)


def _jpeg() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), "gray").save(buf, format="JPEG")
    return buf.getvalue()


def test_parse_json_handles_fences():
    assert parse_json('```json\n{"name": "Himeji Castle"}\n```')["name"] == "Himeji Castle"
    assert parse_json('Here: [{"fact": "x", "sources": ["S1"]}]')[0]["fact"] == "x"


def test_alias_matching():
    assert matches("Kinkaku-ji (Golden Pavilion)", "Kinkaku-ji", ["金閣寺"])
    assert matches("Todai-ji temple", "Tōdai-ji", [])
    assert matches("Rainbow Bridge", "Rainbow Bridge (Tokyo)", [])
    assert not matches("Osaka Castle", "Himeji Castle", ["姫路城"])


def test_pipeline_marks_unsupported_and_uncited_facts():
    facts = [{"fact": "Built in 1609.", "sources": ["S1"]},
             {"fact": "It is painted gold.", "sources": ["S1"]},
             {"fact": "No source given.", "sources": []}]
    client = FakeClient([
        json.dumps({"name": "Himeji Castle", "category": "landmark",
                    "search_query": "Himeji Castle", "confidence": "high"}),
        json.dumps(facts),
        "yes",   # check fact 1
        "no",    # check fact 2 (fact 3 has no source, so no call)
    ])
    sources = [{"id": "S1", "title": "Himeji Castle (en)", "url": "", "text": "Completed in 1609."}]
    out = run(_jpeg(), "Himeji", client=client, search=lambda q: sources)
    assert [f["supported"] for f in out["facts"]] == [True, False, False]
    assert len(out["shown_facts"]) == 1
    assert out["usage"]["prompt_tokens"] == 400


def test_user_given_name_skips_vision_step():
    client = FakeClient([json.dumps([{"fact": "Built in 1989.", "sources": ["S1"]}]), "yes"])
    sources = [{"id": "S1", "title": "Asahi Beer Hall (en)", "url": "", "text": "Completed in 1989."}]
    queries = []
    out = run(_jpeg(), "", client=client, search=lambda q: queries.append(q) or sources,
              name="Asahi Beer Hall")
    assert queries == ["Asahi Beer Hall"] and out["subject"]["named_by_user"]
    assert len(out["shown_facts"]) == 1 and client.replies == []


def _fake_wiki(calls, ja_link=True):
    def wiki(lang, **params):
        calls.append((lang, params.get("gsrsearch") or params.get("titles"),
                      "lead" if "exintro" in params else "full"))
        if lang == "en":
            page = {"title": "Himeji Castle", "index": 1, "extract": "Text.", "fullurl": "https://x"}
            if ja_link:
                page["langlinks"] = [{"lang": "ja", "title": "姫路城"}]
            return {"pages": [page]}
        return {"pages": [{"title": "姫路城", "extract": "本文。", "fullurl": "https://y"}]}
    return wiki


def test_japanese_article_comes_from_language_link(monkeypatch):
    calls = []
    monkeypatch.setattr(pipeline, "_wiki", _fake_wiki(calls))
    sources = wikipedia_sources("Himeji Castle")
    assert [s["title"] for s in sources] == ["Himeji Castle (en)", "姫路城 (ja)"]
    # search, then full text of the best match and of its Japanese version; no English query on ja
    assert calls == [("en", "Himeji Castle", "lead"), ("en", "Himeji Castle", "full"),
                     ("ja", "姫路城", "full")]


def test_japanese_search_is_fallback_without_language_link(monkeypatch):
    calls = []
    monkeypatch.setattr(pipeline, "_wiki", _fake_wiki(calls, ja_link=False))
    sources = wikipedia_sources("Himeji Castle")
    assert [s["title"] for s in sources] == ["Himeji Castle (en)", "姫路城 (ja)"]
    assert calls[-1] == ("ja", "Himeji Castle", "lead")


def test_rate_limit_is_reported_not_treated_as_no_results(monkeypatch):
    busy = SimpleNamespace(status_code=429, headers={"retry-after": "20"})
    monkeypatch.setattr(pipeline.requests, "get", lambda *a, **kw: busy)
    with pytest.raises(pipeline.WikipediaBusy, match="20 seconds"):
        wikipedia_sources("Himeji Castle")


def test_no_sources_means_no_facts():
    client = FakeClient([json.dumps({"name": "Unknown", "category": "object", "search_query": "x"})])
    out = run(_jpeg(), "", client=client, search=lambda q: [])
    assert out["facts"] == [] and out["shown_facts"] == []
