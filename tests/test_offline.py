"""Runs without internet or API keys, using a fake model client."""
import io
import json
from types import SimpleNamespace

from PIL import Image

from eval.run_eval import matches
from visrag.pipeline import parse_json, run


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


def test_no_sources_means_no_facts():
    client = FakeClient([json.dumps({"name": "Unknown", "category": "object", "search_query": "x"})])
    out = run(_jpeg(), "", client=client, search=lambda q: [])
    assert out["facts"] == [] and out["shown_facts"] == []
