"""Evaluate the pipeline on landmark photos taken from Wikipedia (run where internet works, e.g. Colab).

For each landmark: download the article's lead image, run the pipeline, then measure
- identification accuracy (the predicted name matches the title or an alias),
- facts per image and the share the fact-checker marks as supported,
- time per stage and tokens per image.
It also writes a CSV of random facts with their sources so you can label some by hand and check
how often the automatic fact-checker agrees with you.

Usage:
  export GROQ_API_KEY=...            # or Colab secrets
  python eval/run_eval.py --limit 60 --out results/run1.json
  python eval/run_eval.py --no-location --out results/no_location.json    # ablation
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import statistics
import sys
import time
import unicodedata
from pathlib import Path
from urllib.parse import quote

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from visrag.pipeline import HEADERS, get_client, run  # noqa: E402


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^0-9a-z\u3040-\u30ff\u4e00-\u9fff]", "", s.lower())


def matches(pred: str, title: str, aliases: list[str]) -> bool:
    p = norm(pred)
    for name in [title, re.sub(r"\s*\(.*\)", "", title), *aliases]:
        n = norm(name)
        if n and (n in p or (len(p) >= 4 and p in n)):
            return True
    return False


def lead_image(title: str) -> bytes | None:
    url = "https://en.wikipedia.org/api/rest_v1/page/summary/" + quote(title.replace(" ", "_"))
    try:
        info = requests.get(url, headers=HEADERS, timeout=15).json()
        src = (info.get("originalimage") or info.get("thumbnail") or {}).get("source")
        return requests.get(src, headers=HEADERS, timeout=30).content if src else None
    except (requests.RequestException, ValueError):
        return None


def pct(xs, q):
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(q * len(xs)))], 2) if xs else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--landmarks", default=str(Path(__file__).with_name("landmarks.csv")))
    ap.add_argument("--limit", type=int)
    ap.add_argument("--sample", type=int, help="evaluate a fixed random subset of this size (see --seed)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-location", action="store_true", help="ablation: hide the location hint")
    ap.add_argument("--sleep", type=float, default=2.0, help="pause between images (rate limits)")
    ap.add_argument("--out", default="results/run.json")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.landmarks, encoding="utf-8")))
    if args.sample:
        rows = random.Random(args.seed).sample(rows, min(args.sample, len(rows)))
    rows = rows[:args.limit]
    client, results = get_client(), []
    for i, row in enumerate(rows, 1):
        aliases = [a for a in row["aliases"].split("|") if a]
        img = lead_image(row["title"])
        if not img:
            print(f"[{i}/{len(rows)}] skip (no image): {row['title']}")
            continue
        try:
            r = run(img, "" if args.no_location else row["location"], client=client)
        except Exception as e:  # keep going; count failures
            results.append({"title": row["title"], "error": str(e)})
            print(f"[{i}/{len(rows)}] ERROR {row['title']}: {e}")
            continue
        ok = matches(r["subject"].get("name", ""), row["title"], aliases)
        results.append({"title": row["title"], "pred": r["subject"].get("name"), "identified": ok,
                        "facts": r["facts"], "sources": r["sources"], "timing": r["timing"],
                        "usage": r["usage"]})
        n_ok = sum(f["supported"] for f in r["facts"])
        print(f"[{i}/{len(rows)}] {'OK ' if ok else 'BAD'} {row['title']!r} -> "
              f"{r['subject'].get('name')!r} | facts {len(r['facts'])}, supported {n_ok}")
        time.sleep(args.sleep)

    done = [r for r in results if "error" not in r]
    facts = [f for r in done for f in r["facts"]]
    summary = {
        "images": len(results), "errors": len(results) - len(done),
        "identification_acc": round(sum(r["identified"] for r in done) / len(done), 3) if done else None,
        "facts_per_image": round(len(facts) / len(done), 2) if done else None,
        "supported_fact_rate": round(sum(f["supported"] for f in facts) / len(facts), 3) if facts else None,
        "images_with_unsupported_fact": sum(any(not f["supported"] for f in r["facts"]) for r in done),
        "p50_total_s": pct([r["timing"]["total_s"] for r in done], 0.5),
        "p95_total_s": pct([r["timing"]["total_s"] for r in done], 0.95),
        "tokens_per_image": (round(statistics.mean(sum(r["usage"].values()) for r in done), 1)
                             if done else None),
        "location_hint": not args.no_location,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    sample = random.Random(0).sample(facts, min(30, len(facts)))
    by_fact = {id(f): r for r in done for f in r["facts"]}
    with open(out.with_suffix(".manual_check.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["landmark", "fact", "checker_says_supported", "your_label (yes/no)", "evidence"])
        for f in sample:
            r = by_fact[id(f)]
            ev = " | ".join(s["text"][:400] for s in r["sources"] if s["id"] in f.get("sources", []))
            w.writerow([r["title"], f["fact"], f["supported"], "", ev])
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
