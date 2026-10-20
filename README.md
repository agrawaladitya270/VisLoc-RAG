# VisLoc-RAG — photo in, cited and fact-checked facts out

Take a photo of a landmark, building, dish or object. The app identifies it with a vision model,
pulls evidence from Wikipedia (English and Japanese), writes up to 5 facts that **cite their
sources**, and then **checks each fact** against the sources it cites. Only supported facts are shown.

```
photo ─► vision LLM: what is it? ─► Wikipedia en + ja (numbered sources) ─► LLM: facts with [S#] citations
      ─► per-fact check: "is this claim supported by its sources?" ─► show supported facts + sources
```

## What changed from the first notebook version
- **Security:** the API key was hard-coded in the notebook. It is now read from `GROQ_API_KEY`.
  (If an old notebook with a key was ever pushed, revoke that key and clean the git history.)
- **Crash fix:** the facts step used a variable that was never defined (`image_location`).
- **Real grounding:** facts must cite source ids; a separate check marks each fact supported or not.
  Previously the prompt only *asked* the model to use the search text, and nothing verified it.
- **Better evidence:** Wikipedia's API instead of scraping search-result snippets; Japanese text is
  given to the model directly instead of being machine-translated twice.
- **Engineering:** timeouts and retries, low temperature for factual output, token counting,
  per-stage timing, FastAPI endpoint, Docker image, CI with offline tests (fake model client).

## Results

Run the evaluation below and fill in this table. **Do not quote numbers you have not run.**

| Setting | Images | Identified correctly | Facts per image | Facts supported (checker) | Checker agrees with you (30 hand-labeled) | p50 time |
|---|---|---|---|---|---|---|
| With location hint | | | | | | |
| Without location hint | | | | | | |

## Run the evaluation (Colab)

```python
!git clone https://github.com/<your-user>/VisLoc-RAG
%cd VisLoc-RAG
!pip install -q -r requirements.txt

import os
from google.colab import userdata                        # store the key in Colab "Secrets"
os.environ["GROQ_API_KEY"] = userdata.get("GROQ_API_KEY")

!python eval/run_eval.py --limit 60 --out results/with_location.json
!python eval/run_eval.py --limit 60 --no-location --out results/no_location.json
```

- `eval/landmarks.csv` holds 58 landmarks (33 in Japan) with aliases. The script downloads each
  article's lead image from Wikipedia, runs the pipeline and scores it.
- Each run also writes `*.manual_check.csv` with 30 random facts and their evidence. Fill in the
  `your_label` column (yes/no), then count how often the automatic checker agrees with you. That
  agreement number is what makes the "supported" rate believable.
- Each image makes up to 7 model calls. If you hit rate limits, raise `--sleep`.
- Wikipedia lead images are well-known photos, so this is an easier test than random phone photos.
  Add 20 of your own photos (with the right answer) for a harder, more honest check.

## Run the app

```bash
pip install -r requirements.txt
export GROQ_API_KEY=...                      # never commit keys; see .env.example
uvicorn visrag.api:app --port 8000
curl -F "file=@photo.jpg" -F "location=Kobe" localhost:8000/facts

docker build -t visloc-rag . && docker run -e GROQ_API_KEY=$GROQ_API_KEY -p 8000:8000 visloc-rag
```

## Tests

`pytest -q` runs offline: a fake model client checks that uncited or unsupported facts are hidden,
that JSON replies are parsed safely, and that name matching works with aliases and accents.
