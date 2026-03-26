# AI Gateway

A smart traffic controller that routes every prompt to the right LLM — and proves it made the right call.

---

## Models Used

| Label | Model | Handles |
|---|---|---|
| **Fast model** | Groq — Llama 3.1 8B Instant (free tier) | Simple queries, factual Q&A, short summaries, low-complexity tasks |
| **Capable model** | Google Gemini 1.5 Flash (free tier) | Reasoning, code generation, multi-step tasks, complex analysis |

Every request is tagged with which model handled it and why.

---

## Setup (5 commands)

```bash
git clone <repo-url> && cd ai-gateway
pip install -r requirements.txt
cp .env.example .env          # fill in GROQ_API_KEY and GEMINI_API_KEY
uvicorn main:app --reload     # gateway on http://localhost:8000
streamlit run dashboard.py    # log viewer on http://localhost:8501
```

---

## Routing Model

The routing model is a **rule-based weighted scoring model** with 5 documented features:

| Feature | Weight | What it measures |
|---|---|---|
| `prompt_length` | 0.20 | Word count — longer prompts tend to be more complex |
| `keyword_complexity` | 0.35 | Presence of reasoning/code/analysis keywords (highest weight) |
| `question_count` | 0.15 | Multiple questions signal multi-step tasks |
| `code_signal` | 0.20 | Code tokens (`def`, `class`, `import`, backticks, etc.) |
| `sentence_complexity` | 0.10 | Average words per sentence |

**Decision boundary:** weighted score ≥ 0.40 → Capable model, else → Fast model.

Each decision includes a confidence score (0.5–1.0) and a human-readable reason string logged with every request.

---

## PoC — Standalone Routing Evaluator

Runs independently of the server. No API keys needed.

```bash
python poc.py                          # uses test_suite.json (20 prompts)
python poc.py --file my_prompts.json   # custom JSON: [{prompt, label}]
python poc.py --file prompts.csv       # CSV with columns: prompt, label
```

Labels accepted: `simple` / `complex` (or `fast` / `capable`).

Output: per-prompt decision table + accuracy, false positive rate, false negative rate.

---

## API

### POST /chat

```json
{ "prompt": "your question here" }
```

Response:
```json
{
  "response": "...",
  "model": "Fast model (Groq llama-3.1-8b-instant)",
  "routing_reason": "score=0.123 (threshold=0.4); top signals: ...",
  "routing_score": 0.123,
  "routing_confidence": 0.872,
  "latency_ms": 412.3,
  "cache_hit": false,
  "tokens": 187
}
```

### GET /health

Returns server status and current cache size.

---

## Project Structure

```
ai-gateway/
├── main.py          # FastAPI gateway server
├── router.py        # Routing model (weighted scoring)
├── cache.py         # TF-IDF cosine similarity cache
├── logger.py        # JSON-lines request logger
├── dashboard.py     # Streamlit log viewer
├── poc.py           # Standalone routing model evaluator
├── test_suite.json  # 20-prompt test suite with ground-truth labels
├── requirements.txt
├── .env.example
└── logs/
    └── requests.jsonl
```

---

## Research Questions (Summary)

1. **Routing accuracy** — run `python poc.py` for live results on the 20-prompt suite.
2. **Cost comparison** — Capable model (Gemini) costs ~$0.075/1M input tokens; Fast model (Groq) is free tier. Smart routing avoids Capable model calls for ~50% of prompts.
3. **Failure cases** — see PoC output; hardest mis-routes are mid-complexity prompts that score near the 0.40 boundary.
4. **Cache hit rate** — default threshold 0.82; tune `CACHE_THRESHOLD` in `cache.py`. Lower threshold → more hits, risk of stale answers.
5. **What to change** — train a lightweight logistic regression classifier on a larger labelled dataset to replace the hand-tuned weights.
