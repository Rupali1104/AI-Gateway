"""
poc.py — Standalone routing model evaluator.

Usage:
    python poc.py                          # uses default test_suite.json
    python poc.py --file my_prompts.json   # custom file
    python poc.py --file prompts.csv       # CSV with columns: prompt, label

Output: per-prompt decision table + accuracy / FP / FN summary.
Runs independently of the gateway server (no API keys needed).
"""

import argparse
import csv
import json
import os
import sys
import time

# Allow running from any directory
sys.path.insert(0, os.path.dirname(__file__))
from router import route

# ── Label normalisation ───────────────────────────────────────────────────────

def _normalise(label: str) -> str:
    """Map ground-truth label to 'fast' or 'capable'."""
    l = label.strip().lower()
    if l in ("simple", "fast", "easy"):
        return "fast"
    if l in ("complex", "capable", "hard", "difficult"):
        return "capable"
    raise ValueError(f"Unknown label '{label}'. Use simple/complex or fast/capable.")


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_json(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return [{"prompt": d["prompt"], "label": _normalise(d["label"])} for d in data]


def load_csv(path: str) -> list[dict]:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({"prompt": row["prompt"], "label": _normalise(row["label"])})
    return rows


# ── Cost model ───────────────────────────────────────────────────────────────
# Groq Llama 3.1 8B  — free tier (no cost)
# Gemini 2.5 Flash   — $0.075 per 1M input tokens, $0.30 per 1M output tokens
GEMINI_INPUT_COST_PER_TOKEN  = 0.075 / 1_000_000
GEMINI_OUTPUT_COST_PER_TOKEN = 0.300 / 1_000_000
AVG_OUTPUT_TOKENS = 300  # conservative estimate per response

def _tokens(prompt: str) -> int:
    """Approximate token count (words * 1.3)."""
    return int(len(prompt.split()) * 1.3)

def _gemini_cost(input_tokens: int) -> float:
    return (input_tokens * GEMINI_INPUT_COST_PER_TOKEN +
            AVG_OUTPUT_TOKENS * GEMINI_OUTPUT_COST_PER_TOKEN)


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(items: list[dict]) -> None:
    COL = "{:<4} {:<55} {:<10} {:<10} {:<8} {:<7} {}"
    print("\n" + "=" * 110)
    print(COL.format("ID", "Prompt (truncated)", "GT Label", "Predicted", "Conf", "Score", "Result"))
    print("=" * 110)

    correct = fp = fn = 0
    smart_cost = 0.0
    always_capable_cost = 0.0
    smart_capable_calls = 0
    t_start = time.perf_counter()

    for i, item in enumerate(items, 1):
        prompt = item["prompt"]
        gt = item["label"]
        tokens = _tokens(prompt)

        result = route(prompt)
        pred = result["model"]
        conf = result["confidence"]
        score = result["score"]

        # Cost tracking
        always_capable_cost += _gemini_cost(tokens)
        if pred == "capable":
            smart_cost += _gemini_cost(tokens)
            smart_capable_calls += 1
        # fast model is free — no cost added

        is_correct = pred == gt
        if is_correct:
            correct += 1
            verdict = "OK"
        else:
            verdict = "FAIL"
            if gt == "capable" and pred == "fast":
                fp += 1
            else:
                fn += 1

        snippet = (prompt[:52] + "...") if len(prompt) > 55 else prompt
        print(COL.format(i, snippet, gt, pred, f"{conf:.3f}", f"{score:.3f}", verdict))

    elapsed = time.perf_counter() - t_start
    total = len(items)
    accuracy = correct / total * 100
    cost_saved = always_capable_cost - smart_cost
    cost_reduction = (cost_saved / always_capable_cost * 100) if always_capable_cost else 0

    print("=" * 110)
    print(f"\nROUTING SUMMARY")
    print(f"  Total prompts      : {total}")
    print(f"  Correct            : {correct}")
    print(f"  Accuracy           : {accuracy:.1f}%")
    print(f"  False Positives    : {fp}  (complex -> fast   -- quality risk)")
    print(f"  False Negatives    : {fn}  (simple  -> capable -- cost waste)")
    print(f"  Routing latency    : {elapsed*1000:.1f} ms total  ({elapsed*1000/total:.2f} ms/prompt)")
    print()
    print(f"COST COMPARISON (estimated)")
    print(f"  Always-Capable cost : ${always_capable_cost:.6f}  ({total} Gemini calls)")
    print(f"  Smart routing cost  : ${smart_cost:.6f}  ({smart_capable_calls} Gemini + {total-smart_capable_calls} Groq free)")
    print(f"  Cost saved          : ${cost_saved:.6f}")
    print(f"  Cost reduction      : {cost_reduction:.1f}%")
    print()

    if accuracy >= 75:
        print("  PASS: Routing accuracy meets success bar (>75%)")
    else:
        print("  WARN: Below success bar (<75% accuracy)")
    if cost_reduction >= 25:
        print("  PASS: Cost reduction meets success bar (>25%)")
    else:
        print("  WARN: Cost reduction below success bar (<25%)")
    print()


def cache_analysis(items: list[dict]) -> None:
    """Show hit rate at different similarity thresholds using prompts as a simulated cache."""
    import math, re
    from collections import Counter

    def tokenize(text):
        return re.findall(r"\b\w+\b", text.lower())

    def cosine(a, b):
        common = set(a) & set(b)
        if not common:
            return 0.0
        dot = sum(a[t] * b[t] for t in common)
        ma = math.sqrt(sum(v*v for v in a.values()))
        mb = math.sqrt(sum(v*v for v in b.values()))
        return dot / (ma * mb) if ma and mb else 0.0

    def tf_vec(tokens):
        c = Counter(tokens)
        total = len(tokens) or 1
        return {t: v/total for t, v in c.items()}

    # Simulate: first half = cache, second half = queries
    cache_prompts = [it["prompt"] for it in items[:10]]
    query_prompts = [it["prompt"] for it in items[10:]]
    # Add paraphrased near-duplicates to queries for realistic hit testing
    near_dupes = [
        "What is the capital city of France?",
        "Who is the author of Romeo and Juliet?",
        "Write a Python binary search function with time complexity explanation.",
        "Compare GraphQL and REST APIs for microservices.",
        "Explain supervised vs unsupervised machine learning with examples.",
    ]
    all_queries = query_prompts + near_dupes

    thresholds = [0.60, 0.70, 0.82, 0.95]
    print("\nCACHE THRESHOLD ANALYSIS")
    print("=" * 60)
    print(f"  Cache size : {len(cache_prompts)} entries")
    print(f"  Queries    : {len(all_queries)} prompts")
    print()
    print(f"  {'Threshold':<12} {'Hits':<8} {'Misses':<8} {'Hit Rate':<12} {'Risk'}")
    print("  " + "-" * 55)

    cache_vecs = [tf_vec(tokenize(p)) for p in cache_prompts]

    for thresh in thresholds:
        hits = 0
        for q in all_queries:
            q_vec = tf_vec(tokenize(q))
            best = max((cosine(q_vec, cv) for cv in cache_vecs), default=0.0)
            if best >= thresh:
                hits += 1
        misses = len(all_queries) - hits
        hit_rate = hits / len(all_queries) * 100
        risk = "High (stale answers likely)" if thresh < 0.70 else \
               "Medium" if thresh < 0.82 else \
               "Low (safe)" if thresh < 0.95 else "Very low (too strict)"
        marker = " <-- chosen" if thresh == 0.82 else ""
        print(f"  {thresh:<12} {hits:<8} {misses:<8} {hit_rate:<11.1f}%  {risk}{marker}")
    print()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="AI Gateway -- Routing Model PoC Evaluator")
    default_suite = os.path.join(os.path.dirname(__file__), "test_suite.json")
    parser.add_argument("--file", default=default_suite, help="Path to JSON or CSV test suite")
    parser.add_argument("--failures", action="store_true", help="Show 3 hardest near-miss cases")
    parser.add_argument("--cache-analysis", action="store_true", help="Show cache threshold analysis")
    args = parser.parse_args()

    path = args.file
    if not os.path.exists(path):
        print(f"Error: file not found: {path}")
        sys.exit(1)

    items = load_csv(path) if path.endswith(".csv") else load_json(path)
    print(f"Loaded {len(items)} prompts from: {path}")
    evaluate(items)

    if args.failures:
        print("\n3 HARDEST NEAR-MISS CASES (closest score to threshold=0.35)")
        print("=" * 90)
        from router import THRESHOLD
        scored = []
        for item in items:
            r = route(item["prompt"])
            scored.append((abs(r["score"] - THRESHOLD), item["prompt"], item["label"], r))
        scored.sort(key=lambda x: x[0])
        for rank, (dist, prompt, gt, r) in enumerate(scored[:3], 1):
            print(f"\n#{rank} | Distance from threshold: {dist:.4f}")
            print(f"  Prompt    : {prompt[:90]}")
            print(f"  GT label  : {gt}")
            print(f"  Predicted : {r['model']} (score={r['score']:.3f}, conf={r['confidence']:.3f})")
            print(f"  Top signal: {r['reason'].split('top signals:')[-1].strip()}")
            print(f"  Root cause: Score near boundary -- small keyword change would flip decision")
        print()

    if args.cache_analysis:
        cache_analysis(items)


if __name__ == "__main__":
    main()
