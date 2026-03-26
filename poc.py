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


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(items: list[dict]) -> None:
    COL = "{:<4} {:<55} {:<10} {:<10} {:<8} {:<7} {}"
    print("\n" + "=" * 110)
    print(COL.format("ID", "Prompt (truncated)", "GT Label", "Predicted", "Conf", "Score", "Result"))
    print("=" * 110)

    correct = fp = fn = 0
    t_start = time.perf_counter()

    for i, item in enumerate(items, 1):
        prompt = item["prompt"]
        gt = item["label"]

        result = route(prompt)
        pred = result["model"]
        conf = result["confidence"]
        score = result["score"]

        is_correct = pred == gt
        if is_correct:
            correct += 1
            verdict = "OK"
        else:
            verdict = "FAIL"
            if gt == "capable" and pred == "fast":
                fp += 1   # complex sent to fast (false positive for "fast")
            else:
                fn += 1   # simple sent to capable (false negative for "fast")

        snippet = (prompt[:52] + "...") if len(prompt) > 55 else prompt
        print(COL.format(i, snippet, gt, pred, f"{conf:.3f}", f"{score:.3f}", verdict))

    elapsed = time.perf_counter() - t_start
    total = len(items)
    accuracy = correct / total * 100

    print("=" * 110)
    print(f"\nSUMMARY")
    print(f"  Total prompts   : {total}")
    print(f"  Correct         : {correct}")
    print(f"  Accuracy        : {accuracy:.1f}%")
    print(f"  False Positives : {fp}  (complex -> fast   -- quality risk)")
    print(f"  False Negatives : {fn}  (simple  -> capable -- cost waste)")
    print(f"  Elapsed time    : {elapsed*1000:.1f} ms  ({elapsed*1000/total:.1f} ms/prompt)")
    print()

    if accuracy >= 75:
        print("  PASS: Meets success bar (>75% accuracy)")
    else:
        print("  WARN: Below success bar (<75% accuracy)")
    print()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="AI Gateway — Routing Model PoC Evaluator")
    default_suite = os.path.join(os.path.dirname(__file__), "test_suite.json")
    parser.add_argument("--file", default=default_suite, help="Path to JSON or CSV test suite")
    args = parser.parse_args()

    path = args.file
    if not os.path.exists(path):
        print(f"Error: file not found: {path}")
        sys.exit(1)

    if path.endswith(".csv"):
        items = load_csv(path)
    else:
        items = load_json(path)

    print(f"Loaded {len(items)} prompts from: {path}")
    evaluate(items)


if __name__ == "__main__":
    main()
