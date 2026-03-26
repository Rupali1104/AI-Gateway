"""
Cache layer — TF-IDF cosine similarity for near-duplicate prompt detection.
Persists to disk so cache survives server restarts.
"""

import json
import math
import os
import re
from collections import Counter

CACHE_THRESHOLD = 0.82
CACHE_FILE = os.path.join(os.path.dirname(__file__), "logs", "cache.json")


def _load() -> list[dict]:
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, encoding="utf-8") as f:
            return json.load(f)
    return []


def _save(store: list[dict]) -> None:
    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(store, f)


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


def _tfidf_vector(tokens: list[str], idf: dict) -> dict:
    tf = Counter(tokens)
    total = len(tokens) or 1
    # When IDF is available use it, but floor at 0.1 so common words aren't zeroed out
    return {t: (c / total) * max(idf.get(t, 1.0), 0.1) for t, c in tf.items()}


def _cosine(a: dict, b: dict) -> float:
    common = set(a) & set(b)
    if not common:
        return 0.0
    dot = sum(a[t] * b[t] for t in common)
    mag_a = math.sqrt(sum(v * v for v in a.values()))
    mag_b = math.sqrt(sum(v * v for v in b.values()))
    return dot / (mag_a * mag_b) if mag_a and mag_b else 0.0


def _build_idf(store: list[dict]) -> dict:
    if not store:
        return {}
    N = len(store)
    df: Counter = Counter()
    for entry in store:
        for t in set(_tokenize(entry["prompt"])):
            df[t] += 1
    return {t: math.log(N / c) for t, c in df.items()}


def lookup(prompt: str) -> dict | None:
    store = _load()
    if not store:
        return None
    idf = _build_idf(store)
    q_vec = _tfidf_vector(_tokenize(prompt), idf)
    best_score, best_entry = 0.0, None
    for entry in store:
        c_vec = _tfidf_vector(_tokenize(entry["prompt"]), idf)
        score = _cosine(q_vec, c_vec)
        if score > best_score:
            best_score, best_entry = score, entry
    if best_score >= CACHE_THRESHOLD:
        return {"entry": best_entry, "similarity": round(best_score, 4)}
    return None


def store(prompt: str, response: str, metadata: dict) -> None:
    data = _load()
    data.append({"prompt": prompt, "response": response, **metadata})
    _save(data)


def size() -> int:
    return len(_load())
