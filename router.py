"""
Routing Model — Rule-based scoring model with 5 documented features.

Features:
  F1: prompt_length       — longer prompts tend to be more complex
  F2: keyword_complexity  — presence of reasoning/code/analysis keywords
  F3: question_count      — multi-step questions signal complexity
  F4: code_signal         — code-related tokens (def, class, import, etc.)
  F5: sentence_complexity — average words per sentence

Decision boundary: weighted score >= THRESHOLD → Capable model
"""

import re

# Feature weights (sum = 1.0)
WEIGHTS = {
    "prompt_length": 0.20,
    "keyword_complexity": 0.35,
    "question_count": 0.15,
    "code_signal": 0.20,
    "sentence_complexity": 0.10,
}

THRESHOLD = 0.35  # score >= threshold -> Capable model

COMPLEX_KEYWORDS = {
    "explain", "analyze", "analyse", "compare", "design", "implement",
    "algorithm", "optimize", "debug", "refactor", "architecture", "evaluate",
    "reasoning", "proof", "derive", "calculate", "predict", "classify",
    "summarize", "translate", "generate code", "write a function", "write a program",
    "step by step", "multi-step", "trade-off", "tradeoff", "research",
    "hypothesis", "inference", "neural", "machine learning", "deep learning",
    "difference between", "when would you", "how does", "why does", "ethical",
    "implications", "surveillance", "bias", "regulatory", "window function",
    "microservices", "production", "rate-limit", "rate limit", "data structure",
    "time complexity", "space complexity", "trade off", "outperform",
    "attention mechanism", "transformer", "supervised", "unsupervised",
    "rest api", "endpoints", "request/response", "http methods", "system design",
    "scalab", "distributed", "concurren", "fault toleran",
    "linked list", "write a class", "data structure", "implement a",
    "insert", "delete", "search method", "sorting", "recursion",
}

CODE_TOKENS = {"def ", "class ", "import ", "return ", "for ", "while ", "if __name__",
               "```", "function", "async ", "await ", "lambda ", "->", "=>"}


def _score_prompt_length(prompt: str) -> float:
    """Normalised length score: 0 at <=30 words, 1.0 at >=150 words."""
    words = len(prompt.split())
    return min(max((words - 30) / 120, 0.0), 1.0)


def _score_keyword_complexity(prompt: str) -> float:
    """Fraction of complex keywords found (capped at 1.0)."""
    lower = prompt.lower()
    hits = sum(1 for kw in COMPLEX_KEYWORDS if kw in lower)
    return min(hits / 3, 1.0)


def _score_question_count(prompt: str) -> float:
    """Multiple questions → more complex. Score 0 for 0-1, scales to 1.0 at 4+."""
    count = prompt.count("?")
    return min(max((count - 1) / 3, 0.0), 1.0)


def _score_code_signal(prompt: str) -> float:
    """Any code token present → 1.0, else 0."""
    lower = prompt.lower()
    return 1.0 if any(tok in lower for tok in CODE_TOKENS) else 0.0


def _score_sentence_complexity(prompt: str) -> float:
    """Average words per sentence. 0 at <=8, 1.0 at >=25."""
    sentences = [s.strip() for s in re.split(r"[.!?]", prompt) if s.strip()]
    if not sentences:
        return 0.0
    avg = sum(len(s.split()) for s in sentences) / len(sentences)
    return min(max((avg - 8) / 17, 0.0), 1.0)


def route(prompt: str) -> dict:
    """
    Returns:
        model: "fast" | "capable"
        confidence: float 0-1 (distance from threshold, mapped to 0.5-1.0)
        score: raw weighted score
        features: individual feature scores
        reason: human-readable explanation
    """
    features = {
        "prompt_length": _score_prompt_length(prompt),
        "keyword_complexity": _score_keyword_complexity(prompt),
        "question_count": _score_question_count(prompt),
        "code_signal": _score_code_signal(prompt),
        "sentence_complexity": _score_sentence_complexity(prompt),
    }

    score = sum(WEIGHTS[f] * v for f, v in features.items())
    model = "capable" if score >= THRESHOLD else "fast"

    # Confidence: how far from threshold, mapped to [0.5, 1.0]
    distance = abs(score - THRESHOLD)
    confidence = round(0.5 + min(distance / THRESHOLD, 0.5), 3)

    # Build reason string from top contributing features
    top = sorted(features.items(), key=lambda x: -x[1])
    active = [f"{k}={v:.2f}" for k, v in top if v > 0][:2]
    reason = f"score={score:.3f} (threshold={THRESHOLD}); top signals: {', '.join(active) if active else 'none'}"

    return {
        "model": model,
        "confidence": confidence,
        "score": round(score, 4),
        "features": {k: round(v, 3) for k, v in features.items()},
        "reason": reason,
    }
