"""
AI Gateway — POST /chat
Routes prompts to Fast model (Groq Llama 3.1 8B) or
Capable model (Google Gemini 1.5 Flash) based on routing model decision.
"""

import os
import time

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import cache
import logger
from router import route

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

GROQ_MODEL = "llama-3.1-8b-instant"
GEMINI_MODELS = [
    "gemini-2.0-flash",
    "gemini-2.0-flash-001",
    "gemini-2.5-flash",
]

app = FastAPI(title="AI Gateway")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class ChatRequest(BaseModel):
    prompt: str


# ── LLM callers ──────────────────────────────────────────────────────────────

async def call_groq(prompt: str) -> tuple[str, int]:
    """Returns (response_text, total_tokens)."""
    url = "https://api.groq.com/openai/v1/chat/completions"
    payload = {
        "model": GROQ_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1024,
    }
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(url, json=payload, headers={"Authorization": f"Bearer {GROQ_API_KEY}"})
        r.raise_for_status()
        data = r.json()
    text = data["choices"][0]["message"]["content"]
    tokens = data.get("usage", {}).get("total_tokens", 0)
    return text, tokens


async def call_gemini(prompt: str) -> tuple[str, int, str]:
    """Try each model in GEMINI_MODELS until one works. Returns (text, tokens, model_used)."""
    last_err = None
    for model in GEMINI_MODELS:
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model}:generateContent?key={GEMINI_API_KEY}"
        )
        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                r = await client.post(url, json=payload)
                if r.status_code == 404:
                    last_err = f"404 on {model}"
                    continue
                r.raise_for_status()
                data = r.json()
            text = data["candidates"][0]["content"]["parts"][0]["text"]
            meta = data.get("usageMetadata", {})
            tokens = meta.get("totalTokenCount", 0)
            return text, tokens, model
        except Exception as e:
            last_err = str(e)
            continue
    raise RuntimeError(f"All Gemini models failed. Last error: {last_err}")


# ── Main endpoint ─────────────────────────────────────────────────────────────

@app.post("/chat")
async def chat(req: ChatRequest):
    prompt = req.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="prompt cannot be empty")

    t0 = time.perf_counter()

    # 1. Cache check
    cached = cache.lookup(prompt)
    if cached:
        latency_ms = round((time.perf_counter() - t0) * 1000, 1)
        entry = cached["entry"]
        log_entry = {
            "prompt_snippet": prompt[:80],
            "model": entry["model"],
            "routing_reason": entry["routing_reason"],
            "latency_ms": latency_ms,
            "cache_hit": True,
            "similarity": cached["similarity"],
            "tokens": entry.get("tokens", 0),
        }
        logger.log(log_entry)
        return {
            "response": entry["response"],
            "model": entry["model"],
            "routing_reason": entry["routing_reason"],
            "latency_ms": latency_ms,
            "cache_hit": True,
            "similarity": cached["similarity"],
        }

    # 2. Route
    t_route = time.perf_counter()
    routing = route(prompt)
    routing_latency_ms = round((time.perf_counter() - t_route) * 1000, 3)
    model_label = routing["model"]

    # 3. Call LLM
    try:
        if model_label == "fast":
            response_text, tokens = await call_groq(prompt)
            model_name = f"Fast model (Groq {GROQ_MODEL})"
        else:
            response_text, tokens, gemini_model_used = await call_gemini(prompt)
            model_name = f"Capable model (Gemini {gemini_model_used})"
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"LLM call failed: {exc}")

    latency_ms = round((time.perf_counter() - t0) * 1000, 1)

    # 4. Store in cache
    cache.store(prompt, response_text, {
        "model": model_name,
        "routing_reason": routing["reason"],
        "tokens": tokens,
    })

    # 5. Log
    log_entry = {
        "prompt_snippet": prompt[:80],
        "model": model_name,
        "routing_reason": routing["reason"],
        "routing_score": routing["score"],
        "routing_confidence": routing["confidence"],
        "routing_latency_ms": routing_latency_ms,
        "latency_ms": latency_ms,
        "cache_hit": False,
        "tokens": tokens,
    }
    logger.log(log_entry)

    return {
        "response": response_text,
        "model": model_name,
        "routing_reason": routing["reason"],
        "routing_score": routing["score"],
        "routing_confidence": routing["confidence"],
        "routing_latency_ms": routing_latency_ms,
        "latency_ms": latency_ms,
        "cache_hit": False,
        "tokens": tokens,
    }


@app.get("/health")
def health():
    return {"status": "ok", "cache_size": cache.size()}
