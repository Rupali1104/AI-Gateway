"""Request logger — appends JSON lines to logs/requests.jsonl."""

import json
import os
from datetime import datetime, timezone

LOG_FILE = os.path.join(os.path.dirname(__file__), "logs", "requests.jsonl")


def _ensure_dir():
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)


def log(entry: dict) -> None:
    _ensure_dir()
    entry["timestamp"] = datetime.now(timezone.utc).isoformat()
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def read_all() -> list[dict]:
    _ensure_dir()
    if not os.path.exists(LOG_FILE):
        return []
    with open(LOG_FILE, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
