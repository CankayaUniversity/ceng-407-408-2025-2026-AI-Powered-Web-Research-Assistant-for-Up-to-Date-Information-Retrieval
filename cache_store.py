import json
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CACHE_FILE = Path(__file__).parent / "cache.json"
_lock = threading.Lock()

_PUNCT_RE = re.compile(r"[^\w\s]+", re.UNICODE)
_WS_RE = re.compile(r"\s+")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _now_ts() -> float:
    return datetime.now(timezone.utc).timestamp()


def _read() -> dict:
    if not CACHE_FILE.exists():
        return {"entries": {}}
    try:
        with CACHE_FILE.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
            if not isinstance(data, dict) or "entries" not in data:
                return {"entries": {}}
            return data
    except Exception:
        return {"entries": {}}


def _write(data: dict) -> None:
    tmp_path = CACHE_FILE.with_suffix(".tmp")
    with tmp_path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
    tmp_path.replace(CACHE_FILE)


def normalize_question(text: str) -> str:
    cleaned = (text or "").lower().strip()
    cleaned = _PUNCT_RE.sub(" ", cleaned)
    cleaned = _WS_RE.sub(" ", cleaned).strip()
    return cleaned


def _build_key(model_key: str, question: str) -> str:
    return f"{model_key}::{normalize_question(question)}"


def get(model_key: str, question: str, ttl_seconds: int) -> dict[str, Any] | None:
    normalized = normalize_question(question)
    if not normalized:
        return None
    key = _build_key(model_key, question)
    with _lock:
        data = _read()
        entry = data["entries"].get(key)
        if not entry:
            return None
        cached_ts = entry.get("cached_ts", 0)
        if _now_ts() - cached_ts > ttl_seconds:
            return None
        return entry


def put(model_key: str, question: str, payload: dict[str, Any]) -> None:
    normalized = normalize_question(question)
    if not normalized:
        return
    key = _build_key(model_key, question)
    with _lock:
        data = _read()
        entry = dict(payload)
        entry["model_key"] = model_key
        entry["original_question"] = question
        entry["cached_at"] = _now_iso()
        entry["cached_ts"] = _now_ts()
        data["entries"][key] = entry
        _write(data)


def clear() -> None:
    with _lock:
        _write({"entries": {}})


def stats() -> dict[str, Any]:
    with _lock:
        data = _read()
        return {"entry_count": len(data["entries"])}
