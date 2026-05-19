"""Runtime-tunable settings persisted to settings.json.

These shadow the constants in config.py — they're read at request time so the
user can tweak them from the UI without restarting the server. If settings.json
is missing or malformed, the defaults below apply.
"""

import json
import threading
from pathlib import Path
from typing import Any

SETTINGS_FILE = Path(__file__).parent / "settings.json"
_lock = threading.Lock()

DEFAULTS: dict[str, Any] = {
    "cache_ttl_seconds": 24 * 3600,
    "history_turn_limit": 5,
    "verification_enabled": True,
    "fact_extraction_enabled": True,
}


def _read() -> dict[str, Any]:
    if not SETTINGS_FILE.exists():
        return dict(DEFAULTS)
    try:
        with SETTINGS_FILE.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, dict):
            return dict(DEFAULTS)
        merged = dict(DEFAULTS)
        for key, value in data.items():
            if key in DEFAULTS:
                merged[key] = value
        return merged
    except Exception:
        return dict(DEFAULTS)


def _write(data: dict[str, Any]) -> None:
    tmp = SETTINGS_FILE.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
    tmp.replace(SETTINGS_FILE)


def get_all() -> dict[str, Any]:
    with _lock:
        return _read()


def get(key: str) -> Any:
    with _lock:
        return _read().get(key, DEFAULTS.get(key))


def update(patch: dict[str, Any]) -> dict[str, Any]:
    """Validate types against DEFAULTS and persist. Returns the new full dict."""
    with _lock:
        data = _read()
        for key, value in patch.items():
            if key not in DEFAULTS:
                continue
            default_val = DEFAULTS[key]
            if isinstance(default_val, bool):
                data[key] = bool(value)
            elif isinstance(default_val, int):
                try:
                    data[key] = max(0, int(value))
                except (TypeError, ValueError):
                    pass
        _write(data)
        return data


def reset() -> dict[str, Any]:
    with _lock:
        _write(dict(DEFAULTS))
        return dict(DEFAULTS)
