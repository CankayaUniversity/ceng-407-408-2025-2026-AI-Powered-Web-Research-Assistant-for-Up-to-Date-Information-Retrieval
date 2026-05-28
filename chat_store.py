import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CHATS_FILE = Path(__file__).parent / "chats.json"
_lock = threading.Lock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read() -> dict:
    if not CHATS_FILE.exists():
        return {"chats": []}
    try:
        with CHATS_FILE.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
            if not isinstance(data, dict) or "chats" not in data:
                return {"chats": []}
            return data
    except Exception:
        return {"chats": []}


def _write(data: dict) -> None:
    tmp_path = CHATS_FILE.with_suffix(".tmp")
    with tmp_path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
    tmp_path.replace(CHATS_FILE)


def _truncate(text: str, limit: int = 80) -> str:
    cleaned = (text or "").strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit].rstrip() + "…"


def _summary(chat: dict) -> dict:
    turns = chat.get("turns", []) or []
    last_question = turns[-1].get("question", "") if turns else ""
    return {
        "id": chat["id"],
        "title": chat.get("title") or "Untitled",
        "created_at": chat.get("created_at"),
        "updated_at": chat.get("updated_at"),
        "turn_count": len(turns),
        "preview": _truncate(last_question, 100),
    }


def list_chats() -> list[dict]:
    with _lock:
        data = _read()
        items = [_summary(chat) for chat in data["chats"]]
        items.sort(key=lambda c: c["updated_at"] or "", reverse=True)
        return items


def search_chats(query: str) -> list[dict]:
    query_lower = (query or "").strip().lower()
    if not query_lower:
        return list_chats()

    with _lock:
        data = _read()
        matches = []
        for chat in data["chats"]:
            title = (chat.get("title") or "").lower()
            hit = query_lower in title
            if not hit:
                for turn in chat.get("turns", []) or []:
                    question = (turn.get("question") or "").lower()
                    answer = (turn.get("answer") or "").lower()
                    if query_lower in question or query_lower in answer:
                        hit = True
                        break
            if hit:
                matches.append(_summary(chat))

        matches.sort(key=lambda c: c["updated_at"] or "", reverse=True)
        return matches


def get_chat(chat_id: str) -> dict | None:
    with _lock:
        data = _read()
        for chat in data["chats"]:
            if chat["id"] == chat_id:
                return chat
        return None


def create_chat(initial_question: str | None = None) -> dict:
    with _lock:
        data = _read()
        chat = {
            "id": uuid.uuid4().hex[:12],
            "title": _truncate(initial_question or "New research", 80),
            "created_at": _now(),
            "updated_at": _now(),
            "turns": [],
        }
        data["chats"].append(chat)
        _write(data)
        return chat


def append_turn(chat_id: str, turn: dict[str, Any]) -> dict | None:
    with _lock:
        data = _read()
        for chat in data["chats"]:
            if chat["id"] == chat_id:
                stored_turn = dict(turn)
                stored_turn["timestamp"] = _now()
                chat["turns"].append(stored_turn)
                chat["updated_at"] = _now()
                current_title = (chat.get("title") or "").strip()
                if (not current_title or current_title == "New research") and turn.get("question"):
                    chat["title"] = _truncate(turn["question"], 80)
                _write(data)
                return chat
        return None


def replace_last_turn(chat_id: str, new_turn: dict[str, Any]) -> dict | None:
    """Atomically drop the most recent turn and append `new_turn` in its place.
    Used by the regenerate flow so a regenerated answer overwrites the original
    rather than appending a duplicate. Returns the updated chat, or None if the
    chat doesn't exist."""
    with _lock:
        data = _read()
        for chat in data["chats"]:
            if chat["id"] == chat_id:
                turns = chat.get("turns", []) or []
                if turns:
                    turns.pop()
                stored_turn = dict(new_turn)
                stored_turn["timestamp"] = _now()
                turns.append(stored_turn)
                chat["turns"] = turns
                chat["updated_at"] = _now()
                # Title intentionally NOT updated — keep the original.
                _write(data)
                return chat
        return None


def update_title(chat_id: str, title: str) -> dict | None:
    with _lock:
        data = _read()
        for chat in data["chats"]:
            if chat["id"] == chat_id:
                chat["title"] = _truncate(title, 120)
                chat["updated_at"] = _now()
                _write(data)
                return chat
        return None


def delete_chat(chat_id: str) -> bool:
    with _lock:
        data = _read()
        before = len(data["chats"])
        data["chats"] = [c for c in data["chats"] if c["id"] != chat_id]
        if len(data["chats"]) == before:
            return False
        _write(data)
        return True


def delete_all() -> int:
    with _lock:
        data = _read()
        count = len(data["chats"])
        data["chats"] = []
        _write(data)
        return count
