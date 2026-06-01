"""Helpers for ordinal/rank-style factual questions."""

from __future__ import annotations

import re


_ORDINAL_WORDS = {
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "fifth": 5,
    "sixth": 6,
    "seventh": 7,
    "eighth": 8,
    "ninth": 9,
    "tenth": 10,
}


def requested_rank(text: str) -> int | None:
    """Return requested rank for phrases like '3rd top scorer' or 'third place'."""
    if not text:
        return None
    lowered = text.lower()
    numeric = re.search(
        r"\b([1-9]|1[0-9]|20)(?:st|nd|rd|th)?\s+"
        r"(?:top\s+)?(?:scorer|goal\s*scorer|goalscorer|leader|rank|place|position|highest|best)\b",
        lowered,
    )
    if numeric:
        return int(numeric.group(1))
    for word, rank in _ORDINAL_WORDS.items():
        if re.search(
            rf"\b{word}\s+(?:top\s+)?(?:scorer|goal\s*scorer|goalscorer|leader|rank|place|position|highest|best)\b",
            lowered,
        ):
            return rank
    return None


def ordinal_label(rank: int) -> str:
    if 10 <= rank % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(rank % 10, "th")
    return f"{rank}{suffix}"
