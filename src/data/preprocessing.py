from __future__ import annotations

import re

URL_RE = re.compile(r"https?://\\S+|www\\.\\S+")
MENTION_RE = re.compile(r"@[A-Za-z0-9_]+")
SPACE_RE = re.compile(r"\\s+")


def clean_text(text: str, remove_urls: bool = True, remove_mentions: bool = True) -> str:
    value = text if isinstance(text, str) else str(text)
    if remove_urls:
        value = URL_RE.sub("", value)
    if remove_mentions:
        value = MENTION_RE.sub("", value)
    return SPACE_RE.sub(" ", value).strip()

