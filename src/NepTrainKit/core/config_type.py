"""Helpers for managing `Config_type` tags in a stable, compact format.

`Config_type` convention
------------------------
- Stored in ``atoms.info["Config_type"]`` as a short tag string.
- Tags are delimited by ``|`` (pipe), e.g. ``neptrainkit|Comp(Co0.33Ni0.67)|Occ(E,s=123)``.
- Tags must be short and stable (human-readable summary only).
- No single/double quotes are allowed inside tags (they are stripped).
- Avoid embedding ``|`` inside a tag (it is replaced by ``_``).
- Output is de-duplicated and truncated (default ``max_len=200``) to keep exports tidy.

For card implementations: always use ``append_config_tag(atoms, tag)`` instead of manual
string concatenation.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any


_SPLIT_RE = re.compile(r"[|\s]+")
_QUOTE_RE = re.compile(r"[\"']")
_WS_RE = re.compile(r"\s+")


def sanitize_config_tag(tag: str, *, sep: str = "|") -> str:
    """Sanitize a tag so it is safe inside a sep-delimited `Config_type` string."""
    tag = str(tag or "")
    tag = _QUOTE_RE.sub("", tag)
    tag = tag.replace(sep, "_")
    tag = _WS_RE.sub("", tag)
    return tag.strip()


def append_config_tag(
    atoms: Any,
    tag: str,
    *,
    sep: str = "|",
    max_len: int = 200,
    dedupe: bool = True,
) -> str:
    """Append a tag into ``atoms.info['Config_type']`` using a stable separator.

    Notes
    -----
    - Keeps EXTXYZ-friendly plain strings (no forced JSON).
    - Canonicalizes legacy space-delimited Config_type into ``sep``-delimited tags.
    - Avoid putting the separator character inside ``tag``.
    """
    info = getattr(atoms, "info", None)
    if info is None:
        raise TypeError("append_config_tag expects an object with an `.info` dict-like attribute.")

    tag = sanitize_config_tag(tag, sep=sep)
    if not tag:
        return str(info.get("Config_type", "") or "")

    current = info.get("Config_type", "")
    if current is None:
        current_text = ""
    elif isinstance(current, (list, tuple, set)):
        current_text = " ".join(str(x) for x in current if str(x).strip())
    else:
        current_text = str(current)

    tags = [sanitize_config_tag(t, sep=sep) for t in _SPLIT_RE.split(current_text.strip())]
    tags = [t for t in tags if t]
    if (not dedupe) or (tag not in tags):
        tags.append(tag)

    result = sep.join(tags)
    if max_len and len(result) > int(max_len):
        result = _truncate_tags(tags, sep=sep, max_len=int(max_len))

    info["Config_type"] = result
    return result


def stable_config_id(atoms: Any, *, mod: int = 1000003) -> int:
    """Derive a stable small integer id from `Config_type` (no extra info keys needed)."""
    info = getattr(atoms, "info", None)
    if info is None:
        return 0
    cfg = info.get("Config_type", "") or ""
    cfg = str(cfg)
    if not cfg:
        return 0
    digest = hashlib.sha1(cfg.encode("utf-8")).digest()
    val = int.from_bytes(digest[:4], "big", signed=False)
    return int(val % int(mod)) if mod else int(val)


def _truncate_tags(tags: list[str], *, sep: str, max_len: int) -> str:
    if max_len <= 0:
        return sep.join(tags)
    if not tags:
        return ""
    if len(sep.join(tags)) <= max_len:
        return sep.join(tags)

    # Prefer preserving the first and last tag; compress the middle.
    if len(tags) >= 3:
        mid = len(tags) - 2
        candidate_tags = [tags[0], f"...+{mid}", tags[-1]]
    else:
        candidate_tags = tags[:]

    candidate = sep.join(candidate_tags)
    if len(candidate) <= max_len:
        return candidate

    # If still too long (very long tags), truncate the last tag to fit.
    if len(candidate_tags) == 1:
        return candidate[:max_len]

    head = sep.join(candidate_tags[:-1])
    if head:
        head += sep
    remaining = max_len - len(head)
    if remaining <= 0:
        return (sep.join(candidate_tags))[:max_len]
    tail = candidate_tags[-1]
    if len(tail) > remaining:
        tail = tail[: max(remaining, 1)]
    return head + tail
