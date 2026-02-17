"""Utility functions for K2DO."""

from pathlib import Path
import re


def ensure_dir(path: Path) -> Path:
    """Ensure a directory exists, create if not."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_workspace_path() -> Path:
    return Path.home() / ".k2do" / "workspace"


def get_data_path() -> Path:
    path = Path.home() / ".k2do"
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_filename(value: str, replacement: str = "_") -> str:
    """Convert an arbitrary string into a filesystem-safe filename."""
    # Replace path separators and reserved characters across platforms.
    value = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', replacement, value)
    value = value.strip().strip(".")
    return value or "untitled"


_THINK_CLOSE_RE = re.compile(r"</\s*think(?:[_-][a-z0-9_-]+)?\s*>", re.IGNORECASE)
_THINK_BLOCK_RE = re.compile(
    r"<\s*think(?:[_-][a-z0-9_-]+)?\s*>.*?</\s*think(?:[_-][a-z0-9_-]+)?\s*>",
    re.IGNORECASE | re.DOTALL,
)
_THINK_TAG_RE = re.compile(r"</?\s*think(?:[_-][a-z0-9_-]+)?\s*>", re.IGNORECASE)
_SPECIAL_TOKENS = (
    "<|im_end|>",
    "<|im_start|>",
    "<|eot_id|>",
    "<｜end▁of▁sentence｜>",
    "<｜User｜>",
    "<｜Assistant｜>",
)


def sanitize_model_output(text: str | None) -> str | None:
    """
    Remove leaked reasoning wrappers and transport tokens from model output.

    Some K2 responses may include internal scratchpad content followed by a
    closing tag like </think_fast>. In that case we keep only the tail after
    the last closing think tag.
    """
    if text is None:
        return None

    cleaned = text.replace("\r\n", "\n")
    for token in _SPECIAL_TOKENS:
        cleaned = cleaned.replace(token, "")

    close_matches = list(_THINK_CLOSE_RE.finditer(cleaned))
    if close_matches:
        tail = cleaned[close_matches[-1].end():].strip()
        if tail:
            cleaned = tail
        else:
            cleaned = _THINK_BLOCK_RE.sub("", cleaned)
    else:
        cleaned = _THINK_BLOCK_RE.sub("", cleaned)

    cleaned = _THINK_TAG_RE.sub("", cleaned)
    cleaned = cleaned.strip()
    return cleaned
