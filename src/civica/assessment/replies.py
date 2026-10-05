"""Helpers for reading JSON replies from chat models."""

import re

_FENCE_PATTERN = re.compile(r"^```[a-zA-Z]*\s*(.*?)\s*```$", re.DOTALL)


def strip_code_fence(reply: str) -> str:
    """Trim the reply and unwrap a surrounding markdown code fence if present."""
    body = reply.strip()
    fenced = _FENCE_PATTERN.match(body)
    return fenced.group(1) if fenced else body
