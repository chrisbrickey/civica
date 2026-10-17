"""Fingerprint of a prompt set's text, so tests can pin it alongside the set's version."""

import hashlib
import json
from dataclasses import fields
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from _typeshed import DataclassInstance


def prompt_text_hash(prompt_set: "DataclassInstance") -> str:
    """sha256 over every field of the prompt set except its version."""
    texts = {f.name: getattr(prompt_set, f.name) for f in fields(prompt_set) if f.name != "version"}
    payload = json.dumps(texts, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()
