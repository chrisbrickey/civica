"""Unit tests for civica.domain.chunk: Chunk pydantic domain model."""

from typing import Any

import pytest

from civica.domain.chunk import Chunk
from civica.domain.themes import DROITS_ET_DEVOIRS, Theme

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_CHUNK_FIELDS: dict[str, Any] = {
    "theme": DROITS_ET_DEVOIRS,
    "page_slug": "test-page",
    "section_id": "sec_001",
    "chunk_index": 0,
    "content_hash": "sample-hash",
    "text": "sample text",
}


def make_chunk(**overrides: Any) -> Chunk:
    """Build a Chunk from the shared valid fields, overriding a subset."""
    return Chunk(**{**VALID_CHUNK_FIELDS, **overrides})


# ---------------------------------------------------------------------------
# Validation: invalid field values rejected
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field, invalid_value",
    [
        ("text", ""),
        ("page_slug", ""),
        ("section_id", ""),
        ("content_hash", ""),
        ("chunk_index", -1),
    ],
)
def test_invalid_field_value_raises_validation_error(
    field: str, invalid_value: Any
) -> None:
    with pytest.raises(ValueError):
        make_chunk(**{field: invalid_value})


# ---------------------------------------------------------------------------
# Frozen model: mutation raises
# ---------------------------------------------------------------------------


def test_frozen_chunk_raises_on_mutation() -> None:
    chunk = make_chunk()
    with pytest.raises(ValueError):
        chunk.text = "mutated text"


# ---------------------------------------------------------------------------
# theme field: accepts a Theme instance and round-trips unchanged
# ---------------------------------------------------------------------------


def test_theme_field_round_trips_theme_instance() -> None:
    chunk = make_chunk(theme=DROITS_ET_DEVOIRS)
    assert chunk.theme is DROITS_ET_DEVOIRS
    assert isinstance(chunk.theme, Theme)
