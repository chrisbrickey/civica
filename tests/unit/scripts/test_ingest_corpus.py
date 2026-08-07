"""Unit tests for civica.scripts.ingest_corpus content-hash composition."""

import json
from pathlib import Path

from civica.domain.themes import DROITS_ET_DEVOIRS
from civica.scripts import ingest_corpus

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_THEME = DROITS_ET_DEVOIRS
_PAGE_SLUG = "sample-page"
_SECTION_ID = "section-001"
_SECTION_HEADING = "sample-heading"
_SECTION_TEXT = "sample corpus text for content hashing"

_MODEL_A = "sample-model-a"
_MODEL_B = "sample-model-b"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_corpus_page(corpus_root: Path) -> None:
    """Write one minimal normalized-page JSON under corpus_root."""
    page = {
        "theme": _THEME.slug,
        "slug": _PAGE_SLUG,
        "title": "sample-title",
        "source_url": "https://example.com",
        "description": "sample-description",
        "sections": [
            {"id": _SECTION_ID, "heading": _SECTION_HEADING, "text": _SECTION_TEXT}
        ],
    }
    (corpus_root / f"{_PAGE_SLUG}.json").write_text(json.dumps(page), encoding="utf-8")


def _hashes_under_model(corpus_root: Path, model: str) -> list[str]:
    """Collect content hashes produced under embedding model `model`."""
    return [
        chunk.content_hash
        for chunk in ingest_corpus.collect_pending_chunks(corpus_root, embedding_model=model)
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestModelIdentityInContentHash:
    """content_hash encodes the embedding model that produced the vector."""

    def test_different_models_produce_different_hashes(self, corpus_root: Path) -> None:
        _write_corpus_page(corpus_root)

        hashes_a = _hashes_under_model(corpus_root, _MODEL_A)
        hashes_b = _hashes_under_model(corpus_root, _MODEL_B)

        assert hashes_a
        assert hashes_a != hashes_b

    def test_same_model_is_deterministic(self, corpus_root: Path) -> None:
        _write_corpus_page(corpus_root)

        first = _hashes_under_model(corpus_root, _MODEL_A)
        second = _hashes_under_model(corpus_root, _MODEL_A)

        assert first
        assert first == second
