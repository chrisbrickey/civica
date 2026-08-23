"""Unit tests for civica.scripts.ingest_corpus content-hash composition."""

import json
import unicodedata
from pathlib import Path

from civica.domain.themes import DROITS_ET_DEVOIRS
from civica.embeddings.embedder import DEFAULT_EMBEDDER, Embedder
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

# The same accented text in both normalization forms: different code points, same characters.
_TEXT_NFC = unicodedata.normalize("NFC", "sample accented text with a café in it")
_TEXT_NFD = unicodedata.normalize("NFD", _TEXT_NFC)

# Two (model, title) pairs whose plain concatenation is byte-identical, so only a
# field separator can keep their hash payloads apart.
_SPOOF_MODEL_SHORT = "sample-model-x"
_SPOOF_TITLE_LONG = "y-sample-title"
_SPOOF_MODEL_LONG = "sample-model-xy"
_SPOOF_TITLE_SHORT = "-sample-title"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_corpus_page(
    corpus_root: Path,
    *,
    slug: str = _PAGE_SLUG,
    title: str = "sample-title",
    section_id: str = _SECTION_ID,
    section_heading: str = _SECTION_HEADING,
    section_text: str = _SECTION_TEXT,
) -> None:
    """Write one minimal normalized-page JSON under corpus_root."""
    page = {
        "theme": _THEME.slug,
        "slug": slug,
        "title": title,
        "source_url": "https://example.com",
        "description": "sample-description",
        "sections": [
            {"id": section_id, "heading": section_heading, "text": section_text}
        ],
    }
    (corpus_root / f"{slug}.json").write_text(json.dumps(page), encoding="utf-8")


def _fake_embedder(model: str) -> Embedder:
    """Build an Embedder whose functions are never expected to be called here.

    collect_pending_chunks() never embeds; it only needs embedder.model.
    """

    def _unreachable_embed(texts: list[str]) -> list[list[float]]:
        raise AssertionError("collect_pending_chunks must not embed")

    def _unreachable_embed_query(text: str) -> list[float]:
        raise AssertionError("collect_pending_chunks must not embed")

    return Embedder(model=model, embed=_unreachable_embed, embed_query=_unreachable_embed_query)


def _hashes_under_model(corpus_root: Path, model: str) -> list[str]:
    """Collect content hashes produced under embedding model `model`."""
    return [
        chunk.content_hash
        for chunk in ingest_corpus.collect_pending_chunks(
            corpus_root, embedder=_fake_embedder(model)
        )
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


class TestUnicodeNormalizationInContentHash:
    """Accent-encoding drift in the source must not fork one chunk into two rows."""

    def test_nfc_and_nfd_text_hash_identically(self, corpus_root: Path) -> None:
        assert _TEXT_NFC != _TEXT_NFD

        _write_corpus_page(corpus_root, section_text=_TEXT_NFC)
        composed = _hashes_under_model(corpus_root, _MODEL_A)

        _write_corpus_page(corpus_root, section_text=_TEXT_NFD)
        decomposed = _hashes_under_model(corpus_root, _MODEL_A)

        assert composed
        assert composed == decomposed


class TestModelNameCannotSpoofTheHashedText:
    """A model name must not be able to impersonate the leading characters of the text."""

    def test_shifting_a_character_from_model_to_text_changes_the_hash(
        self, corpus_root: Path
    ) -> None:
        assert _SPOOF_MODEL_SHORT + _SPOOF_TITLE_LONG == _SPOOF_MODEL_LONG + _SPOOF_TITLE_SHORT

        _write_corpus_page(corpus_root, title=_SPOOF_TITLE_LONG)
        short_model = _hashes_under_model(corpus_root, _SPOOF_MODEL_SHORT)

        _write_corpus_page(corpus_root, title=_SPOOF_TITLE_SHORT)
        long_model = _hashes_under_model(corpus_root, _SPOOF_MODEL_LONG)

        assert short_model
        assert short_model != long_model


class TestPinnedContentHash:
    """Tripwire against an unintended change to the hashing algorithm."""

    def test_known_text_hashes_to_the_pinned_digest(self, corpus_root: Path) -> None:
        _write_corpus_page(
            corpus_root,
            slug="sample-pinned-page",
            title="sample-pinned-title",
            section_id="section-pinned",
            section_heading="sample-pinned-heading",
            section_text="sample pinned text for digest verification",
        )

        chunks = ingest_corpus.collect_pending_chunks(corpus_root, embedder=DEFAULT_EMBEDDER)
        assert len(chunks) == 1

        # Pinned to detect any change to the hash payload format
        pinned_digest = "9930e665d31ac51708eee36c43ebceea76e0560cc12d985e41208043f9d44d01"
        reembed_warning = (
            "content_hash for a known input has changed. This means that something in the "
            "production code changed how chunks are hashed, which will trigger re-embedding "
            "of the entire corpus on the next ingestion. This test is intended to flag the "
            "issue before such a change is merged to production. If this change is intentional, "
            "update the `pinned_digest` variable to capture the new hashing format."
        )
        assert chunks[0].content_hash == pinned_digest, reembed_warning
