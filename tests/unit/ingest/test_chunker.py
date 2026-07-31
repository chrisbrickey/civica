"""
Unit tests for civica.ingest.chunker.
"""

from civica.ingest.chunker import chunk_section

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CHUNK_SIZE = 800
_OVERLAP = 100

_PLACEHOLDER_PHRASE = "phrase exemple de remplissage generique pour le decoupage "

_SHORT_TEXT = (_PLACEHOLDER_PHRASE * 4)[:200]
_LONG_TEXT = (_PLACEHOLDER_PHRASE * 70)[:4000]

_WHITESPACE_ONLY_TEXT = "   \n\t  \n  "
_EMPTY_TEXT = ""


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDeterminism:
    """Chunking the same input twice must produce identical output."""

    def test_repeated_calls_on_same_input_return_identical_lists(self) -> None:
        first_result = chunk_section(_LONG_TEXT)
        second_result = chunk_section(_LONG_TEXT)

        assert first_result == second_result


class TestShortInput:
    """Input smaller than the chunk size should not be split at all."""

    def test_text_below_chunk_size_returns_single_chunk_equal_to_input(self) -> None:
        assert len(_SHORT_TEXT) < _CHUNK_SIZE

        result = chunk_section(_SHORT_TEXT)

        assert result == [_SHORT_TEXT]


class TestEmptyOrWhitespaceInput:
    """The chunker must never emit empty (or whitespace-only) chunks."""

    def test_empty_string_returns_empty_list(self) -> None:
        assert chunk_section(_EMPTY_TEXT) == []

    def test_whitespace_only_string_returns_empty_list(self) -> None:
        assert chunk_section(_WHITESPACE_ONLY_TEXT) == []


class TestSizeInvariant:
    """No chunk produced from a long section may exceed the target chunk size."""

    def test_every_chunk_of_long_text_is_at_most_chunk_size(self) -> None:
        result = chunk_section(_LONG_TEXT)

        assert len(result) > 1, "test text must be long enough to require multiple chunks"
        for chunk in result:
            assert len(chunk) <= _CHUNK_SIZE


class TestOverlapInvariant:
    """Consecutive chunks share the declared overlap: B starts with the tail of A."""

    def test_each_chunk_starts_with_the_overlap_tail_of_the_previous_chunk(self) -> None:
        result = chunk_section(_LONG_TEXT)

        assert len(result) > 1, "test text must be long enough to require multiple chunks"
        for previous_chunk, next_chunk in zip(result, result[1:]):
            overlap_tail = previous_chunk[-_OVERLAP:]
            assert next_chunk.startswith(overlap_tail), (
                "expected each chunk to start with the previous chunk's "
                f"trailing {_OVERLAP} characters"
            )


class TestCoverageInvariant:
    """Stripping the declared overlap and concatenating chunks reconstructs the original text."""

    def test_chunks_reconstruct_original_text_exactly(self) -> None:
        result = chunk_section(_LONG_TEXT)

        reconstructed = result[0]
        for next_chunk in result[1:]:
            reconstructed += next_chunk[_OVERLAP:]

        assert reconstructed == _LONG_TEXT
