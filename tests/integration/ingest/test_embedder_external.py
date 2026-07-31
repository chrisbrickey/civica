"""
External test for civica.ingest.embedder.

Makes a real call to the OpenAI embeddings API, so it is marked @pytest.mark.external
and excluded from the default `uv run pytest` run.

Run explicitly with `uv run pytest -m external`.
"""

import pytest

from civica.ingest.embedder import EMBEDDING_DIMENSIONS, embed


@pytest.mark.external
def test_embed_returns_one_vector_of_the_expected_dimension() -> None:
    result = embed(["texte d'exemple generique"])

    assert len(result) == 1
    vector = result[0]
    assert len(vector) == EMBEDDING_DIMENSIONS
    assert all(isinstance(value, float) for value in vector)
