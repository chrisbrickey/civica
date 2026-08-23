"""Unit tests for civica.retrieval.content wiring (no database, no network)."""

import inspect

from civica.embeddings.embedder import DEFAULT_EMBEDDER
from civica.retrieval import content


class TestDefaultEmbedder:
    """search()'s default embedder must be the same object ingestion defaults to.

    If the two sides default to different embedders, search() would query a
    vector space that ingest() never wrote to.
    """

    def test_default_embedder_is_the_shared_default_embedder(self) -> None:
        signature = inspect.signature(content.search)

        assert signature.parameters["embedder"].default is DEFAULT_EMBEDDER
