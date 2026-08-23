"""Unit tests for civica.embeddings.embedder."""

import importlib.util
from dataclasses import FrozenInstanceError
from types import ModuleType

import pytest

from civica.embeddings.embedder import (
    DEFAULT_EMBEDDER,
    EMBEDDING_MODEL,
    Embedder,
    make_embedder,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SAMPLE_MODEL = "sample-embedding-model"
_SAMPLE_TEXTS = ["sample text for embedding"]
_SAMPLE_VECTORS = [[0.1, 0.2, 0.3]]
_SAMPLE_QUERY_VECTOR = [0.4, 0.5, 0.6]

_EMBEDDER_MODULE_NAME = "civica.embeddings.embedder"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_isolated_embedder_module(name: str) -> ModuleType:
    """Import a fresh copy of civica.embeddings.embedder under its own module name.

    Distinct from sys.modules[_EMBEDDER_MODULE_NAME] so this test's monkeypatching
    cannot leak into other tests that import the real module.
    """
    source_spec = importlib.util.find_spec(_EMBEDDER_MODULE_NAME)
    assert source_spec is not None and source_spec.origin is not None
    spec = importlib.util.spec_from_file_location(name, source_spec.origin)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _recording_client(constructions: list[str]) -> type:
    """A fake OpenAIEmbeddings that records the model it was built with."""

    class _RecordingOpenAIEmbeddings:
        def __init__(self, model: str) -> None:
            constructions.append(model)

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return _SAMPLE_VECTORS

        def embed_query(self, text: str) -> list[float]:
            return _SAMPLE_QUERY_VECTOR

    return _RecordingOpenAIEmbeddings


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMakeEmbedder:
    """make_embedder binds the embedder's identity to the model it was built from."""

    def test_returns_embedder_with_given_model(self) -> None:
        result = make_embedder(_SAMPLE_MODEL)

        assert isinstance(result, Embedder)
        assert result.model == _SAMPLE_MODEL

    def test_shared_default_embedder_uses_the_project_embedding_model(self) -> None:
        assert DEFAULT_EMBEDDER.model == EMBEDDING_MODEL


class TestNoKeyImport:
    """Importing the module and building an embedder never requires OPENAI_API_KEY."""

    def test_import_and_make_embedder_construct_no_client_without_api_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        constructions: list[str] = []
        module = _load_isolated_embedder_module("civica_embeddings_embedder_no_key")
        monkeypatch.setattr(module, "OpenAIEmbeddings", _recording_client(constructions))

        embedder = module.make_embedder(_SAMPLE_MODEL)

        assert embedder.model == _SAMPLE_MODEL
        assert constructions == []


class TestClientConstructionOnFirstEmbed:
    """The OpenAI client is built lazily, from the model the embedder was made with."""

    def test_first_embed_call_constructs_client_with_embedder_model(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        constructions: list[str] = []
        module = _load_isolated_embedder_module(
            "civica_embeddings_embedder_client_construction"
        )
        monkeypatch.setattr(module, "OpenAIEmbeddings", _recording_client(constructions))
        embedder = module.make_embedder(_SAMPLE_MODEL)
        assert constructions == []

        result = embedder.embed(_SAMPLE_TEXTS)

        assert constructions == [_SAMPLE_MODEL]
        assert result == _SAMPLE_VECTORS


class TestEmbedderImmutability:
    """Embedder is a frozen dataclass; its fields cannot be reassigned."""

    def test_assigning_to_a_field_raises(self) -> None:
        embedder = make_embedder(_SAMPLE_MODEL)

        with pytest.raises(FrozenInstanceError):
            embedder.model = "sample-other-model"  # type: ignore[misc]
