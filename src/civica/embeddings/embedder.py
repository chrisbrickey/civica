"""
Embedder: a model identity paired with the embed functions built from it.
This is shared by ingestion and retrieval pathways to ensure that both
flows use the same model and dimensions.

Clients are instantiated lazily on first call so importing this module
never requires OPENAI_API_KEY to be set.
"""

from collections.abc import Callable
from dataclasses import dataclass

from langchain_openai import OpenAIEmbeddings

EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIMENSIONS = 3072  # property of EMBEDDING_MODEL; must match schema.sql

# Shape of embed_query, exported so callers can type an injected fake the same way.
EmbedQueryFn = Callable[[str], list[float]]

# Shape of embed (batch), exported so callers can type an injected fake the same way.
EmbedBatchFn = Callable[[list[str]], list[list[float]]]

_clients: dict[str, OpenAIEmbeddings] = {}


def _get_client(model: str) -> OpenAIEmbeddings:
    if model not in _clients:
        _clients[model] = OpenAIEmbeddings(model=model)
    return _clients[model]


@dataclass(frozen=True)
class Embedder:
    """A model identity bound to the embed functions built from that model."""

    model: str
    embed: EmbedBatchFn
    embed_query: EmbedQueryFn


def make_embedder(model: str = EMBEDDING_MODEL) -> Embedder:
    """Build an embedder whose identity and functions all derive from `model`."""
    return Embedder(
        model=model,
        embed=lambda texts: _get_client(model).embed_documents(texts),
        embed_query=lambda text: _get_client(model).embed_query(text),
    )


DEFAULT_EMBEDDER = make_embedder()
