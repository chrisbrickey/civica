"""
Thin wrapper around the OpenAI embeddings client, shared by corpus
ingestion and content retrieval so both use the same model and dimensions.

The client is instantiated lazily on first call so importing this module
never requires OPENAI_API_KEY to be set.
"""

from collections.abc import Callable

from langchain_openai import OpenAIEmbeddings

EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIMENSIONS = 3072  # property of EMBEDDING_MODEL; must match schema.sql

# Shape of embed_query, exported so callers can type an injected fake the same way.
EmbedQueryFn = Callable[[str], list[float]]

# Shape of embed (batch), exported so callers can type an injected fake the same way.
EmbedBatchFn = Callable[[list[str]], list[list[float]]]

_client: OpenAIEmbeddings | None = None


def _get_client() -> OpenAIEmbeddings:
    global _client
    if _client is None:
        _client = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    return _client


def embed(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts, returning one vector per input text."""
    return _get_client().embed_documents(texts)


def embed_query(text: str) -> list[float]:
    """Embed a single retrieval query with the same model used for ingestion."""
    return _get_client().embed_query(text)
