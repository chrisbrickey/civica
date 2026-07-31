"""
Thin wrapper around the OpenAI embeddings client for corpus ingestion.

The client is instantiated lazily on first call so importing this module
never requires OPENAI_API_KEY to be set.
"""

from langchain_openai import OpenAIEmbeddings

EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIMENSIONS = 3072  # property of EMBEDDING_MODEL; must match schema.sql

_client: OpenAIEmbeddings | None = None


def _get_client() -> OpenAIEmbeddings:
    global _client
    if _client is None:
        _client = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    return _client


def embed(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts, returning one vector per input text."""
    return _get_client().embed_documents(texts)
