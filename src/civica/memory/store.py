"""LangGraph long-term memory: a process-wide PostgresStore singleton.

This is durable memory (key-value store, no semantic search)
that outlives any one conversation.
It persists things like what the user knows and their mistakes.

The store is constructed directly over the shared connection pool
so it stays open for the life of the process.
setup() runs once, on first construction, to create LangGraph's store tables.
"""

from typing import cast

import psycopg
import psycopg.rows
import psycopg_pool
from langgraph.store.postgres import PostgresStore

from civica.db.pool import get_pool

_store: PostgresStore | None = None

# At runtime the pool yields dict_row connections (see db/pool.py) - a PostgresSaver requirement.
# But get_pool()'s type omits the row type, so mypy assumes tuple rows.
# The cast below tells mypy what is already true at runtime.
_DictPool = psycopg_pool.ConnectionPool[psycopg.Connection[psycopg.rows.DictRow]]


def get_store() -> PostgresStore:
    """Returns the singleton PostgresStore, creating and setting it up once."""
    global _store
    if _store is None:
        store = PostgresStore(cast(_DictPool, get_pool()))
        store.setup()
        _store = store
    return _store
