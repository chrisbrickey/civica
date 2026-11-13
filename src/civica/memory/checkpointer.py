"""LangGraph short-term thread state: a process-wide PostgresSaver singleton.

 LangGraph uses this to snapshot the state of a conversation "thread"
 (e.g., where the user is mid-quiz, mid-explanation).
 This facilitates pausing and resuming of sessions by "thread" id.

The saver is constructed directly over the shared connection pool
so it stays open for the life of the process.
setup() runs once, on first construction, to create LangGraph's checkpoint DB tables.
"""

from typing import Final, cast

import psycopg
import psycopg.rows
import psycopg_pool
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from civica.db.pool import get_pool

_saver: PostgresSaver | None = None

# At runtime the pool yields dict_row connections (see db/pool.py) - a PostgresSaver requirement.
# But get_pool()'s type omits the row type, so mypy assumes tuple rows.
# The cast below tells mypy what is already true at runtime.
_DictPool = psycopg_pool.ConnectionPool[psycopg.Connection[psycopg.rows.DictRow]]


# Our types held in checkpointed state. Reads refuse any other type, so add new state types here.
_STATE_TYPES: Final = (
    ("civica.domain.question", "Question"),
    ("civica.domain.question", "QuestionKind"),
    ("civica.explanation.engine", "Explanation"),
    ("civica.graph.graph", "InputKind"),
    ("civica.graph.planner", "PreviousExchange"),
    ("civica.graph.planner", "ToolCall"),
    ("civica.graph.trace", "HaltReason"),
)


def state_serializer() -> JsonPlusSerializer:
    """The serializer with the allow-list of state types. Test savers use it too."""
    return JsonPlusSerializer(allowed_msgpack_modules=_STATE_TYPES)


def get_saver() -> PostgresSaver:
    """Return the singleton PostgresSaver, creating and setting it up once."""
    global _saver
    if _saver is None:
        saver = PostgresSaver(cast(_DictPool, get_pool()), serde=state_serializer())
        saver.setup()
        _saver = saver
    return _saver
