"""
Integration tests for civica.memory.writer.

Exercises the real LangGraph PostgresStore against the test database.
"""

import uuid
from collections.abc import Generator

import pytest
from langgraph.store.postgres import PostgresStore

from civica.domain.source_ref import SourceRef
from civica.domain.themes import DROITS_ET_DEVOIRS
from civica.domain.user import UserId
from civica.memory.records import MistakeEpisode, TopicMastery
from civica.memory.store import get_store
from civica.memory.writer import MemoryNotAllowed, get, put, put_raw

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MASTERY_KEY = "current"
_MASTERY_VALUE = 0.5

_MISTAKE_KEY = "mistake-001"
_QUESTION_ID = "question-001"
_SOURCE_PAGE_SLUG = "sample-page"
_SOURCE_SECTION_ID = "section-001"

_INVALID_KIND = "not_a_real_kind"
_RAW_KEY = "raw-key-001"
_RAW_VALUE: dict[str, object] = {"field": "value"}


def _new_user_id() -> UserId:
    return UserId(uuid.uuid4())


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


_StoreFixture = tuple[PostgresStore, list[tuple[tuple[str, ...], str]]]


@pytest.fixture()
def store() -> Generator[_StoreFixture, None, None]:
    """The singleton store, cleaning up any namespace/key entries the test tracks.

    Tests append (namespace, key) tuples to the yielded list for every entry
    they write; teardown deletes each one so nothing leaks past the test.
    """
    live_store = get_store()
    written: list[tuple[tuple[str, ...], str]] = []
    try:
        yield live_store, written
    finally:
        for namespace, key in written:
            live_store.delete(namespace, key)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_typed_record_round_trips_through_put_and_get(
    store: _StoreFixture,
) -> None:
    """A record written via writer.put is readable back via writer.get, unchanged."""
    _live_store, written = store
    user_id = _new_user_id()
    record = TopicMastery(theme=DROITS_ET_DEVOIRS, mastery=_MASTERY_VALUE)
    written.append(((record.kind.value, str(user_id)), _MASTERY_KEY))

    put(user_id, _MASTERY_KEY, record)
    restored = get(user_id, _MASTERY_KEY, TopicMastery)

    assert restored == record
    assert isinstance(restored, TopicMastery)


def test_put_raw_rejects_kind_outside_allowlist(
    store: _StoreFixture,
) -> None:
    """put_raw is the untrusted graph/LLM boundary: an unknown kind is rejected."""
    user_id = _new_user_id()

    with pytest.raises(MemoryNotAllowed):
        put_raw(user_id, _INVALID_KIND, _RAW_KEY, _RAW_VALUE)


def test_get_is_scoped_to_the_requesting_user(
    store: _StoreFixture,
) -> None:
    """A record written for one user is invisible to a different user."""
    _live_store, written = store
    alex_id = _new_user_id()
    jordan_id = _new_user_id()
    record = TopicMastery(theme=DROITS_ET_DEVOIRS, mastery=_MASTERY_VALUE)
    written.append(((record.kind.value, str(alex_id)), _MASTERY_KEY))

    put(alex_id, _MASTERY_KEY, record)
    restored_for_jordan = get(jordan_id, _MASTERY_KEY, TopicMastery)

    assert restored_for_jordan is None


def test_mistake_episode_with_sources_round_trips(
    store: _StoreFixture,
) -> None:
    """A MistakeEpisode's nested SourceRefs survive a real write and read."""
    _live_store, written = store
    user_id = _new_user_id()
    record = MistakeEpisode(
        theme=DROITS_ET_DEVOIRS,
        question_id=_QUESTION_ID,
        sources=[SourceRef(page_slug=_SOURCE_PAGE_SLUG, section_id=_SOURCE_SECTION_ID)],
    )
    written.append(((record.kind.value, str(user_id)), _MISTAKE_KEY))

    put(user_id, _MISTAKE_KEY, record)
    restored = get(user_id, _MISTAKE_KEY, MistakeEpisode)

    assert restored == record
    assert restored is not None
    assert restored.sources == record.sources
