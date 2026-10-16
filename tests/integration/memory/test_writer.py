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
from civica.memory.writer import MemoryNotAllowed, get, list_records, put, put_planner, put_raw

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

# Spans more than one internal page of list_records, so paging runs against Postgres.
_MANY_EPISODE_COUNT = 150


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


def test_list_records_returns_every_record_across_pages(
    store: _StoreFixture,
) -> None:
    """list_records pages through the real store and returns each record once."""
    _live_store, written = store
    user_id = _new_user_id()
    episodes = [
        MistakeEpisode(theme=DROITS_ET_DEVOIRS, question_id=f"question-{i:03d}")
        for i in range(_MANY_EPISODE_COUNT)
    ]
    for episode in episodes:
        key = f"mistake-{episode.question_id}"
        written.append(((episode.kind.value, str(user_id)), key))
        put(user_id, key, episode)

    records = list_records(user_id, MistakeEpisode)

    assert len(records) == _MANY_EPISODE_COUNT
    assert {r.question_id for r in records} == {e.question_id for e in episodes}


def test_list_records_is_scoped_to_the_requesting_user(
    store: _StoreFixture,
) -> None:
    """Records written for one user are not listed for another."""
    _live_store, written = store
    alex_id = _new_user_id()
    jordan_id = _new_user_id()
    record = MistakeEpisode(theme=DROITS_ET_DEVOIRS, question_id=_QUESTION_ID)
    written.append(((record.kind.value, str(alex_id)), _MISTAKE_KEY))

    put(alex_id, _MISTAKE_KEY, record)

    assert list_records(alex_id, MistakeEpisode) == [record]
    assert list_records(jordan_id, MistakeEpisode) == []


def test_put_planner_writes_an_allowlisted_kind(
    store: _StoreFixture,
) -> None:
    """A mistake episode written on the planner's behalf is readable as a typed record."""
    _live_store, written = store
    user_id = _new_user_id()
    record = MistakeEpisode(theme=DROITS_ET_DEVOIRS, question_id=_QUESTION_ID)
    written.append(((record.kind.value, str(user_id)), _MISTAKE_KEY))

    put_planner(user_id, record.kind.value, _MISTAKE_KEY, record.model_dump(mode="json"))

    assert get(user_id, _MISTAKE_KEY, MistakeEpisode) == record


def test_put_planner_rejects_topic_mastery_without_writing(
    store: _StoreFixture,
) -> None:
    """The planner cannot write mastery, even with a well-formed value."""
    _live_store, written = store
    user_id = _new_user_id()
    record = TopicMastery(theme=DROITS_ET_DEVOIRS, mastery=_MASTERY_VALUE)
    written.append(((record.kind.value, str(user_id)), _MASTERY_KEY))

    with pytest.raises(MemoryNotAllowed):
        put_planner(user_id, record.kind.value, _MASTERY_KEY, record.model_dump(mode="json"))

    assert get(user_id, _MASTERY_KEY, TopicMastery) is None
