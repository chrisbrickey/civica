"""Unit tests for civica.memory.writer: list_records and the write allowlists.

Uses an injected in-memory LangGraph store, so no database is needed.
"""

import pytest
from langgraph.store.memory import InMemoryStore

from civica.domain.themes import DROITS_ET_DEVOIRS, PRINCIPES_ET_VALEURS_DE_LA_REPUBLIQUE
from civica.memory.records import MemoryKind, MistakeEpisode, TopicMastery
from civica.memory.writer import (
    MEMORY_WRITE_ALLOWLIST,
    PLANNER_WRITE_ALLOWLIST,
    get,
    list_records,
    put,
)
from tests.support.memory import HALF_MASTERY
from tests.support.users import OTHER_USER_ID, USER_ID

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MISTAKE_A = MistakeEpisode(theme=DROITS_ET_DEVOIRS, question_id="q_001")
MISTAKE_B = MistakeEpisode(
    theme=PRINCIPES_ET_VALEURS_DE_LA_REPUBLIQUE, question_id="q_002"
)
MASTERY = TopicMastery(theme=DROITS_ET_DEVOIRS, mastery=HALF_MASTERY)

# More than the store's default search page size, so a truncated read is caught.
MANY_EPISODE_COUNT = 25


def _key(record: MistakeEpisode) -> str:
    return f"mistake-{record.question_id}"


# ---------------------------------------------------------------------------
# list_records
# ---------------------------------------------------------------------------


def test_list_records_returns_empty_list_when_user_has_none(
    store: InMemoryStore,
) -> None:
    assert list_records(USER_ID, MistakeEpisode, store) == []


def test_list_records_returns_every_stored_record_typed(
    store: InMemoryStore,
) -> None:
    for episode in (MISTAKE_A, MISTAKE_B):
        put(USER_ID, _key(episode), episode, store)

    records = list_records(USER_ID, MistakeEpisode, store)

    assert all(isinstance(r, MistakeEpisode) for r in records)
    assert sorted(records, key=lambda r: r.question_id) == [MISTAKE_A, MISTAKE_B]


def test_list_records_returns_all_records_beyond_default_page_size(
    store: InMemoryStore,
) -> None:
    episodes = [
        MistakeEpisode(theme=DROITS_ET_DEVOIRS, question_id=f"q_{i:03d}")
        for i in range(MANY_EPISODE_COUNT)
    ]
    for episode in episodes:
        put(USER_ID, _key(episode), episode, store)

    records = list_records(USER_ID, MistakeEpisode, store)

    assert len(records) == MANY_EPISODE_COUNT


def test_list_records_excludes_other_users_and_other_kinds(
    store: InMemoryStore,
) -> None:
    put(USER_ID, _key(MISTAKE_A), MISTAKE_A, store)
    put(OTHER_USER_ID, _key(MISTAKE_B), MISTAKE_B, store)
    put(USER_ID, MASTERY.theme.slug, MASTERY, store)

    assert list_records(USER_ID, MistakeEpisode, store) == [MISTAKE_A]
    assert list_records(USER_ID, TopicMastery, store) == [MASTERY]


# ---------------------------------------------------------------------------
# Write allowlists
# ---------------------------------------------------------------------------


def test_planner_allowlist_is_exactly_mistake_episodes_and_session_summaries() -> None:
    assert PLANNER_WRITE_ALLOWLIST == frozenset(
        {MemoryKind.MISTAKE_EPISODE, MemoryKind.SESSION_SUMMARY}
    )


@pytest.mark.parametrize("kind", [MemoryKind.TOPIC_MASTERY, MemoryKind.LEARNER_PROFILE])
def test_planner_allowlist_excludes_derived_and_onboarding_kinds(kind: MemoryKind) -> None:
    assert kind not in PLANNER_WRITE_ALLOWLIST


def test_planner_allowlist_is_narrower_than_the_in_process_allowlist() -> None:
    assert PLANNER_WRITE_ALLOWLIST != MEMORY_WRITE_ALLOWLIST
    assert PLANNER_WRITE_ALLOWLIST < MEMORY_WRITE_ALLOWLIST
    assert MemoryKind.TOPIC_MASTERY in MEMORY_WRITE_ALLOWLIST


def test_in_process_put_still_writes_topic_mastery(store: InMemoryStore) -> None:
    put(USER_ID, MASTERY.theme.slug, MASTERY, store)

    assert get(USER_ID, MASTERY.theme.slug, TopicMastery, store) == MASTERY
