"""Unit tests for civica.memory.records: LangGraph long-term memory record models.

Pure serialization/typing tests - no database, no network. These lock in the
JSON round-trip contract for the four memory record kinds, including the
Theme-as-slug convention (a theme field dumps to a plain slug string and
rehydrates back into a `Theme` via `Theme.from_slug`).
"""

from typing import Any

import pytest

from civica.domain.source_ref import SourceRef
from civica.domain.themes import DROITS_ET_DEVOIRS, Theme
from civica.memory.records import (
    LearnerProfile,
    MemoryKind,
    MistakeEpisode,
    SessionSummary,
    TopicMastery,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PAGE_SLUG = "test-page"
SECTION_ID = "sec_001"
QUESTION_ID = "q_001"
GOAL = "pass the exam"
SUMMARY = "sample session summary"
MASTERY = 0.5

# Free-form prose fields (goal, summary) are capped so they stay bounded when
# fed into an LLM prompt; a string one character past the cap must be rejected.
MAX_PROSE_LENGTH = 500
OVER_LENGTH_PROSE = "x" * (MAX_PROSE_LENGTH + 1)

VALID_SOURCE_REF_FIELDS: dict[str, Any] = {
    "page_slug": PAGE_SLUG,
    "section_id": SECTION_ID,
}

VALID_LEARNER_PROFILE_FIELDS: dict[str, Any] = {"goal": GOAL}

VALID_TOPIC_MASTERY_FIELDS: dict[str, Any] = {
    "theme": DROITS_ET_DEVOIRS,
    "mastery": MASTERY,
}

VALID_MISTAKE_EPISODE_FIELDS: dict[str, Any] = {
    "theme": DROITS_ET_DEVOIRS,
    "question_id": QUESTION_ID,
    "sources": [SourceRef(**VALID_SOURCE_REF_FIELDS)],
}

VALID_SESSION_SUMMARY_FIELDS: dict[str, Any] = {"summary": SUMMARY}


def make_source_ref(**overrides: Any) -> SourceRef:
    """Build a SourceRef from the shared valid fields, overriding a subset."""
    return SourceRef(**{**VALID_SOURCE_REF_FIELDS, **overrides})


def make_learner_profile(**overrides: Any) -> LearnerProfile:
    """Build a LearnerProfile from the shared valid fields, overriding a subset."""
    return LearnerProfile(**{**VALID_LEARNER_PROFILE_FIELDS, **overrides})


def make_topic_mastery(**overrides: Any) -> TopicMastery:
    """Build a TopicMastery from the shared valid fields, overriding a subset."""
    return TopicMastery(**{**VALID_TOPIC_MASTERY_FIELDS, **overrides})


def make_mistake_episode(**overrides: Any) -> MistakeEpisode:
    """Build a MistakeEpisode from the shared valid fields, overriding a subset."""
    return MistakeEpisode(**{**VALID_MISTAKE_EPISODE_FIELDS, **overrides})


def make_session_summary(**overrides: Any) -> SessionSummary:
    """Build a SessionSummary from the shared valid fields, overriding a subset."""
    return SessionSummary(**{**VALID_SESSION_SUMMARY_FIELDS, **overrides})


# ---------------------------------------------------------------------------
# Validation: free-form prose fields reject over-length input
# ---------------------------------------------------------------------------


def test_learner_profile_rejects_over_length_goal() -> None:
    with pytest.raises(ValueError):
        make_learner_profile(goal=OVER_LENGTH_PROSE)


def test_session_summary_rejects_over_length_summary() -> None:
    with pytest.raises(ValueError):
        make_session_summary(summary=OVER_LENGTH_PROSE)


# ---------------------------------------------------------------------------
# SourceRef: round-trips through JSON with plain string fields intact
# ---------------------------------------------------------------------------


def test_source_ref_round_trips_through_json_with_plain_string_fields() -> None:
    source_ref = make_source_ref()

    dumped = source_ref.model_dump(mode="json")
    assert isinstance(dumped["page_slug"], str)
    assert isinstance(dumped["section_id"], str)
    assert dumped["page_slug"] == PAGE_SLUG
    assert dumped["section_id"] == SECTION_ID

    rehydrated = SourceRef.model_validate(dumped)
    assert rehydrated == source_ref
    assert rehydrated.page_slug == PAGE_SLUG
    assert rehydrated.section_id == SECTION_ID


# ---------------------------------------------------------------------------
# Theme-as-slug convention: a theme field dumps to a plain slug string
# ---------------------------------------------------------------------------


def test_topic_mastery_dumps_theme_as_plain_slug_string() -> None:
    topic_mastery = make_topic_mastery()

    dumped = topic_mastery.model_dump(mode="json")
    assert dumped["theme"] == DROITS_ET_DEVOIRS.slug
    assert isinstance(dumped["theme"], str)

    rehydrated = TopicMastery.model_validate(dumped)
    assert rehydrated == topic_mastery
    assert rehydrated.theme == DROITS_ET_DEVOIRS
    assert isinstance(rehydrated.theme, Theme)


# ---------------------------------------------------------------------------
# All four record kinds: JSON round-trip equality and kind identity
# ---------------------------------------------------------------------------


def test_learner_profile_round_trips_and_reports_its_kind() -> None:
    learner_profile = make_learner_profile()

    dumped = learner_profile.model_dump(mode="json")
    assert "kind" not in dumped

    rehydrated = LearnerProfile.model_validate(dumped)
    assert rehydrated == learner_profile
    assert LearnerProfile.kind is MemoryKind.LEARNER_PROFILE
    assert rehydrated.kind is MemoryKind.LEARNER_PROFILE


def test_topic_mastery_round_trips_and_reports_its_kind() -> None:
    topic_mastery = make_topic_mastery()

    dumped = topic_mastery.model_dump(mode="json")
    assert "kind" not in dumped

    rehydrated = TopicMastery.model_validate(dumped)
    assert rehydrated == topic_mastery
    assert TopicMastery.kind is MemoryKind.TOPIC_MASTERY
    assert rehydrated.kind is MemoryKind.TOPIC_MASTERY


def test_mistake_episode_round_trips_with_sources_and_reports_its_kind() -> None:
    mistake_episode = make_mistake_episode()

    dumped = mistake_episode.model_dump(mode="json")
    assert "kind" not in dumped

    rehydrated = MistakeEpisode.model_validate(dumped)
    assert rehydrated == mistake_episode
    assert rehydrated.sources == VALID_MISTAKE_EPISODE_FIELDS["sources"]
    assert MistakeEpisode.kind is MemoryKind.MISTAKE_EPISODE
    assert rehydrated.kind is MemoryKind.MISTAKE_EPISODE


def test_session_summary_round_trips_and_reports_its_kind() -> None:
    session_summary = make_session_summary()

    dumped = session_summary.model_dump(mode="json")
    assert "kind" not in dumped

    rehydrated = SessionSummary.model_validate(dumped)
    assert rehydrated == session_summary
    assert SessionSummary.kind is MemoryKind.SESSION_SUMMARY
    assert rehydrated.kind is MemoryKind.SESSION_SUMMARY
