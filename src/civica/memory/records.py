"""LangGraph long-term memory record models.

These models define the shapes written to and read from the LangGraph `PostgresStore`.
Records are frozen and JSON-serializable.
"""

from enum import StrEnum
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator

from civica.domain.source_ref import SourceRef
from civica.domain.themes import Theme

# Free-form prose fields are used to capture nuance, which is why the maximum
# character allowance is generous. Such fields are intended only for LLM or human consumption.
# Other fields (e.g. identifiers, slugs) are more tightly controlled.
_MAX_PROSE_LENGTH = 500


class MemoryKind(StrEnum):
    """The kind of long-term memory record."""

    LEARNER_PROFILE = "learner_profile"
    TOPIC_MASTERY = "topic_mastery"
    MISTAKE_EPISODE = "mistake_episode"
    SESSION_SUMMARY = "session_summary"


class MemoryRecord(BaseModel):  # type: ignore[explicit-any]
    """Shared base for all long-term memory records.

    `kind` is a ClassVar (not a pydantic field) so it never appears in
    `model_dump()` output. Each concrete subclass assigns its own value.
    """

    model_config = ConfigDict(frozen=True)

    kind: ClassVar[MemoryKind]


class _ThemeBearingRecord(MemoryRecord):  # type: ignore[explicit-any]
    """Shared base for records tied to a civic education theme.

    Dumps `theme` as its plain slug string and rehydrates a dumped slug
    string back into a `Theme` on validation.
    """

    theme: Theme

    @field_serializer("theme")
    def _serialize_theme(self, theme: Theme) -> str:
        return theme.slug

    @field_validator("theme", mode="before")
    @classmethod
    def _validate_theme(cls, value: object) -> object:
        if isinstance(value, str):
            return Theme.from_slug(value)
        return value


class LearnerProfile(MemoryRecord):  # type: ignore[explicit-any]
    """A learner's stated goal for studying the exam."""

    kind: ClassVar[MemoryKind] = MemoryKind.LEARNER_PROFILE

    goal: str = Field(min_length=1, max_length=_MAX_PROSE_LENGTH)


class TopicMastery(_ThemeBearingRecord):  # type: ignore[explicit-any]
    """A learner's mastery level for a given theme."""

    kind: ClassVar[MemoryKind] = MemoryKind.TOPIC_MASTERY

    mastery: float = Field(ge=0.0, le=1.0)


class MistakeEpisode(_ThemeBearingRecord):  # type: ignore[explicit-any]
    """A record of a learner's mistake on a specific question."""

    kind: ClassVar[MemoryKind] = MemoryKind.MISTAKE_EPISODE

    question_id: str = Field(min_length=1)
    sources: list[SourceRef] = Field(default_factory=list)


class SessionSummary(MemoryRecord):  # type: ignore[explicit-any]
    """A summary of a completed study session."""

    kind: ClassVar[MemoryKind] = MemoryKind.SESSION_SUMMARY

    summary: str = Field(min_length=1, max_length=_MAX_PROSE_LENGTH)
