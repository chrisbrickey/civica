"""Shared builders and fakes for assessment tests (unit and integration)."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import PrivateAttr

from civica.assessment.critic import Verdict
from civica.domain.chunk import Chunk
from civica.domain.question import Question, QuestionKind
from civica.domain.source_ref import SourceRef
from civica.domain.themes import DROITS_ET_DEVOIRS, EXAM_QUESTION_COUNTS, Theme
from civica.domain.user import UserId
from civica.retrieval.content import ContentChunk

if TYPE_CHECKING:
    # Type-only import keeps this module importable by tests that never touch rejections.
    from civica.assessment.store import Rejection

SAMPLE_OPTIONS = ("option-a", "option-b", "option-c", "option-d")
SAMPLE_SOURCE = SourceRef(page_slug="sample-page", section_id="section-001")
OPTION_COUNT = len(SAMPLE_OPTIONS)
SAMPLE_CORRECT_INDEX = 1
DEFAULT_SECTION_COUNT = 3

# Course-objective intro sections; the only two section_id variants in the corpus.
OBJECTIVE_SECTION_ID = "objectif-de-la-fiche"
OBJECTIVES_SECTION_ID = "objectifs-de-la-fiche"
OBJECTIVE_SECTION_IDS = (OBJECTIVE_SECTION_ID, OBJECTIVES_SECTION_ID)

# Largest per-theme count in the exam; FakeChatModel returns this many items by default.
MAX_ITEMS_PER_CALL = max(EXAM_QUESTION_COUNTS.values())
CRITIC_REJECTION_REASON = "sample-reason-rejected"


def make_question(
    *,
    text: str = "sample-question-text",
    theme: Theme = DROITS_ET_DEVOIRS,
    kind: QuestionKind = QuestionKind.KNOWLEDGE,
    correct_index: int = 0,
    sources: tuple[SourceRef, ...] = (SAMPLE_SOURCE,),
) -> Question:
    return Question(
        theme=theme,
        kind=kind,
        text=text,
        options=SAMPLE_OPTIONS,
        correct_index=correct_index,
        sources=sources,
    )


def generated_item(text: str, group: int) -> dict[str, object]:
    """One well-formed question as the model returns it, naming its 1-based passage group."""
    return {
        "text": text,
        "options": list(SAMPLE_OPTIONS),
        "correct_index": SAMPLE_CORRECT_INDEX,
        "group": group,
    }


def grouped_generation_reply(*items: tuple[int, str]) -> str:
    """A generation reply listing (group, text) items in the given order."""
    return json.dumps([generated_item(text, group) for group, text in items])


def generated_item_text(call_index: int, item_index: int) -> str:
    """Text of the default item FakeChatModel emits at (call, item)."""
    return f"sample-generated-question-{call_index:02d}-{item_index:02d}"


class FakeChatModel(BaseChatModel):  # type: ignore[explicit-any]
    """Fake chat model: replies with well-formed items naming groups 1, 2, ... in order.

    Item texts embed the call index, so each call yields distinct but reproducible questions.
    `items_by_call` overrides the item count per call; `replies` scripts exact replies by call
    index (later calls fall back to the default); `canned_text` replaces every reply.
    """

    items_per_call: int = MAX_ITEMS_PER_CALL
    items_by_call: tuple[int, ...] = ()
    replies: tuple[str, ...] = ()
    canned_text: str | None = None

    _received: list[list[BaseMessage]] = PrivateAttr(default_factory=list)

    @property
    def invocation_count(self) -> int:
        return len(self._received)

    def system_text(self, call_index: int) -> str:
        """The system message content of one call."""
        return "\n".join(
            str(m.content) for m in self._received[call_index] if isinstance(m, SystemMessage)
        )

    def human_text(self, call_index: int) -> str:
        """The human message content of one call."""
        return "\n".join(
            str(m.content) for m in self._received[call_index] if isinstance(m, HumanMessage)
        )

    def _reply_for(self, call_index: int) -> str:
        if self.canned_text is not None:
            return self.canned_text
        if call_index < len(self.replies):
            return self.replies[call_index]
        count = (
            self.items_by_call[call_index]
            if call_index < len(self.items_by_call)
            else self.items_per_call
        )
        return grouped_generation_reply(*((i + 1, generated_item_text(call_index, i)) for i in range(count)))

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        call_index = len(self._received)
        self._received.append(messages)
        message = AIMessage(content=self._reply_for(call_index))
        return ChatResult(generations=[ChatGeneration(message=message)])

    @property
    def _llm_type(self) -> str:
        return "sample-fake-chat-model"


def reject_all_critic(
    theme: Theme,
    questions: Sequence[Question],
    groups: Sequence[Sequence[ContentChunk]],
) -> list[Verdict]:
    """Fake CritiqueFn that rejects every question it sees."""
    return [Verdict(passed=False, reason=CRITIC_REJECTION_REASON) for _ in questions]


def make_mock_exam_questions() -> tuple[Question, ...]:
    """Forty distinct questions in the official per-theme counts, correct index cycling 0..3."""
    questions: list[Question] = []
    for theme, count in EXAM_QUESTION_COUNTS.items():
        for i in range(count):
            questions.append(
                make_question(
                    text=f"sample-question-{theme.slug}-{i:03d}",
                    theme=theme,
                    correct_index=len(questions) % OPTION_COUNT,
                )
            )
    return tuple(questions)


def wrong_index(question: Question) -> int:
    """An option index that is never the correct one."""
    return (question.correct_index + 1) % OPTION_COUNT


def answers_with_correct_count(questions: Sequence[Question], correct_count: int) -> list[int]:
    """Answer the first `correct_count` questions correctly and the rest wrongly."""
    return [
        q.correct_index if i < correct_count else wrong_index(q) for i, q in enumerate(questions)
    ]


@dataclass(frozen=True)
class LoggedAnswer:
    user_id: UserId
    theme: Theme
    question_id: str
    chosen_index: int
    correct_index: int
    is_correct: bool


@dataclass
class FakeAnswerLogger:
    """Fake AnswerLogger: records every answer instead of writing quiz_answers rows."""

    calls: list[LoggedAnswer] = field(default_factory=list)

    def __call__(
        self,
        user_id: UserId,
        theme: Theme,
        question_id: str,
        chosen_index: int,
        correct_index: int,
        is_correct: bool,
    ) -> None:
        self.calls.append(
            LoggedAnswer(user_id, theme, question_id, chosen_index, correct_index, is_correct)
        )


@dataclass
class FakeQuestionStore:
    """Fake QuestionStore: records each save call as (questions, prompt_version)."""

    saves: list[tuple[tuple[Question, ...], str]] = field(default_factory=list)

    def save(self, questions: Sequence[Question], prompt_version: str) -> None:
        self.saves.append((tuple(questions), prompt_version))

    @property
    def saved_questions(self) -> list[Question]:
        return [q for questions, _ in self.saves for q in questions]

    @property
    def saved_prompt_versions(self) -> set[str]:
        return {version for _, version in self.saves}


@dataclass
class FakeRejectionStore:
    """Fake RejectionStore: records each save call instead of writing rejected_questions rows."""

    saves: list[tuple[Rejection, ...]] = field(default_factory=list)

    def save(self, rejections: Sequence[Rejection]) -> None:
        self.saves.append(tuple(rejections))

    @property
    def rejections(self) -> list[Rejection]:
        return [r for rejections in self.saves for r in rejections]


def make_content_chunk(
    theme: Theme, index: int, chunk_index: int = 0, section_id: str | None = None
) -> ContentChunk:
    """One retrieved chunk of section `index`, with text unique to (theme, index, chunk_index)."""
    return ContentChunk(
        chunk=Chunk(
            theme=theme,
            page_slug=f"sample-page-{index:03d}",
            section_id=section_id if section_id is not None else f"section-{index:03d}",
            chunk_index=chunk_index,
            content_hash=f"sample-hash-{theme.slug}-{index:03d}-{chunk_index:03d}",
            text=f"sample passage {index}.{chunk_index} about {theme.slug}.",
        ),
        similarity=0.9,
    )


class FakeRetriever:
    """Fake SearchFn: records each (query, theme) and returns sections ranked 0, 1, ... for that theme.

    Chunks interleave across sections (s0c0, s1c0, ..., s0c1); `objective_sections` maps rank to objective ID."""

    def __init__(
        self,
        section_count: int = DEFAULT_SECTION_COUNT,
        chunks_per_section: int = 1,
        objective_sections: Mapping[int, str] | None = None,
    ) -> None:
        self.section_count = section_count
        self.chunks_per_section = chunks_per_section
        self.objective_sections = dict(objective_sections or {})
        self.calls: list[tuple[str, Theme | None]] = []

    def __call__(self, query: str, theme: Theme | None) -> list[ContentChunk]:
        self.calls.append((query, theme))
        assert theme is not None, "assessment retrieval must always be scoped to a theme"
        return self.chunks(theme)

    def chunks(self, theme: Theme) -> list[ContentChunk]:
        """Every returned chunk, in retrieval (similarity) order."""
        return [
            self._chunk(theme, section, part)
            for part in range(self.chunks_per_section)
            for section in range(self.section_count)
        ]

    def section_chunks(self, theme: Theme, section: int) -> list[ContentChunk]:
        """The chunks of the section ranked `section` (0 is the best match)."""
        return [self._chunk(theme, section, part) for part in range(self.chunks_per_section)]

    def objective_chunks(self, theme: Theme) -> list[ContentChunk]:
        """Every chunk of the course-objective sections."""
        return [c for rank in self.objective_sections for c in self.section_chunks(theme, rank)]

    def content_source_refs(self, theme: Theme) -> list[SourceRef]:
        """Distinct non-objective sections in retrieval rank order."""
        objective = {_source_ref(c) for c in self.objective_chunks(theme)}
        return [ref for ref in self.ranked_source_refs(theme) if ref not in objective]

    def ranked_source_refs(self, theme: Theme) -> list[SourceRef]:
        """Distinct sections in retrieval rank order."""
        return list(dict.fromkeys(_source_ref(c) for c in self.chunks(theme)))

    def source_refs(self, theme: Theme) -> set[SourceRef]:
        return set(self.ranked_source_refs(theme))

    def _chunk(self, theme: Theme, section: int, part: int) -> ContentChunk:
        return make_content_chunk(theme, section, part, self.objective_sections.get(section))


def _source_ref(chunk: ContentChunk) -> SourceRef:
    return SourceRef(page_slug=chunk.chunk.page_slug, section_id=chunk.chunk.section_id)


# A (query call index, rank) position in one theme's sub-topic search results.
Slot = tuple[int, int]


class FakeQueryAwareRetriever:
    """Fake SearchFn: call q of a theme returns distinct sections at slots (q, 0), (q, 1), ...

    `shared_slots` makes a slot repeat another slot's section; `objective_slots` gives it an objective ID."""

    def __init__(
        self,
        # Section count for each call in order; later calls get default_section_count.
        sections_per_query: Sequence[int] = (),
        default_section_count: int = DEFAULT_SECTION_COUNT,
        shared_slots: Mapping[Slot, Slot] | None = None,
        objective_slots: Mapping[Slot, str] | None = None,
    ) -> None:
        self.sections_per_query = tuple(sections_per_query)
        self.default_section_count = default_section_count
        self.shared_slots = dict(shared_slots or {})
        self.objective_slots = dict(objective_slots or {})
        self.calls: list[tuple[str, Theme | None]] = []

    def __call__(self, query: str, theme: Theme | None) -> list[ContentChunk]:
        assert theme is not None, "assessment retrieval must always be scoped to a theme"
        call = self.call_count(theme)
        self.calls.append((query, theme))
        return [self._chunk(theme, (call, rank)) for rank in range(self.section_count(call))]

    def call_count(self, theme: Theme) -> int:
        return sum(1 for _, called in self.calls if called == theme)

    def queries(self, theme: Theme) -> list[str]:
        """The queries sent for `theme`, in call order."""
        return [query for query, called in self.calls if called == theme]

    def section_count(self, call: int) -> int:
        if call < len(self.sections_per_query):
            return self.sections_per_query[call]
        return self.default_section_count

    def slot_ref(self, slot: Slot) -> SourceRef:
        """The section returned at `slot`, after any sharing or objective remapping."""
        return _source_ref(self._chunk(_ANY_THEME, slot))

    def slot_chunk(self, theme: Theme, slot: Slot) -> ContentChunk:
        return self._chunk(theme, slot)

    def _chunk(self, theme: Theme, slot: Slot) -> ContentChunk:
        call, rank = self.shared_slots.get(slot, slot)
        section_id = self.objective_slots.get(slot, f"section-r{rank:03d}")
        return ContentChunk(
            chunk=Chunk(
                theme=theme,
                page_slug=f"sample-page-q{call:02d}",
                section_id=section_id,
                chunk_index=0,
                content_hash=f"sample-hash-{theme.slug}-q{call:02d}-{section_id}",
                text=f"sample passage q{call}.r{rank} ({section_id}) about {theme.slug}.",
            ),
            similarity=0.9,
        )


# Section identity ignores theme, so slot_ref needs no theme from the caller.
_ANY_THEME = DROITS_ET_DEVOIRS
