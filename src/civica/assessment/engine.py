"""Assessment engine: builds quizzes and mock exams from retrieved corpus passages.

Per theme: one retrieval of chunks, one generation call, and an optional critic pass
Sources and kind are assigned deterministically in code, not by the model.

Questions rejected by the critic (off by default) get regenerated up to MAX_REFINE_ATTEMPTS.
Every rejection is logged. Pass e.g. LlmCritic() as critic to enable it.
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from civica.assessment.critic import CritiqueFn
from civica.assessment.errors import InsufficientMaterialError, QuestionGenerationError
from civica.assessment.prompts import QUIZ_PROMPTS
from civica.assessment.replies import strip_code_fence
from civica.assessment.scoring import AnswerLogger, MockExam, Quiz
from civica.assessment.store import (
    PostgresQuestionStore,
    PostgresRejectionStore,
    QuestionStore,
    Rejection,
    RejectionStore,
)
from civica.domain.question import Question, QuestionKind
from civica.domain.source_ref import SourceRef
from civica.domain.themes import (
    EXAM_QUESTION_COUNTS,
    EXAM_SCENARIO_COUNTS,
    HISTOIRE_GEOGRAPHIE_ET_CULTURE,
    Theme,
)
from civica.domain.user import UserId
from civica.llm.client import QUIZ_PROFILE, get_chat_model
from civica.progress.quiz_log import log_answer
from civica.retrieval.content import ContentChunk, SearchFn, search

__all__ = [
    "MAX_REFINE_ATTEMPTS",
    "QuestionGenerationError",
    "generate_mock_exam",
    "generate_quiz",
]

logger = logging.getLogger(__name__)

_RETRIEVAL_K = 40
_MINIMUM_SIMILARITY = 0.3

# Regenerations allowed for questions the critic rejects, after the first generation.
MAX_REFINE_ATTEMPTS = 2

# Course-objective intros carry no civic facts to test.
_OBJECTIVE_SECTION_IDS = frozenset({"objectif-de-la-fiche", "objectifs-de-la-fiche"})

SectionKey = tuple[str, str]

# The theme name is too broad to clear the similarity floor, so its sub-topics are searched
# separately (these queries were chosen by measurement: 45 usable sections vs 7).
_SUBTOPIC_QUERIES: dict[Theme, tuple[str, ...]] = {
    HISTOIRE_GEOGRAPHIE_ET_CULTURE: (
        "Histoire de France",
        "Géographie de la France",
        "Culture française",
    ),
}


class _GeneratedItem(BaseModel):  # type: ignore[explicit-any]
    """One multiple choice question (MCQ) as the model returns it, validated at the edge."""

    model_config = ConfigDict(frozen=True)

    group: int = Field(ge=1, strict=True)
    text: str = Field(min_length=1)
    options: tuple[str, str, str, str]
    correct_index: int = Field(ge=0, le=3)


_ITEMS_ADAPTER = TypeAdapter(list[_GeneratedItem])


def _grounded_search(query: str, theme: Theme | None) -> list[ContentChunk]:
    """Default retriever: search() sized for a full exam theme, with a similarity floor."""
    return search(query, theme, k=_RETRIEVAL_K, min_similarity=_MINIMUM_SIMILARITY)


def _section_key(chunk: ContentChunk) -> SectionKey:
    return (chunk.chunk.page_slug, chunk.chunk.section_id)


def _sections(chunks: Sequence[ContentChunk]) -> list[list[ContentChunk]]:
    """Group chunks into sections, in order of first appearance."""
    by_section: dict[SectionKey, list[ContentChunk]] = {}
    for chunk in chunks:
        by_section.setdefault(_section_key(chunk), []).append(chunk)
    return list(by_section.values())


def _interleave(per_query: Sequence[Sequence[list[ContentChunk]]]) -> list[ContentChunk]:
    """Take sections round-robin by rank across queries; a section stays at its first position."""
    seen: set[SectionKey] = set()
    chunks: list[ContentChunk] = []
    for rank in range(max(map(len, per_query), default=0)):
        for sections in per_query:
            if rank < len(sections) and _section_key(sections[rank][0]) not in seen:
                seen.add(_section_key(sections[rank][0]))
                chunks.extend(sections[rank])
    return chunks


def _group_chunks(chunks: Sequence[ContentChunk], n: int) -> list[list[ContentChunk]]:
    """One section per group in retrieval order; extras dropped, reused if too few."""
    sections = _sections(chunks)
    return [sections[i % len(sections)] for i in range(n)]


def _group_sources(group: Sequence[ContentChunk]) -> tuple[SourceRef, ...]:
    refs = dict.fromkeys(
        SourceRef(page_slug=c.chunk.page_slug, section_id=c.chunk.section_id) for c in group
    )
    return tuple(refs)


def _kinds(n: int, scenario_count: int) -> list[QuestionKind]:
    """Knowledge first, then scenario for the last `scenario_count` groups."""
    return [
        QuestionKind.SCENARIO if i >= n - scenario_count else QuestionKind.KNOWLEDGE
        for i in range(n)
    ]


def _build_messages(
    theme: Theme, groups: Sequence[Sequence[ContentChunk]], kinds: Sequence[QuestionKind]
) -> list[BaseMessage]:
    blocks = []
    for number, (group, kind) in enumerate(zip(groups, kinds, strict=True), start=1):
        label = "mise en situation" if kind is QuestionKind.SCENARIO else "connaissance"
        passages = "\n".join(c.chunk.text for c in group)
        blocks.append(f"Groupe {number} (question de {label}) :\n{passages}")
    human = QUIZ_PROMPTS.human(
        theme=theme.display_name_fr, count=len(groups), groups="\n\n".join(blocks)
    )
    return [SystemMessage(content=QUIZ_PROMPTS.system), HumanMessage(content=human)]


def _parse_items(reply: str) -> list[_GeneratedItem]:
    """Parse the model reply, tolerating a code fence."""
    try:
        return _ITEMS_ADAPTER.validate_json(strip_code_fence(reply))
    except ValidationError as error:
        raise QuestionGenerationError(f"Model reply is not a valid question list: {error}") from error


def _items_by_group(
    theme: Theme,
    groups: Sequence[Sequence[ContentChunk]],
    kinds: Sequence[QuestionKind],
    client: BaseChatModel,
) -> dict[int, _GeneratedItem]:
    """One model call; the first item naming each group 1..len(groups). Others are ignored."""
    response = client.invoke(_build_messages(theme, groups, kinds))
    if QUIZ_PROFILE.was_truncated(response):
        logger.warning("Question generation for %r hit the token cap; reply may be cut off.", theme.slug)

    matched: dict[int, _GeneratedItem] = {}
    for item in _parse_items(response.text):
        if item.group <= len(groups):
            matched.setdefault(item.group, item)
    return matched


def _generate_batch(
    theme: Theme,
    groups: Sequence[Sequence[ContentChunk]],
    kinds: Sequence[QuestionKind],
    client: BaseChatModel,
) -> list[Question]:
    """One question per group, in group order; groups the model skipped get one retry call."""
    matched = _items_by_group(theme, groups, kinds, client)
    missing = [i for i in range(1, len(groups) + 1) if i not in matched]
    if missing:
        logger.warning("Model skipped groups %s for %r; retrying them once.", missing, theme.slug)
        retried = _items_by_group(
            theme, [groups[i - 1] for i in missing], [kinds[i - 1] for i in missing], client
        )
        for new, original in enumerate(missing, start=1):
            if new in retried:
                matched[original] = retried[new]
        still_missing = [i for i in missing if i not in matched]
        if still_missing:
            raise QuestionGenerationError(f"Model returned no question for groups {still_missing} after a retry")

    try:
        return [
            Question(
                theme=theme,
                kind=kind,
                text=matched[number].text,
                options=matched[number].options,
                correct_index=matched[number].correct_index,
                sources=_group_sources(group),
            )
            for number, (group, kind) in enumerate(zip(groups, kinds, strict=True), start=1)
        ]
    except ValidationError as error:
        raise QuestionGenerationError(f"Generated question is invalid: {error}") from error


@dataclass(frozen=True)
class _Collaborators:
    """The injected services one theme's generation uses."""

    client: BaseChatModel
    critic: CritiqueFn | None
    rejection_store: RejectionStore


def _refine(
    theme: Theme,
    questions: list[Question],
    groups: Sequence[Sequence[ContentChunk]],
    kinds: Sequence[QuestionKind],
    services: _Collaborators,
    critic: CritiqueFn,
) -> tuple[Question, ...]:
    """Critique, log rejections, and regenerate only failed items; drop any still failing."""
    accepted: dict[int, Question] = {}
    pending = list(range(len(questions)))
    for attempt in range(1, MAX_REFINE_ATTEMPTS + 2):
        verdicts = critic(theme, questions, [groups[i] for i in pending])
        failed: list[int] = []
        rejections: list[Rejection] = []
        for position, question, verdict in zip(pending, questions, verdicts, strict=True):
            if verdict.passed:
                accepted[position] = question
            else:
                failed.append(position)
                rejections.append(Rejection(QUIZ_PROMPTS.version, theme, verdict.reason, attempt))
        if rejections:
            services.rejection_store.save(rejections)
        if not failed or attempt > MAX_REFINE_ATTEMPTS:
            break
        pending = failed
        questions = _generate_batch(
            theme, [groups[i] for i in pending], [kinds[i] for i in pending], services.client
        )
    return tuple(accepted[position] for position in sorted(accepted))


def _generate_for_theme(
    theme: Theme,
    n: int,
    scenario_count: int,
    *,
    retriever: SearchFn,
    services: _Collaborators,
) -> tuple[Question, ...]:
    queries = _SUBTOPIC_QUERIES.get(theme, (theme.display_name_fr,))
    ranked = _interleave([_sections(retriever(query, theme)) for query in queries])
    chunks = [c for c in ranked if c.chunk.section_id not in _OBJECTIVE_SECTION_IDS]
    if not chunks:
        raise InsufficientMaterialError(f"No testable corpus passages retrieved for theme {theme.slug!r}")

    groups = _group_chunks(chunks, n)
    kinds = _kinds(n, scenario_count)
    questions = _generate_batch(theme, groups, kinds, services.client)
    if services.critic is None:
        return tuple(questions)
    return _refine(theme, questions, groups, kinds, services, services.critic)


def _collaborators(
    chat_client: BaseChatModel | None,
    critic: CritiqueFn | None,
    rejection_store: RejectionStore | None,
) -> _Collaborators:
    return _Collaborators(
        client=chat_client if chat_client is not None else get_chat_model(QUIZ_PROFILE),
        critic=critic,
        rejection_store=rejection_store if rejection_store is not None else PostgresRejectionStore(),
    )


def generate_quiz(
    user_id: UserId,
    theme: Theme,
    n: int = 5,
    *,
    retriever: SearchFn = _grounded_search,
    chat_client: BaseChatModel | None = None,
    question_store: QuestionStore | None = None,
    answer_logger: AnswerLogger = log_answer,
    critic: CritiqueFn | None = None,
    rejection_store: RejectionStore | None = None,
) -> Quiz:
    """Generate and save up to `n` knowledge questions for `theme`; an enabled critic may leave it short.

    Raises QuestionGenerationError. The critic is off by default; pass e.g. LlmCritic() to enable it.
    """
    if n < 1:
        raise ValueError("n must be at least 1")
    services = _collaborators(chat_client, critic, rejection_store)
    store = question_store if question_store is not None else PostgresQuestionStore()

    questions = _generate_for_theme(theme, n, 0, retriever=retriever, services=services)
    if not questions:
        raise QuestionGenerationError(f"No question passed the critic for theme {theme.slug!r}")
    store.save(questions, QUIZ_PROMPTS.version)
    return Quiz(user_id=user_id, questions=questions, answer_logger=answer_logger)


def generate_mock_exam(
    user_id: UserId,
    *,
    retriever: SearchFn = _grounded_search,
    chat_client: BaseChatModel | None = None,
    question_store: QuestionStore | None = None,
    answer_logger: AnswerLogger = log_answer,
    critic: CritiqueFn | None = None,
    rejection_store: RejectionStore | None = None,
) -> MockExam:
    """Generate the 40-question exam in official weighting, saving only if all themes are full.

    The critic is off by default; pass e.g. LlmCritic() to enable it.
    A theme left short by the critic raises at once, so remaining themes cost no LLM calls.
    """
    services = _collaborators(chat_client, critic, rejection_store)
    store = question_store if question_store is not None else PostgresQuestionStore()

    questions: list[Question] = []
    for theme, count in EXAM_QUESTION_COUNTS.items():
        themed = _generate_for_theme(
            theme, count, EXAM_SCENARIO_COUNTS[theme], retriever=retriever, services=services
        )
        if len(themed) < count:
            raise QuestionGenerationError(
                f"Only {len(themed)} of {count} questions passed the critic for theme {theme.slug!r}"
            )
        questions.extend(themed)
    store.save(questions, QUIZ_PROMPTS.version)
    return MockExam(user_id=user_id, questions=tuple(questions), answer_logger=answer_logger)
