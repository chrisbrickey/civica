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
from civica.assessment.errors import QuestionGenerationError
from civica.assessment.prompts import PROMPT_VERSION, PROMPTS
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
from civica.domain.themes import EXAM_QUESTION_COUNTS, EXAM_SCENARIO_COUNTS, Theme
from civica.domain.user import UserId
from civica.llm.client import QUIZ_PROFILE, get_chat_model
from civica.progress.quiz_log import log_answer
from civica.retrieval.content import ContentChunk, SearchFn, search

__all__ = [
    "MAX_REFINE_ATTEMPTS",
    "PROMPT_VERSION",
    "QuestionGenerationError",
    "generate_mock_exam",
    "generate_quiz",
]

logger = logging.getLogger(__name__)

# Regenerations allowed for questions the critic rejects, after the first generation.
MAX_REFINE_ATTEMPTS = 2

# Enough chunks to fill up to 11 passage groups; the floor drops weakly related chunks.
_RETRIEVAL_K = 30
_MINIMUM_SIMILARITY = 0.3

SectionKey = tuple[str, str]

class _GeneratedItem(BaseModel):  # type: ignore[explicit-any]
    """One multiple choice question (MCQ) as the model returns it, validated at the edge."""

    model_config = ConfigDict(frozen=True)

    text: str = Field(min_length=1)
    options: tuple[str, str, str, str]
    correct_index: int = Field(ge=0, le=3)


_ITEMS_ADAPTER = TypeAdapter(list[_GeneratedItem])


def _grounded_search(query: str, theme: Theme | None) -> list[ContentChunk]:
    """Default retriever: search() sized for a full exam theme, with a similarity floor."""
    return search(query, theme, k=_RETRIEVAL_K, min_similarity=_MINIMUM_SIMILARITY)


def _group_chunks(chunks: Sequence[ContentChunk], n: int) -> list[list[ContentChunk]]:
    """Split chunks into n passage groups by distinct section, reusing sections if too few."""
    by_section: dict[SectionKey, list[ContentChunk]] = {}
    for chunk in chunks:
        key = (chunk.chunk.page_slug, chunk.chunk.section_id)
        by_section.setdefault(key, []).append(chunk)
    sections = list(by_section.values())

    groups: list[list[ContentChunk]] = [[] for _ in range(n)]
    if len(sections) >= n:
        for i, section in enumerate(sections):
            groups[i % n].extend(section)
    else:
        for i in range(n):
            groups[i] = sections[i % len(sections)]
    return groups


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
    human = PROMPTS["human"].format(
        theme=theme.display_name_fr, count=len(groups), groups="\n\n".join(blocks)
    )
    return [SystemMessage(content=PROMPTS["system"]), HumanMessage(content=human)]


def _parse_items(reply: str, n: int) -> list[_GeneratedItem]:
    """Parse the model reply (tolerating a code fence) and keep the first n items."""
    try:
        items = _ITEMS_ADAPTER.validate_json(strip_code_fence(reply))
    except ValidationError as error:
        raise QuestionGenerationError(f"Model reply is not a valid question list: {error}") from error
    if len(items) < n:
        raise QuestionGenerationError(f"Model returned {len(items)} questions, expected {n}")
    return items[:n]


def _generate_batch(
    theme: Theme,
    groups: Sequence[Sequence[ContentChunk]],
    kinds: Sequence[QuestionKind],
    client: BaseChatModel,
) -> list[Question]:
    """One generation call: one question per group, in group order."""
    response = client.invoke(_build_messages(theme, groups, kinds))
    if QUIZ_PROFILE.was_truncated(response):
        logger.warning("Question generation for %r hit the token cap; reply may be cut off.", theme.slug)

    items = _parse_items(response.text, len(groups))
    try:
        return [
            Question(
                theme=theme,
                kind=kind,
                text=item.text,
                options=item.options,
                correct_index=item.correct_index,
                sources=_group_sources(group),
            )
            for item, group, kind in zip(items, groups, kinds, strict=True)
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
                rejections.append(Rejection(PROMPT_VERSION, theme, verdict.reason, attempt))
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
    chunks = retriever(theme.display_name_fr, theme)
    if not chunks:
        raise QuestionGenerationError(f"No corpus passages retrieved for theme {theme.slug!r}")

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
    store.save(questions, PROMPT_VERSION)
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
    store.save(questions, PROMPT_VERSION)
    return MockExam(user_id=user_id, questions=tuple(questions), answer_logger=answer_logger)
