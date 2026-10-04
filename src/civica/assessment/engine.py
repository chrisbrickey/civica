"""Assessment engine: builds quizzes and mock exams from retrieved corpus passages.

Per theme: one retrieval of chunks, one LLM call.
Sources and kind are assigned deterministically in code, not by the model.
"""

import logging
import re
from collections.abc import Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from civica.assessment.scoring import AnswerLogger, MockExam, Quiz
from civica.assessment.store import PostgresQuestionStore, QuestionStore
from civica.domain.question import Question, QuestionKind
from civica.domain.source_ref import SourceRef
from civica.domain.themes import EXAM_QUESTION_COUNTS, EXAM_SCENARIO_COUNTS, Theme
from civica.domain.user import UserId
from civica.llm.client import QUIZ_PROFILE, get_chat_model
from civica.progress.quiz_log import log_answer
from civica.retrieval.content import ContentChunk, SearchFn, search

logger = logging.getLogger(__name__)

# Bump by hand whenever PROMPTS changes. Stored with every question for later comparison.
PROMPT_VERSION = "quiz-v001"

PROMPTS: dict[str, str] = {
    "system": (
        "Tu es un concepteur de questions pour l'examen civique de naturalisation francaise. "
        "Utilise uniquement les passages officiels fournis, sans connaissances externes. "
        "Redige en francais des questions a choix multiples avec exactement 4 options, "
        "dont une seule est correcte. Les mauvaises options doivent etre plausibles mais fausses. "
        "Une question de connaissance porte sur un fait du passage. "
        "Une question de mise en situation decrit un cas concret a resoudre avec le passage. "
        "Reponds uniquement par un tableau JSON, sans texte autour, de la forme "
        '[{"text": "...", "options": ["...", "...", "...", "..."], "correct_index": 0}]. '
        "correct_index est l'indice (0 a 3) de la bonne option."
    ),
    "human": (
        "Theme : {theme}\n\n"
        "Ecris exactement {count} questions, une par groupe de passages numerote, dans l'ordre "
        "des groupes. Chaque question s'appuie uniquement sur son groupe.\n\n{groups}"
    ),
}

# Enough chunks to fill up to 11 passage groups; the floor drops weakly related chunks.
_RETRIEVAL_K = 30
_MINIMUM_SIMILARITY = 0.3

_FENCE_PATTERN = re.compile(r"^```[a-zA-Z]*\s*(.*?)\s*```$", re.DOTALL)

SectionKey = tuple[str, str]


class QuestionGenerationError(Exception):
    """Raised when questions cannot be generated, so nothing is shown or saved."""


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
    body = reply.strip()
    fenced = _FENCE_PATTERN.match(body)
    if fenced:
        body = fenced.group(1)
    try:
        items = _ITEMS_ADAPTER.validate_json(body)
    except ValidationError as error:
        raise QuestionGenerationError(f"Model reply is not a valid question list: {error}") from error
    if len(items) < n:
        raise QuestionGenerationError(f"Model returned {len(items)} questions, expected {n}")
    return items[:n]


def _generate_for_theme(
    theme: Theme,
    n: int,
    scenario_count: int,
    *,
    retriever: SearchFn,
    client: BaseChatModel,
) -> tuple[Question, ...]:
    chunks = retriever(theme.display_name_fr, theme)
    if not chunks:
        raise QuestionGenerationError(f"No corpus passages retrieved for theme {theme.slug!r}")

    groups = _group_chunks(chunks, n)
    kinds = _kinds(n, scenario_count)
    response = client.invoke(_build_messages(theme, groups, kinds))
    if QUIZ_PROFILE.was_truncated(response):
        logger.warning("Question generation for %r hit the token cap; reply may be cut off.", theme.slug)

    items = _parse_items(response.text, n)
    try:
        return tuple(
            Question(
                theme=theme,
                kind=kind,
                text=item.text,
                options=item.options,
                correct_index=item.correct_index,
                sources=_group_sources(group),
            )
            for item, group, kind in zip(items, groups, kinds, strict=True)
        )
    except ValidationError as error:
        raise QuestionGenerationError(f"Generated question is invalid: {error}") from error


def generate_quiz(
    user_id: UserId,
    theme: Theme,
    n: int = 5,
    *,
    retriever: SearchFn = _grounded_search,
    chat_client: BaseChatModel | None = None,
    question_store: QuestionStore | None = None,
    answer_logger: AnswerLogger = log_answer,
) -> Quiz:
    """Generate and save `n` knowledge questions for `theme`. Raises QuestionGenerationError."""
    if n < 1:
        raise ValueError("n must be at least 1")
    client = chat_client if chat_client is not None else get_chat_model(QUIZ_PROFILE)
    store = question_store if question_store is not None else PostgresQuestionStore()

    questions = _generate_for_theme(theme, n, 0, retriever=retriever, client=client)
    store.save(questions, PROMPT_VERSION)
    return Quiz(user_id=user_id, questions=questions, answer_logger=answer_logger)


def generate_mock_exam(
    user_id: UserId,
    *,
    retriever: SearchFn = _grounded_search,
    chat_client: BaseChatModel | None = None,
    question_store: QuestionStore | None = None,
    answer_logger: AnswerLogger = log_answer,
) -> MockExam:
    """Generate the 40-question exam in official weighting, saving only if all themes succeed."""
    client = chat_client if chat_client is not None else get_chat_model(QUIZ_PROFILE)
    store = question_store if question_store is not None else PostgresQuestionStore()

    questions: list[Question] = []
    for theme, count in EXAM_QUESTION_COUNTS.items():
        questions.extend(
            _generate_for_theme(
                theme, count, EXAM_SCENARIO_COUNTS[theme], retriever=retriever, client=client
            )
        )
    store.save(questions, PROMPT_VERSION)
    return MockExam(user_id=user_id, questions=tuple(questions), answer_logger=answer_logger)
