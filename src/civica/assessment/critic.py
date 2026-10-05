"""Question critic (Step 9B): a corpus-grounded correctness gate for generated questions."""

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

from civica.assessment.errors import QuestionGenerationError
from civica.assessment.prompts import PROMPTS
from civica.assessment.replies import strip_code_fence
from civica.domain.question import Question
from civica.domain.themes import Theme
from civica.llm.client import QUIZ_PROFILE, get_chat_model
from civica.retrieval.content import ContentChunk

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Verdict:
    """The critic's judgment of one question: accepted, or rejected with a reason."""

    passed: bool
    reason: str


class CritiqueFn(Protocol):
    """Judges a batch of questions against their passage groups, one verdict per question."""

    def __call__(
        self,
        theme: Theme,
        questions: Sequence[Question],
        groups: Sequence[Sequence[ContentChunk]],
    ) -> list[Verdict]: ...


class _VerdictModel(BaseModel):  # type: ignore[explicit-any]
    """One verdict as the model returns it, validated at the edge."""

    model_config = ConfigDict(frozen=True, strict=True)

    passed: bool
    reason: str = ""


_VERDICTS_ADAPTER = TypeAdapter(list[_VerdictModel])


def _format_item(number: int, question: Question, group: Sequence[ContentChunk]) -> str:
    options = "\n".join(f"{i}. {option}" for i, option in enumerate(question.options))
    passages = "\n".join(c.chunk.text for c in group)
    return (
        f"Question {number} :\n{question.text}\nOptions :\n{options}\n"
        f"Reponse indiquee : {question.correct_index}\n"
        f"Passages de la question {number} :\n{passages}"
    )


def _build_messages(
    theme: Theme, questions: Sequence[Question], groups: Sequence[Sequence[ContentChunk]]
) -> list[BaseMessage]:
    items = "\n\n".join(
        _format_item(number, question, group)
        for number, (question, group) in enumerate(zip(questions, groups, strict=True), start=1)
    )
    human = PROMPTS["critic_human"].format(
        theme=theme.display_name_fr, count=len(questions), items=items
    )
    return [SystemMessage(content=PROMPTS["critic_system"]), HumanMessage(content=human)]


def _parse_verdicts(reply: str, n: int) -> list[Verdict]:
    try:
        parsed = _VERDICTS_ADAPTER.validate_json(strip_code_fence(reply))
    except ValidationError as error:
        raise QuestionGenerationError(f"Critic reply is not a valid verdict list: {error}") from error
    if len(parsed) != n:
        raise QuestionGenerationError(f"Critic returned {len(parsed)} verdicts, expected {n}")
    return [Verdict(passed=v.passed, reason=v.reason) for v in parsed]


class LlmCritic:
    """CritiqueFn backed by one chat call per batch. The client is built on first use."""

    def __init__(self, chat_client: BaseChatModel | None = None) -> None:
        self._chat_client = chat_client

    def __call__(
        self,
        theme: Theme,
        questions: Sequence[Question],
        groups: Sequence[Sequence[ContentChunk]],
    ) -> list[Verdict]:
        if self._chat_client is None:
            self._chat_client = get_chat_model(QUIZ_PROFILE)
        response = self._chat_client.invoke(_build_messages(theme, questions, groups))
        if QUIZ_PROFILE.was_truncated(response):
            logger.warning("Critique for %r hit the token cap; reply may be cut off.", theme.slug)
        return _parse_verdicts(response.text, len(questions))
