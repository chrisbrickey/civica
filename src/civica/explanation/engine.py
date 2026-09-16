"""Explanation engine: answers a learner's question using only retrieved corpus passages.

Explanations are in English and retain the French vocabulary the exam uses.
"""

import logging

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict

from civica.domain.themes import Theme
from civica.llm.client import EXPLANATION_PROFILE, get_chat_model
from civica.retrieval.content import ContentChunk, SearchFn, search

logger = logging.getLogger(__name__)

PROMPTS: dict[str, str] = {
    "system": (
        "You are a coach helping a learner prepare for the French naturalization civics exam. "
        "Use only the provided passages. Do not draw on outside knowledge. "
        "Respond in English, retaining key French vocabulary the learner will need to recognize on the exam."
    ),
    "human": "Learner question: {question}\n\nOfficial passages:\n{passages}",
}

_INSUFFICIENT_MATERIAL_MESSAGE = (
    "There is insufficient official material to answer this question."
)


class Explanation(BaseModel):  # type: ignore[explicit-any]
    """An explanation for a learner question, paired with the passages that grounded it."""

    model_config = ConfigDict(frozen=True)

    text: str
    context_chunks: tuple[ContentChunk, ...]


def _build_human_message(user_question: str, chunks: list[ContentChunk]) -> HumanMessage:
    passages = "\n\n".join(chunk.chunk.text for chunk in chunks)
    content = PROMPTS["human"].format(question=user_question, passages=passages)
    return HumanMessage(content=content)


def explain(
    user_question: str,
    theme: Theme,
    *,
    retriever: SearchFn = search,
    chat_client: BaseChatModel | None = None,
) -> Explanation:
    """Answer a learner's question about `theme`, grounded only in retrieved passages.

    With no passages, returns early without calling the LLM so it cannot free-associate.
    """
    chunks = retriever(user_question, theme)
    if not chunks:
        return Explanation(text=_INSUFFICIENT_MATERIAL_MESSAGE, context_chunks=())

    client = chat_client if chat_client is not None else get_chat_model(EXPLANATION_PROFILE)
    messages: list[BaseMessage] = [
        SystemMessage(content=PROMPTS["system"]),
        _build_human_message(user_question, chunks),
    ]
    response = client.invoke(messages)
    if EXPLANATION_PROFILE.was_truncated(response):
        logger.warning(
            "Explanation for %r hit the generation token cap and may end mid-sentence.",
            user_question,
        )
    # .text reads only the text blocks; response.content may be a list of typed blocks.
    return Explanation(text=response.text, context_chunks=tuple(chunks))
