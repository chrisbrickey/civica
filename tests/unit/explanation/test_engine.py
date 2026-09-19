"""Unit tests for civica.explanation.engine

No network calls. No DB spinup."""

import logging
from typing import Any

import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import PrivateAttr

from civica.domain.chunk import Chunk
from civica.domain.themes import PRINCIPES_ET_VALEURS_DE_LA_REPUBLIQUE, Theme
from civica.explanation.engine import MINIMUM_SIMILARITY, Explanation, explain
from civica.retrieval.content import ContentChunk

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SAMPLE_THEME = PRINCIPES_ET_VALEURS_DE_LA_REPUBLIQUE
_SAMPLE_QUESTION = "What does sample-value mean in the Republic?"
_SAMPLE_RESPONSE_TEXT = "sample explanation text"

_SAMPLE_PASSAGE_ONE = "sample passage one about civic values."
_SAMPLE_PASSAGE_TWO = "sample passage two about civic duties."

_BELOW_FLOOR_SIMILARITY = MINIMUM_SIMILARITY - 0.1
_ABOVE_FLOOR_SIMILARITY = MINIMUM_SIMILARITY + 0.1

_BELOW_FLOOR_PASSAGE = "sample passage below the similarity floor."
_ABOVE_FLOOR_PASSAGE = "sample passage above the similarity floor."
_BOUNDARY_PASSAGE = "sample passage exactly at the similarity floor."

# Hard product contracts. Pinned here as literal strings so that a passing assertion proves
# the implementation actually contains this text, rather reusing the same constant.
_GROUNDING_CONTRACT = "Use only the provided passages. Do not draw on outside knowledge."
_ENGLISH_CONTRACT = "Respond in English, retaining key French vocabulary"

_INSUFFICIENT_MATERIAL_SUBSTRING = "insufficient official material"

# Providers return either a plain string or a list of typed blocks.
# When thinking is on, the text the learner should see is one block among several.
_BLOCK_STRUCTURED_CONTENT = [
    {"type": "thinking", "thinking": "sample internal reasoning"},
    {"type": "text", "text": _SAMPLE_RESPONSE_TEXT},
]

_TRUNCATED_METADATA = {"stop_reason": "max_tokens"}
_COMPLETE_METADATA = {"stop_reason": "end_turn"}


# ---------------------------------------------------------------------------
# Fakes (dependency injection, not monkeypatching)
# ---------------------------------------------------------------------------


def _make_content_chunk(text: str, page_slug: str, similarity: float = 0.9) -> ContentChunk:
    return ContentChunk(
        chunk=Chunk(
            theme=_SAMPLE_THEME,
            page_slug=page_slug,
            section_id="sample-section",
            chunk_index=0,
            content_hash="sample-hash",
            text=text,
        ),
        similarity=similarity,
    )


class _RecordingRetriever:
    """Fake SearchFn: records the (query, theme) it was called with, returns a fixed result."""

    def __init__(self, result: list[ContentChunk]) -> None:
        self.result = result
        self.calls: list[tuple[str, Theme | None]] = []

    def __call__(self, query: str, theme: Theme | None) -> list[ContentChunk]:
        self.calls.append((query, theme))
        return self.result


class _RecordingChatModel(BaseChatModel):  # type: ignore[explicit-any]
    """Fake chat model: records every prompt it receives, returns a canned AIMessage.

    Content and metadata are settable so tests can exercise block-structured and truncated responses.
    """

    canned_content: str | list[dict[str, str]] = _SAMPLE_RESPONSE_TEXT
    canned_metadata: dict[str, str] = {}

    _received: list[list[BaseMessage]] = PrivateAttr(default_factory=list)

    @property
    def invocation_count(self) -> int:
        return len(self._received)

    @property
    def received_messages(self) -> list[list[BaseMessage]]:
        return self._received

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self._received.append(messages)
        message = AIMessage(
            content=self.canned_content, response_metadata=dict(self.canned_metadata)
        )
        return ChatResult(generations=[ChatGeneration(message=message)])

    @property
    def _llm_type(self) -> str:
        return "sample-recording-chat-model"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _system_message(messages: list[BaseMessage]) -> SystemMessage:
    matches = [m for m in messages if isinstance(m, SystemMessage)]
    assert len(matches) == 1, "expected exactly one system message in the prompt"
    return matches[0]


def _human_message(messages: list[BaseMessage]) -> HumanMessage:
    matches = [m for m in messages if isinstance(m, HumanMessage)]
    assert len(matches) == 1, "expected exactly one human message in the prompt"
    return matches[0]


def _concatenated_text(messages: list[BaseMessage]) -> str:
    """Every message's content joined, for whole-prompt substring checks."""
    return "\n".join(str(message.content) for message in messages)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_chunks() -> list[ContentChunk]:
    return [
        _make_content_chunk(_SAMPLE_PASSAGE_ONE, "sample-page-one"),
        _make_content_chunk(_SAMPLE_PASSAGE_TWO, "sample-page-two"),
    ]


@pytest.fixture()
def retriever(sample_chunks: list[ContentChunk]) -> _RecordingRetriever:
    return _RecordingRetriever(sample_chunks)


@pytest.fixture()
def empty_retriever() -> _RecordingRetriever:
    return _RecordingRetriever([])


@pytest.fixture()
def chat_client() -> _RecordingChatModel:
    return _RecordingChatModel()


@pytest.fixture()
def below_floor_chunk() -> ContentChunk:
    return _make_content_chunk(
        _BELOW_FLOOR_PASSAGE, "sample-page-below-floor", similarity=_BELOW_FLOOR_SIMILARITY
    )


@pytest.fixture()
def above_floor_chunk() -> ContentChunk:
    return _make_content_chunk(
        _ABOVE_FLOOR_PASSAGE, "sample-page-above-floor", similarity=_ABOVE_FLOOR_SIMILARITY
    )


@pytest.fixture()
def boundary_chunk() -> ContentChunk:
    return _make_content_chunk(
        _BOUNDARY_PASSAGE, "sample-page-boundary", similarity=MINIMUM_SIMILARITY
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPromptGrounding:
    """The prompt sent to the model must contain every retrieved passage verbatim."""

    def test_prompt_contains_every_retrieved_passage(
        self,
        retriever: _RecordingRetriever,
        chat_client: _RecordingChatModel,
    ) -> None:
        explain(_SAMPLE_QUESTION, _SAMPLE_THEME, retriever=retriever, chat_client=chat_client)

        prompt_text = _concatenated_text(chat_client.received_messages[0])

        assert _SAMPLE_PASSAGE_ONE in prompt_text
        assert _SAMPLE_PASSAGE_TWO in prompt_text


class TestSystemMessageContract:
    """The fixed system message must carry both hard product contracts."""

    def test_system_message_contains_grounding_contract(
        self,
        retriever: _RecordingRetriever,
        chat_client: _RecordingChatModel,
    ) -> None:
        explain(_SAMPLE_QUESTION, _SAMPLE_THEME, retriever=retriever, chat_client=chat_client)

        system_text = str(_system_message(chat_client.received_messages[0]).content)

        assert _GROUNDING_CONTRACT in system_text

    def test_system_message_contains_english_output_contract(
        self,
        retriever: _RecordingRetriever,
        chat_client: _RecordingChatModel,
    ) -> None:
        explain(_SAMPLE_QUESTION, _SAMPLE_THEME, retriever=retriever, chat_client=chat_client)

        system_text = str(_system_message(chat_client.received_messages[0]).content)

        assert _ENGLISH_CONTRACT in system_text


class TestPromptInjectionBoundary:
    """The untrusted user_question must only ever reach the human message."""

    def test_user_question_is_in_human_message_and_not_in_system_message(
        self,
        retriever: _RecordingRetriever,
        chat_client: _RecordingChatModel,
    ) -> None:
        explain(_SAMPLE_QUESTION, _SAMPLE_THEME, retriever=retriever, chat_client=chat_client)

        messages = chat_client.received_messages[0]
        human_text = str(_human_message(messages).content)
        system_text = str(_system_message(messages).content)

        assert _SAMPLE_QUESTION in human_text
        assert _SAMPLE_QUESTION not in system_text


class TestExplanationResult:
    """explain() returns the model's text paired with the chunks that grounded it."""

    def test_returns_model_text_and_retrieved_chunks(
        self,
        sample_chunks: list[ContentChunk],
        retriever: _RecordingRetriever,
        chat_client: _RecordingChatModel,
    ) -> None:
        result = explain(
            _SAMPLE_QUESTION, _SAMPLE_THEME, retriever=retriever, chat_client=chat_client
        )

        assert isinstance(result, Explanation)
        assert result.text == _SAMPLE_RESPONSE_TEXT
        assert result.context_chunks == tuple(sample_chunks)

    def test_reads_only_the_text_block_of_a_block_structured_response(
        self,
        retriever: _RecordingRetriever,
    ) -> None:
        """A thinking block alongside the answer must not leak into the learner's text."""
        chat_client = _RecordingChatModel(canned_content=_BLOCK_STRUCTURED_CONTENT)

        result = explain(
            _SAMPLE_QUESTION, _SAMPLE_THEME, retriever=retriever, chat_client=chat_client
        )

        assert result.text == _SAMPLE_RESPONSE_TEXT


class TestTruncatedResponse:
    """A response cut off at the token cap must be reported, not returned silently."""

    def test_warns_when_the_provider_stopped_at_the_token_cap(
        self,
        retriever: _RecordingRetriever,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        chat_client = _RecordingChatModel(canned_metadata=_TRUNCATED_METADATA)

        with caplog.at_level(logging.WARNING):
            explain(
                _SAMPLE_QUESTION, _SAMPLE_THEME, retriever=retriever, chat_client=chat_client
            )

        assert [record.message for record in caplog.records if record.levelno >= logging.WARNING]

    def test_does_not_warn_when_the_response_finished_normally(
        self,
        retriever: _RecordingRetriever,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        chat_client = _RecordingChatModel(canned_metadata=_COMPLETE_METADATA)

        with caplog.at_level(logging.WARNING):
            explain(
                _SAMPLE_QUESTION, _SAMPLE_THEME, retriever=retriever, chat_client=chat_client
            )

        assert not [
            record.message for record in caplog.records if record.levelno >= logging.WARNING
        ]


class TestRetrieverInvocation:
    """explain() scopes retrieval to the raw question and the given theme."""

    def test_retriever_called_with_raw_question_and_theme(
        self,
        retriever: _RecordingRetriever,
        chat_client: _RecordingChatModel,
    ) -> None:
        explain(_SAMPLE_QUESTION, _SAMPLE_THEME, retriever=retriever, chat_client=chat_client)

        assert retriever.calls == [(_SAMPLE_QUESTION, _SAMPLE_THEME)]


class TestEmptyRetrieval:
    """With no grounding material, explain() must not invoke the model at all."""

    def test_empty_retrieval_returns_insufficient_material_without_calling_model(
        self,
        empty_retriever: _RecordingRetriever,
        chat_client: _RecordingChatModel,
    ) -> None:
        result = explain(
            _SAMPLE_QUESTION, _SAMPLE_THEME, retriever=empty_retriever, chat_client=chat_client
        )

        assert isinstance(result, Explanation)
        assert _INSUFFICIENT_MATERIAL_SUBSTRING in result.text
        assert result.context_chunks == ()
        assert chat_client.invocation_count == 0


class TestSimilarityFloor:
    """explain() drops chunks below MINIMUM_SIMILARITY before they reach the prompt or the result."""

    def test_below_floor_chunk_excluded_from_prompt(
        self,
        below_floor_chunk: ContentChunk,
        above_floor_chunk: ContentChunk,
        chat_client: _RecordingChatModel,
    ) -> None:
        retriever = _RecordingRetriever([below_floor_chunk, above_floor_chunk])

        explain(_SAMPLE_QUESTION, _SAMPLE_THEME, retriever=retriever, chat_client=chat_client)

        prompt_text = _concatenated_text(chat_client.received_messages[0])
        assert _ABOVE_FLOOR_PASSAGE in prompt_text
        assert _BELOW_FLOOR_PASSAGE not in prompt_text

    def test_below_floor_chunk_excluded_from_context_chunks(
        self,
        below_floor_chunk: ContentChunk,
        above_floor_chunk: ContentChunk,
        chat_client: _RecordingChatModel,
    ) -> None:
        retriever = _RecordingRetriever([below_floor_chunk, above_floor_chunk])

        result = explain(
            _SAMPLE_QUESTION, _SAMPLE_THEME, retriever=retriever, chat_client=chat_client
        )

        assert above_floor_chunk in result.context_chunks
        assert below_floor_chunk not in result.context_chunks

    def test_chunk_at_exact_floor_is_kept(
        self,
        boundary_chunk: ContentChunk,
        chat_client: _RecordingChatModel,
    ) -> None:
        """The floor is inclusive: similarity == MINIMUM_SIMILARITY passes."""
        retriever = _RecordingRetriever([boundary_chunk])

        result = explain(
            _SAMPLE_QUESTION, _SAMPLE_THEME, retriever=retriever, chat_client=chat_client
        )

        assert boundary_chunk in result.context_chunks
        prompt_text = _concatenated_text(chat_client.received_messages[0])
        assert _BOUNDARY_PASSAGE in prompt_text

    def test_all_below_floor_returns_insufficient_material_without_calling_model(
        self,
        below_floor_chunk: ContentChunk,
        chat_client: _RecordingChatModel,
    ) -> None:
        """When every retrieved chunk is below the floor, explain() short-circuits like empty retrieval."""
        retriever = _RecordingRetriever([below_floor_chunk])

        result = explain(
            _SAMPLE_QUESTION, _SAMPLE_THEME, retriever=retriever, chat_client=chat_client
        )

        assert isinstance(result, Explanation)
        assert _INSUFFICIENT_MATERIAL_SUBSTRING in result.text
        assert result.context_chunks == ()
        assert chat_client.invocation_count == 0
