"""Shared base for fake chat models: records every prompt, subclasses supply only the reply."""

from abc import abstractmethod
from collections.abc import Mapping
from typing import Any, Final

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.messages.ai import UsageMetadata
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import PrivateAttr

# Caller tags as the production chat profiles write them on their models.
PLANNER_TAG: Final = "planner"
EXPLANATION_TAG: Final = "explanation"
QUIZ_TAG: Final = "quiz"


def token_usage(input_tokens: int, output_tokens: int, reasoning: int | None = None) -> UsageMetadata:
    """Usage as a provider reports it; None leaves the reasoning detail out, as most replies do."""
    usage = UsageMetadata(
        input_tokens=input_tokens, output_tokens=output_tokens, total_tokens=input_tokens + output_tokens
    )
    if reasoning is not None:
        usage["output_token_details"] = {"reasoning": reasoning}
    return usage


def raw_usage(
    input_tokens: int, output_tokens: int, cache_read: int = 0, cache_creation: int = 0
) -> dict[str, int]:
    """Usage as Anthropic reports it in response_metadata["usage"], before LangChain maps it."""
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_read_input_tokens": cache_read,
        "cache_creation_input_tokens": cache_creation,
    }


def mapped_usage(raw: Mapping[str, int], reasoning: int | None = None) -> UsageMetadata:
    """The usage_metadata LangChain derives from `raw`: cached input counts as input."""
    input_tokens = raw["input_tokens"] + raw["cache_read_input_tokens"] + raw["cache_creation_input_tokens"]
    return token_usage(input_tokens, raw["output_tokens"], reasoning)


# Distinct numbers per model, so a total that mixes callers up or drops one is caught.
PLANNER_USAGE: Final = token_usage(input_tokens=100, output_tokens=20, reasoning=5)
CONTENT_USAGE: Final = token_usage(input_tokens=3000, output_tokens=400, reasoning=70)


class FakeChatModelBase(BaseChatModel):  # type: ignore[explicit-any]
    """Records each message list it receives; subclasses implement `_reply_for(call_index)`."""

    usage: UsageMetadata | None = None  # Reported on every reply when set; None replies carry no usage_metadata.
    raw_usage: dict[str, int] | None = None
    stop_reason: str | None = None

    _received: list[list[BaseMessage]] = PrivateAttr(default_factory=list)

    @property
    def invocation_count(self) -> int:
        return len(self._received)

    @property
    def received(self) -> list[list[BaseMessage]]:
        """The message list of every call, in call order."""
        return list(self._received)

    def system_text(self, call_index: int) -> str:
        """The system message content of one call."""
        return self._text_of(self._received[call_index], SystemMessage)

    def human_text(self, call_index: int) -> str:
        """The human message content of one call."""
        return self._text_of(self._received[call_index], HumanMessage)

    def prompt_text(self, call_index: int) -> str:
        """Every message of one call joined, so assertions do not depend on message layout."""
        return "\n".join(str(m.content) for m in self._received[call_index])

    def system_texts(self) -> list[str]:
        """Content of every SystemMessage across all invocations."""
        return self._texts_across_calls(SystemMessage)

    def human_texts(self) -> list[str]:
        """Content of every HumanMessage across all invocations."""
        return self._texts_across_calls(HumanMessage)

    @abstractmethod
    def _reply_for(self, call_index: int) -> AIMessage:
        """The reply to the call at `call_index` (0-based)."""

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        call_index = len(self._received)
        self._received.append(list(messages))
        reply = self._reply_for(call_index)
        if self.usage is not None:
            reply = reply.model_copy(update={"usage_metadata": self.usage})
        metadata: dict[str, object] = {}
        if self.raw_usage is not None:
            metadata["usage"] = dict(self.raw_usage)
        if self.stop_reason is not None:
            metadata["stop_reason"] = self.stop_reason
        if metadata:
            reply = reply.model_copy(update={"response_metadata": {**reply.response_metadata, **metadata}})
        return ChatResult(generations=[ChatGeneration(message=reply)])

    @property
    def _llm_type(self) -> str:
        return "sample-fake-chat-model"

    @staticmethod
    def _text_of(messages: list[BaseMessage], kind: type[BaseMessage]) -> str:
        return "\n".join(str(m.content) for m in messages if isinstance(m, kind))

    def _texts_across_calls(self, kind: type[BaseMessage]) -> list[str]:
        return [str(m.content) for msgs in self._received for m in msgs if isinstance(m, kind)]
