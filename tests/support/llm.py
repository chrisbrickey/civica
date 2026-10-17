"""Shared base for fake chat models: records every prompt, subclasses supply only the reply."""

from abc import abstractmethod
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import PrivateAttr


class FakeChatModelBase(BaseChatModel):  # type: ignore[explicit-any]
    """Records each message list it receives; subclasses implement `_reply_for(call_index)`."""

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
        return ChatResult(generations=[ChatGeneration(message=self._reply_for(call_index))])

    @property
    def _llm_type(self) -> str:
        return "sample-fake-chat-model"

    @staticmethod
    def _text_of(messages: list[BaseMessage], kind: type[BaseMessage]) -> str:
        return "\n".join(str(m.content) for m in messages if isinstance(m, kind))

    def _texts_across_calls(self, kind: type[BaseMessage]) -> list[str]:
        return [str(m.content) for msgs in self._received for m in msgs if isinstance(m, kind)]
