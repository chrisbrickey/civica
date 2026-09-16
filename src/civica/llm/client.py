"""Chat-model factory: a flow's model configuration paired with the client built from it.

Mirrors embeddings/embedder.py: one file per pathway picks the provider and default model.
Clients are built lazily, so importing this module never requires ANTHROPIC_API_KEY.
"""

from dataclasses import dataclass
from typing import Protocol

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage

GENERATION_MODEL = "claude-sonnet-5"

# Anthropic's signal that it stopped at the token cap rather than finishing the answer.
_ANTHROPIC_TRUNCATION_STOP_REASON = "max_tokens"


class ChatProfile(Protocol):
    """One flow's model configuration: how to build its client and how to read its responses.

    was_truncated() currently hard-codes "max_tokens", which is Anthropic's stop_reason.
    OpenAI signals the same condition with finish_reason == "length".
    """

    def build(self) -> BaseChatModel: ...

    def was_truncated(self, response: BaseMessage) -> bool: ...


@dataclass(frozen=True)
class AnthropicChat:
    """A Claude model paired with the output token cap it is called with."""

    model: str
    max_tokens: int

    def build(self) -> BaseChatModel:
        """Construct the client. Does not set temperature: current Claude models reject it."""
        return ChatAnthropic(model=self.model, max_tokens=self.max_tokens)  # type: ignore[call-arg]

    def was_truncated(self, response: BaseMessage) -> bool:
        """True when the provider cut the response off at the token cap instead of finishing."""
        stop_reason = response.response_metadata.get("stop_reason")
        return bool(stop_reason == _ANTHROPIC_TRUNCATION_STOP_REASON)


# Sized for a single grounded explanation, not for long multi-question generation.
EXPLANATION_PROFILE = AnthropicChat(model=GENERATION_MODEL, max_tokens=2048)

_clients: dict[ChatProfile, BaseChatModel] = {}


def get_chat_model(profile: ChatProfile = EXPLANATION_PROFILE) -> BaseChatModel:
    """Return the chat client for `profile`, constructing and caching it on first use."""
    if profile not in _clients:
        _clients[profile] = profile.build()
    return _clients[profile]
