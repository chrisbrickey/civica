"""Token usage of one turn's model calls, split by caller."""

import threading
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from civica.llm.client import ModelCaller

# Caller name for a call whose model carries no ModelCaller tag.
UNTAGGED: Final = "untagged"

_CALLER_NAMES: Final = frozenset(caller.value for caller in ModelCaller)


@dataclass(frozen=True)
class TokenCounts:
    """Input, output, and reasoning tokens. Output already includes reasoning."""

    input: int = 0
    output: int = 0
    reasoning: int = 0

    @property
    def total(self) -> int:
        return self.input + self.output


def _counts(raw: TokenCounts | Mapping[str, int]) -> TokenCounts:
    return raw if isinstance(raw, TokenCounts) else TokenCounts(**raw)


@dataclass(frozen=True)
class TurnUsage:
    """Token counts of a turn per caller, plus their sum."""

    by_caller: Mapping[str, TokenCounts]
    total: TokenCounts

    def __post_init__(self) -> None:
        # The checkpointer reads these back as plain dicts, so restore the declared types.
        by_caller = {name: _counts(raw) for name, raw in self.by_caller.items()}
        object.__setattr__(self, "by_caller", by_caller)
        object.__setattr__(self, "total", _counts(self.total))


def _caller_of(tags: list[str] | None) -> str:
    return next((tag for tag in tags or [] if tag in _CALLER_NAMES), UNTAGGED)


class UsageTally(BaseCallbackHandler):
    """Counts the tokens of every model call it is handed, keyed by the call's caller tag."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._by_caller: dict[str, TokenCounts] = {}

    def on_llm_end(
        self, response: LLMResult, *, run_id: UUID, parent_run_id: UUID | None = None, **kwargs: object
    ) -> None:
        generation = response.generations[0][0] if response.generations and response.generations[0] else None
        message = generation.message if isinstance(generation, ChatGeneration) else None
        usage = message.usage_metadata if isinstance(message, AIMessage) else None
        if not usage:
            return
        tags = kwargs.get("tags")
        caller = _caller_of(tags if isinstance(tags, list) else None)
        added = TokenCounts(
            input=usage.get("input_tokens", 0),
            output=usage.get("output_tokens", 0),
            reasoning=usage.get("output_token_details", {}).get("reasoning", 0),
        )
        with self._lock:
            current = self._by_caller.get(caller, TokenCounts())
            self._by_caller[caller] = TokenCounts(
                current.input + added.input, current.output + added.output, current.reasoning + added.reasoning
            )

    def totals(self) -> TurnUsage:
        """A snapshot of the usage so far."""
        with self._lock:
            by_caller = dict(self._by_caller)
        total = TokenCounts(
            sum(c.input for c in by_caller.values()),
            sum(c.output for c in by_caller.values()),
            sum(c.reasoning for c in by_caller.values()),
        )
        return TurnUsage(by_caller=by_caller, total=total)
