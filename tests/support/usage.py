"""Expected token totals for tests that count the usage of fake chat models."""

from collections.abc import Mapping

from langchain_core.messages.ai import UsageMetadata

from civica.llm.usage import TokenCounts, TurnUsage


def counts_of(usage: UsageMetadata, calls: int = 1) -> TokenCounts:
    """What `calls` replies that each report `usage` add up to."""
    reasoning = usage.get("output_token_details", {}).get("reasoning", 0)
    return TokenCounts(
        input=usage["input_tokens"] * calls, output=usage["output_tokens"] * calls, reasoning=reasoning * calls
    )


def sum_counts(*counts: TokenCounts) -> TokenCounts:
    return TokenCounts(
        input=sum(c.input for c in counts),
        output=sum(c.output for c in counts),
        reasoning=sum(c.reasoning for c in counts),
    )


def turn_usage(by_caller: Mapping[str, TokenCounts]) -> TurnUsage:
    """The usage of a turn with these per-caller counts, totaled."""
    return TurnUsage(by_caller=dict(by_caller), total=sum_counts(*by_caller.values()))


def token_line(name: str, counts: TokenCounts) -> str:
    """One developer panel value: the tokens of a caller, or of the whole turn when `name` is "total"."""
    return f"{name}: in={counts.input} out={counts.output} reasoning={counts.reasoning}"
