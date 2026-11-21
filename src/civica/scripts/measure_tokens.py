"""Per-call token measurement of a fixed scenario set against the real API.

Usage: uv run python -m civica.scripts.measure_tokens

Cost of running the script in 2026 was less than 1 euro.
Creates dedicated `token-measurement` user account in DB on first run.
Subsequent runs re-use that user account.
"""

import argparse
import csv
import logging
import secrets
import sys
import threading
import time
import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, fields, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Final
from uuid import UUID

import psycopg
import psycopg.rows
from dotenv import load_dotenv
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import LLMResult
from langchain_core.runnables import RunnableConfig

from civica.assessment.critic import LlmCritic
from civica.domain.themes import EXAM_QUESTION_COUNTS, THEMES_BY_SLUG, Theme
from civica.domain.user import UserId
from civica.graph.graph import GraphInput, InputKind, build_graph
from civica.graph.nodes import PlannerContext
from civica.graph.planner import default_planner_deps, read_learner_state
from civica.llm.client import (
    EXPLANATION_PROFILE,
    PLANNER_PROFILE,
    QUIZ_PROFILE,
    TRUNCATION_STOP_REASON,
    ModelCaller,
)
from civica.llm.usage import TokenCounts, TurnUsage, UsageTally, caller_of, final_message
from civica.ui.dev_panel import TOKEN_COUNTS, build_turn_panels, token_values
from civica.ui.turns import CoachGraph
from civica.users.service import BCRYPT_ROUNDS, UsernameTaken, find_user_id, register

logger = logging.getLogger(__name__)

# Keeps measurement turns apart from real learners' data.
MEASUREMENT_USERNAME: Final = "token-measurement"

_RAW_USAGE_KEY: Final = "usage"
_STOP_REASON_KEY: Final = "stop_reason"
_NO_SAVED_USAGE: Final = ("none",)
_DEFAULT_OUT_DIR: Final = Path("data/token_measurements")  # data/ is git-ignored

# The critic is built from the quiz profile, so quiz calls cover it too.
_CAPS: Final[Mapping[str, int]] = {
    ModelCaller.PLANNER.value: PLANNER_PROFILE.max_tokens,
    ModelCaller.EXPLANATION.value: EXPLANATION_PROFILE.max_tokens,
    ModelCaller.QUIZ.value: QUIZ_PROFILE.max_tokens,
}


@dataclass(frozen=True)
class CallRecord:
    """One model call."""

    scenario: str
    caller: str
    input_tokens: int
    output_tokens: int
    reasoning_tokens: int
    stop_reason: str | None
    latency_seconds: float


class CallRecorder(BaseCallbackHandler):
    """Records every model call of one scenario: caller, raw token counts, stop reason, latency."""

    def __init__(self, scenario: str, clock: Callable[[], float] = time.monotonic) -> None:
        self._scenario = scenario
        self._clock = clock
        self._lock = threading.Lock()
        self._started: dict[UUID, float] = {}
        self._records: list[CallRecord] = []

    def records(self) -> tuple[CallRecord, ...]:
        with self._lock:
            return tuple(self._records)

    def on_chat_model_start(  # type: ignore[explicit-any]
        self, serialized: dict[str, Any], messages: list[list[BaseMessage]], *, run_id: UUID, **kwargs: Any
    ) -> None:
        with self._lock:
            self._started[run_id] = self._clock()

    def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs: object) -> None:
        with self._lock:
            self._started.pop(run_id, None)

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: object) -> None:
        ended = self._clock()
        message = final_message(response)
        metadata = message.response_metadata if message is not None else {}
        # Raw counts, not usage_metadata, so the two can be cross-checked. Cached input counts as input.
        raw = metadata.get(_RAW_USAGE_KEY) or {}
        reasoning = _reasoning_of(message)
        tags = kwargs.get("tags")
        stop_reason = metadata.get(_STOP_REASON_KEY)
        with self._lock:
            started = self._started.pop(run_id, ended)
            self._records.append(
                CallRecord(
                    scenario=self._scenario,
                    caller=caller_of(tags if isinstance(tags, list) else None),
                    input_tokens=sum(
                        raw.get(key, 0)
                        for key in ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")
                    ),
                    output_tokens=raw.get("output_tokens", 0),
                    reasoning_tokens=reasoning,
                    stop_reason=str(stop_reason) if stop_reason is not None else None,
                    latency_seconds=ended - started,
                )
            )


def _reasoning_of(message: AIMessage | None) -> int:
    usage = message.usage_metadata if message is not None else None
    return usage.get("output_token_details", {}).get("reasoning", 0) if usage else 0


def _sum_by_caller(records: Iterable[CallRecord]) -> dict[str, TokenCounts]:
    sums: dict[str, TokenCounts] = {}
    for record in records:
        current = sums.get(record.caller, TokenCounts())
        sums[record.caller] = TokenCounts(
            current.input + record.input_tokens,
            current.output + record.output_tokens,
            current.reasoning + record.reasoning_tokens,
        )
    return sums


def check_turn(records: Sequence[CallRecord], saved: TurnUsage | None, panel_values: Sequence[str]) -> list[str]:
    """Accounting problems of one turn: records vs saved usage, saved total vs callers, panel vs saved."""
    if saved is None:
        return ["no token usage was saved for the turn"] if records else []
    problems: list[str] = []
    recorded = _sum_by_caller(records)
    for caller in sorted(recorded.keys() | saved.by_caller.keys()):
        if caller not in saved.by_caller:
            problems.append(f"{caller}: called but missing from the saved usage")
        elif caller not in recorded:
            problems.append(f"{caller}: saved but never called")
        elif recorded[caller] != saved.by_caller[caller]:
            problems.append(f"{caller}: recorded {recorded[caller]} but saved {saved.by_caller[caller]}")
    callers_sum = _sum_by_caller_counts(saved.by_caller.values())
    if saved.total != callers_sum:
        problems.append(f"saved total {saved.total} is not the sum of its callers {callers_sum}")
    if tuple(panel_values) != token_values(saved):
        problems.append(f"panel shows {tuple(panel_values)} but saved usage is {token_values(saved)}")
    return problems


def _sum_by_caller_counts(counts: Iterable[TokenCounts]) -> TokenCounts:
    items = list(counts)
    return TokenCounts(
        sum(c.input for c in items), sum(c.output for c in items), sum(c.reasoning for c in items)
    )


@dataclass(frozen=True)
class CallerSummary:
    """The calls of one caller across the whole run."""

    caller: str
    calls: int
    max_output: int
    cap: int | None
    cap_share: float | None
    reasoning_tokens: int
    truncations: int
    max_latency_seconds: float


def summarize(records: Iterable[CallRecord]) -> list[CallerSummary]:
    """One row per caller, sorted by name. Output is measured per call, against that caller's cap."""
    grouped: dict[str, list[CallRecord]] = {}
    for record in records:
        grouped.setdefault(record.caller, []).append(record)
    rows: list[CallerSummary] = []
    for caller in sorted(grouped):
        calls = grouped[caller]
        cap = _CAPS.get(caller)
        max_output = max(record.output_tokens for record in calls)
        rows.append(
            CallerSummary(
                caller=caller,
                calls=len(calls),
                max_output=max_output,
                cap=cap,
                cap_share=max_output / cap if cap else None,
                reasoning_tokens=sum(record.reasoning_tokens for record in calls),
                truncations=sum(record.stop_reason == TRUNCATION_STOP_REASON for record in calls),
                max_latency_seconds=max(record.latency_seconds for record in calls),
            )
        )
    return rows


def write_csv(records: Iterable[CallRecord], path: Path) -> None:
    """Write one row per call, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    names = [field.name for field in fields(CallRecord)]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        for record in records:
            writer.writerow({name: getattr(record, name) for name in names})


@dataclass(frozen=True)
class Scenario:
    """One learner message and the callers expected to make model calls for it."""

    name: str
    learner_message: str
    expected_callers: frozenset[str]


@dataclass(frozen=True)
class ScenarioResult:
    """What one scenario's turn did and what is wrong with its accounting."""

    scenario: Scenario
    thread_id: str
    records: tuple[CallRecord, ...]
    saved_usage: TurnUsage | None
    problems: tuple[str, ...]
    notes: tuple[str, ...]


def _panel_token_values(graph: CoachGraph, user_id: UserId, config: RunnableConfig) -> tuple[str, ...]:
    """The Token Counts values the developer panel builds for the saved turn."""
    assert graph.store is not None
    snapshot = graph.get_state(config)
    panels = build_turn_panels(snapshot, config, read_learner_state(user_id, graph.store))
    if panels is None:
        return ()
    for panel in (panels.lesson, panels.quiz, panels.reply):
        if panel is not None:
            for line in panel.lines:
                if line.label == TOKEN_COUNTS:
                    return line.values
    return ()


def _run_scenario(graph: CoachGraph, scenario: Scenario, user_id: UserId) -> ScenarioResult:
    # A fresh thread per scenario, apart from the learner's coach thread.
    thread_id = f"{user_id}:measure:{scenario.name}:{uuid.uuid4().hex[:8]}"
    recorder = CallRecorder(scenario.name)
    tally = UsageTally()
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}, "callbacks": [tally, recorder]}
    graph_input = GraphInput(kind=InputKind.MESSAGE, learner_message=scenario.learner_message)
    try:
        graph.invoke(graph_input, config, context=PlannerContext(user_id=user_id, usage=tally))
    except Exception as error:
        logger.exception("scenario_failed name=%s", scenario.name)
        records = recorder.records()
        return ScenarioResult(scenario, thread_id, records, None, (f"turn failed: {type(error).__name__}",), ())
    records = recorder.records()
    # Read back through the checkpointer, as a page reload does.
    saved = graph.get_state(config).values.get("token_usage")
    problems = check_turn(records, saved, _panel_token_values(graph, user_id, config) or _NO_SAVED_USAGE)
    actual = {record.caller for record in records}
    notes = tuple(
        f"{caller}: {'expected but did not call' if caller in scenario.expected_callers else 'called unexpectedly'}"
        for caller in sorted(actual ^ scenario.expected_callers)
    )
    return ScenarioResult(scenario, thread_id, records, saved, tuple(problems), notes)


def run_scenarios(graph: CoachGraph, scenarios: Sequence[Scenario], user_id: UserId) -> list[ScenarioResult]:
    """Run each scenario as one turn on its own thread, recording every model call."""
    results: list[ScenarioResult] = []
    for scenario in scenarios:
        logger.info("scenario_start name=%s", scenario.name)
        results.append(_run_scenario(graph, scenario, user_id))
    return results


_LESSON: Final = frozenset({ModelCaller.PLANNER.value, ModelCaller.EXPLANATION.value})
# The planner prompt adds a one-question check after a full lesson.
_LESSON_WITH_CHECK: Final = _LESSON | {ModelCaller.QUIZ.value}
_QUIZ: Final = frozenset({ModelCaller.PLANNER.value, ModelCaller.QUIZ.value})
_REPLY: Final = frozenset({ModelCaller.PLANNER.value})

_REPLY_MESSAGES: Final = (
    "Bonjour, je prépare l'examen civique.",
    "Merci beaucoup pour ton aide !",
    "Peux-tu m'expliquer comment tu m'aides à réviser ?",
    "Je suis un peu stressé avant l'examen.",
    "Quel est ton conseil pour organiser mes révisions ?",
)


def _theme_scenarios(theme: Theme) -> list[Scenario]:
    name = theme.display_name_fr
    return [
        Scenario(f"lesson-{theme.slug}", f"Explique-moi le thème « {name} » pour mon examen civique.", _LESSON_WITH_CHECK),
        Scenario(f"lesson-brief-{theme.slug}", f"En deux phrases seulement, résume le thème « {name} ».", _LESSON),
    ]


def default_scenarios() -> list[Scenario]:
    """About 25 turns of French learner messages, built from the theme registry."""
    scenarios = [s for theme in THEMES_BY_SLUG.values() for s in _theme_scenarios(theme)]
    biggest = max(EXAM_QUESTION_COUNTS, key=lambda theme: EXAM_QUESTION_COUNTS[theme])
    name = biggest.display_name_fr
    quiz_messages = (
        f"Fais-moi un quiz complet sur le thème « {name} ».",
        f"Je voudrais m'entraîner avec toutes les questions possibles sur « {name} ».",
        f"Donne-moi un quiz de douze questions sur « {name} ».",
        f"Un petit quiz de cinq questions sur « {name} », s'il te plaît.",
        f"Teste mes connaissances sur « {name} ».",
    )
    scenarios += [Scenario(f"quiz-{number}", text, _QUIZ) for number, text in enumerate(quiz_messages, start=1)]
    scenarios += [Scenario(f"reply-{number}", text, _REPLY) for number, text in enumerate(_REPLY_MESSAGES, start=1)]
    checks = (
        f"Explique-moi « {name} » puis pose-moi une question pour vérifier que j'ai compris.",
        "Explique-moi la laïcité et vérifie ensuite ma compréhension avec une question.",
        "Après ta leçon sur les droits et devoirs, fais-moi un contrôle de compréhension.",
    )
    scenarios += [Scenario(f"check-{number}", text, _LESSON_WITH_CHECK) for number, text in enumerate(checks, start=1)]
    hard = (
        "Explique-moi en détail les différences entre la Ve République, la IVe République et la IIIe, "
        "avec les dates clés, le rôle du Président, du Parlement et du Conseil constitutionnel.",
        "Compare la laïcité, la liberté de conscience et la neutralité des agents publics, avec des exemples "
        "précis tirés de la loi de 1905 et des textes récents.",
    )
    scenarios += [Scenario(f"hard-{number}", text, _LESSON) for number, text in enumerate(hard, start=1)]
    return scenarios


def _format_summary(rows: Sequence[CallerSummary]) -> str:
    header = f"{'caller':<12}{'calls':>6}{'max out':>9}{'cap':>7}{'share':>8}{'reasoning':>11}{'trunc':>7}{'max s':>8}"
    lines = [header]
    for row in rows:
        cap = "-" if row.cap is None else str(row.cap)
        share = "-" if row.cap_share is None else f"{row.cap_share:.0%}"
        lines.append(
            f"{row.caller:<12}{row.calls:>6}{row.max_output:>9}{cap:>7}{share:>8}"
            f"{row.reasoning_tokens:>11}{row.truncations:>7}{row.max_latency_seconds:>8.1f}"
        )
    return "\n".join(lines)


def _default_out() -> Path:
    return _DEFAULT_OUT_DIR / f"calls-{datetime.now():%Y%m%d-%H%M%S}.csv"


def measurement_user_id(
    conn: psycopg.Connection[psycopg.rows.TupleRow] | None = None, *, rounds: int = BCRYPT_ROUNDS
) -> UserId:
    """The measurement account's id, registering it on first use."""
    existing = find_user_id(MEASUREMENT_USERNAME, conn)
    if existing is not None:
        return existing
    try:
        # A random secret nobody keeps, so the account cannot be signed into.
        return register(MEASUREMENT_USERNAME, secrets.token_urlsafe(32), conn, rounds=rounds)
    except UsernameTaken:
        found = find_user_id(MEASUREMENT_USERNAME, conn)
        assert found is not None
        return found


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Measure per-call token use of the coach on the real API.")
    parser.add_argument("--out", type=Path, default=None, help="CSV path. Default: under data/.")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    load_dotenv()

    user_id = measurement_user_id()

    # The critic is on for this run only, so its cost is measured before it ships.
    graph = build_graph(replace(default_planner_deps(), critic=LlmCritic()))
    results = run_scenarios(graph, default_scenarios(), user_id)

    records = [record for result in results for record in result.records]
    out = args.out or _default_out()
    write_csv(records, out)

    print(_format_summary(summarize(records)))
    failed = 0
    for result in results:
        for note in result.notes:
            print(f"note [{result.scenario.name}]: {note}")
        for problem in result.problems:
            print(f"PROBLEM [{result.scenario.name}]: {problem}")
        failed += bool(result.problems)
    print(f"{len(results)} scenarios, {len(records)} calls, {failed} with problems. CSV: {out}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
