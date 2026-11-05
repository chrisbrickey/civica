"""Unit tests for civica.graph.graph.

The graph runs on an InMemorySaver and an InMemoryStore with every planner seam faked.
No calls to network or database.
"""

from dataclasses import replace
from typing import Any

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph

from civica.graph.graph import InputKind, build_graph
from civica.graph.nodes import PLANNER_MODEL, PlannerContext
from civica.graph.planner import SUMMARY_KEY, HaltReason, ToolCall
from civica.memory.records import SessionSummary, TopicMastery
from civica.memory.writer import get, list_records
from tests.support.assessment import FakeRetriever
from tests.support.memory import FULL_MASTERY
from tests.support.planner import (
    DEFAULT_THEME,
    FakePlannerModel,
    PlannerFakes,
    final_message,
    planner_deps,
    tool_call,
)
from tests.support.users import USER_ID

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LEARNER_MESSAGE = "sample-learner-message-001"
THREAD_ID = "sample-thread-001"
CONFIG = {"configurable": {"thread_id": THREAD_ID}}
FINAL_TEXT = "sample-final-reply"
FOCUS = "sample-focus"
UNKNOWN_KIND = "sample-unknown-kind"
UNKNOWN_THEME_SLUG = "sample-unknown-theme"
PLANNER_SUMMARY = "sample-planner-summary"
QUIZ_SIZE = 2

SUGGEST = "suggest_review_theme"
TEACH = "teach"
GENERATE_QUIZ = "generate_quiz"
RECORD = "record_outcome"

State = dict[str, Any]  # type: ignore[explicit-any]
Graph = CompiledStateGraph[Any, PlannerContext, Any, Any]  # type: ignore[explicit-any]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compile(fakes: PlannerFakes, model: FakePlannerModel) -> Graph:
    return build_graph(planner_deps(fakes, model), checkpointer=InMemorySaver())


def _input(kind: str = InputKind.MESSAGE) -> dict[str, str]:
    return {"kind": kind, "learner_message": LEARNER_MESSAGE}


def _invoke(graph: Graph) -> State:
    state: State = graph.invoke(_input(), config=CONFIG, context=PlannerContext(user_id=USER_ID))
    return state


def _scripted(*turns: AIMessage) -> FakePlannerModel:
    return FakePlannerModel(script=tuple(turns))


def _teach(call_id: str, theme_slug: str = DEFAULT_THEME.slug) -> AIMessage:
    return tool_call(TEACH, call_id, theme=theme_slug, focus=FOCUS)


def _summary(fakes: PlannerFakes) -> str:
    record = get(USER_ID, SUMMARY_KEY, SessionSummary, fakes.store)
    assert record is not None, "memory_writer wrote no session summary"
    return record.summary


# ---------------------------------------------------------------------------
# Routing and output contract
# ---------------------------------------------------------------------------


class TestRouting:
    """Every turn enters by kind; a message runs the planner."""

    def test_message_runs_the_planner_and_returns_its_outputs(
        self, planner_fakes: PlannerFakes
    ) -> None:
        model = _scripted(_teach("call-teach"), final_message(FINAL_TEXT))

        state = _invoke(_compile(planner_fakes, model))

        assert state["reply"] == FINAL_TEXT
        assert state["halt_reason"] == HaltReason.FINAL_MESSAGE
        assert state["tool_calls"] == [
            ToolCall(name=TEACH, args={"theme": DEFAULT_THEME.slug, "focus": FOCUS})
        ]
        assert state["lesson"] is not None
        assert state["quiz_questions"] == []

    def test_turn_result_is_readable_from_the_checkpoint_after_streaming(
        self, planner_fakes: PlannerFakes
    ) -> None:
        graph = _compile(planner_fakes, _scripted(final_message(FINAL_TEXT)))

        list(graph.stream(_input(), config=CONFIG, context=PlannerContext(user_id=USER_ID)))

        assert graph.get_state(CONFIG).values["reply"] == FINAL_TEXT

    def test_each_planner_step_streams_from_the_subgraph(self, planner_fakes: PlannerFakes) -> None:
        graph = _compile(planner_fakes, _scripted(final_message(FINAL_TEXT)))

        updates = graph.stream(
            _input(), config=CONFIG, context=PlannerContext(user_id=USER_ID), subgraphs=True
        )

        assert any(PLANNER_MODEL in update for _, update in updates)

    def test_unknown_kind_is_refused(self, planner_fakes: PlannerFakes) -> None:
        graph = _compile(planner_fakes, _scripted(final_message(FINAL_TEXT)))

        with pytest.raises(ValueError, match=UNKNOWN_KIND):
            graph.invoke(
                _input(UNKNOWN_KIND), config=CONFIG, context=PlannerContext(user_id=USER_ID)
            )


# ---------------------------------------------------------------------------
# memory_writer
# ---------------------------------------------------------------------------


class TestSessionSummary:
    """Every path ends at memory_writer, which recaps what the turn actually delivered."""

    def test_names_the_theme_of_a_grounded_lesson(self, planner_fakes: PlannerFakes) -> None:
        _invoke(_compile(planner_fakes, _scripted(_teach("call-teach"), final_message(FINAL_TEXT))))

        assert DEFAULT_THEME.slug in _summary(planner_fakes)

    def test_reports_a_lesson_without_official_material(self, planner_fakes: PlannerFakes) -> None:
        fakes = replace(planner_fakes, retriever=FakeRetriever(section_count=0))

        _invoke(_compile(fakes, _scripted(_teach("call-teach"), final_message(FINAL_TEXT))))

        assert "no official material" in _summary(fakes).lower()

    def test_names_the_quiz_theme_and_size(self, planner_fakes: PlannerFakes) -> None:
        quiz = tool_call(GENERATE_QUIZ, "call-quiz", theme=DEFAULT_THEME.slug, n=QUIZ_SIZE)

        _invoke(_compile(planner_fakes, _scripted(quiz, final_message(FINAL_TEXT))))

        summary = _summary(planner_fakes)
        assert DEFAULT_THEME.slug in summary
        assert str(QUIZ_SIZE) in summary

    def test_turn_without_a_lesson_or_quiz_still_writes_a_summary(
        self, planner_fakes: PlannerFakes
    ) -> None:
        _invoke(_compile(planner_fakes, _scripted(final_message(FINAL_TEXT))))

        assert _summary(planner_fakes)

    def test_failed_tool_call_is_not_reported_as_delivered(self, planner_fakes: PlannerFakes) -> None:
        model = _scripted(_teach("call-teach", UNKNOWN_THEME_SLUG), final_message(FINAL_TEXT))

        _invoke(_compile(planner_fakes, model))

        assert UNKNOWN_THEME_SLUG not in _summary(planner_fakes)

    def test_turn_halted_at_the_step_bound_reports_the_halt(
        self, planner_fakes: PlannerFakes
    ) -> None:
        model = FakePlannerModel(repeat=tool_call(SUGGEST, "call-forever"))

        _invoke(_compile(planner_fakes, model))

        assert HaltReason.MAX_STEPS.value in _summary(planner_fakes)

    def test_forbidden_write_writes_the_summary_but_not_the_mastery(
        self, planner_fakes: PlannerFakes
    ) -> None:
        forbidden = tool_call(
            RECORD,
            "call-forbidden",
            kind="topic_mastery",
            key=DEFAULT_THEME.slug,
            value={"theme": DEFAULT_THEME.slug, "mastery": FULL_MASTERY},
        )

        _invoke(_compile(planner_fakes, _scripted(forbidden)))

        assert HaltReason.FORBIDDEN_WRITE.value in _summary(planner_fakes)
        assert list_records(USER_ID, TopicMastery, planner_fakes.store) == []

    def test_later_turn_replaces_the_summary(self, planner_fakes: PlannerFakes) -> None:
        model = _scripted(
            _teach("call-teach"), final_message(FINAL_TEXT), final_message(FINAL_TEXT)
        )
        graph = _compile(planner_fakes, model)

        _invoke(graph)
        first = _summary(planner_fakes)
        _invoke(graph)

        assert _summary(planner_fakes) != first
        assert len(list_records(USER_ID, SessionSummary, planner_fakes.store)) == 1

    def test_replaces_a_summary_the_planner_wrote_this_turn(
        self, planner_fakes: PlannerFakes
    ) -> None:
        planner_write = tool_call(
            RECORD,
            "call-record",
            kind="session_summary",
            key=SUMMARY_KEY,
            value={"summary": PLANNER_SUMMARY},
        )

        _invoke(_compile(planner_fakes, _scripted(planner_write, final_message(FINAL_TEXT))))

        assert _summary(planner_fakes) != PLANNER_SUMMARY
