"""Unit tests for civica.graph.nodes: the planner loop as a compiled LangGraph subgraph.

The planner model is a scripted fake whose recorded inputs are the external LLM boundary.
The subgraph runs with an InMemorySaver checkpointer and an InMemoryStore;
retrieval, content generation and the trace writer are fakes. No network, no database.
"""

import re
from collections.abc import Callable
from dataclasses import replace
from typing import Any

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.memory import InMemoryStore

from civica.assessment.critic import CritiqueFn
from civica.domain.themes import (
    EXAM_QUESTION_COUNTS,
    HISTOIRE_GEOGRAPHIE_ET_CULTURE,
    THEMES_BY_SLUG,
    VIVRE_DANS_LA_SOCIETE_FRANCAISE,
)
from civica.graph.nodes import PlannerContext, build_planner_graph
from civica.graph.planner import (
    LESSON_EXCERPT_LIMIT,
    MAX_PLANNER_STEPS,
    MAX_QUIZ_QUESTIONS,
    PROFILE_KEY,
    SUMMARY_KEY,
    HaltReason,
    ToolCall,
)
from civica.explanation.prompts import WordLimits
from civica.graph.prompts import PLANNER_PROMPTS
from civica.graph.trace import RetrievalGap, TraceStep
from civica.memory.records import LearnerProfile, MistakeEpisode, SessionSummary, TopicMastery
from civica.memory.writer import get, list_records, put
from tests.support.assessment import (
    SAMPLE_OPTIONS,
    FakeChatModel,
    FakeRetriever,
    explanation_for,
    reject_all_critic,
)
from tests.support.memory import (
    FULL_MASTERY,
    WEIGHTED_CHOICE,
    WEIGHTED_MASTERY,
    seed_mastery,
)
from tests.support.planner import (
    DEFAULT_THEME,
    GOAL,
    MISTAKE,
    SAMPLE_KEY,
    PLANNER_FAILURE_TEXT,
    SAMPLE_QUESTION_ID,
    ControlledPlannerModel,
    FakePlannerModel,
    PlannerFakes,
    final_message,
    parallel_tool_calls,
    planner_deps,
    tool_call,
)
from tests.support.users import USER_ID

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LEARNER_MESSAGE = "sample-learner-message-001"
SECOND_LEARNER_MESSAGE = "sample-learner-message-002"
THIRD_LEARNER_MESSAGE = "sample-learner-message-003"
THREAD_ID = "sample-thread-001"
FINAL_TEXT = "sample-final-reply"
SECOND_FINAL_TEXT = "sample-second-final-reply"
THIRD_FINAL_TEXT = "sample-third-final-reply"
LESSON_TEXT = "sample-lesson-text"
FOCUS = "sample-focus"
QUERY = "sample-query"
UNKNOWN_THEME_SLUG = "sample-unknown-theme"
UNKNOWN_KIND = "sample-unknown-kind"
UNKNOWN_TOOL = "sample-unknown-tool"
SAMPLE_ARG_VALUE = "sample-arg-value"
SUMMARY = "sample-session-summary"

SUGGESTED_THEME = VIVRE_DANS_LA_SOCIETE_FRANCAISE
MULTI_QUERY_THEME = HISTOIRE_GEOGRAPHIE_ET_CULTURE

QUIZ_SIZE = 2
COMPREHENSION_CHECK_SIZE = 1

# A lesson exactly at the excerpt limit, then text that must be cut from the previous exchange.
LESSON_AT_LIMIT = ("sample-lesson-head-" * LESSON_EXCERPT_LIMIT)[:LESSON_EXCERPT_LIMIT]
LESSON_TAIL = "sample-text-beyond-the-limit"

# Word limits the content model must receive, pinned as literals.
LESSON_WORD_RULE = "150 words"
BRIEF_WORD_RULE = "80 words"
INJECTED_WORD_LIMITS = WordLimits(lesson=42, brief=7)
INJECTED_LESSON_RULE = "42 words"
INJECTED_BRIEF_RULE = "7 words"
BRIEF = "brief"

# Final graph state keys: the subgraph's output contract.
REPLY = "reply"
HALT_REASON = "halt_reason"
TOOL_CALLS = "tool_calls"
LESSON = "lesson"
QUIZ_QUESTIONS = "quiz_questions"

# Tool names are the contract with the planner model.
SUGGEST = "suggest_review_theme"
READ_STATE = "read_learner_state"
RETRIEVE = "retrieve_corpus"
TEACH = "teach"
GENERATE_QUIZ = "generate_quiz"
RECORD = "record_outcome"
PLANNER_TOOLS = {SUGGEST, READ_STATE, RETRIEVE, TEACH, GENERATE_QUIZ, RECORD}

# Parameters a tool may receive from the graph, never from the model.
INJECTED_PARAMS = {"user_id", "runtime"}

# A field that would tell the learner-facing model which option is correct.
ANSWER_FIELD = re.compile(
    r"""["']?\b(correct_index|correct_answer|correct_option|answer_index|answer)\b["']?\s*[:=]""",
    re.IGNORECASE,
)

MISTAKE_VALUE: dict[str, object] = {
    "theme": DEFAULT_THEME.slug,
    "question_id": SAMPLE_QUESTION_ID,
    "sources": [],
}

# Well-formed values for kinds the planner must never write, so only the allowlist can refuse them.
FORBIDDEN_WRITES: dict[str, dict[str, object]] = {
    "learner_profile": {"goal": GOAL},
    "topic_mastery": {"theme": DEFAULT_THEME.slug, "mastery": FULL_MASTERY},
    UNKNOWN_KIND: {"sample-field": "sample-value"},
}

TOPIC_MASTERY_ARGS: dict[str, object] = {
    "kind": "topic_mastery",
    "key": DEFAULT_THEME.slug,
    "value": FORBIDDEN_WRITES["topic_mastery"],
}

# Final graph state, keyed by the names above.
State = dict[str, Any]  # type: ignore[explicit-any]
PlannerGraph = CompiledStateGraph[Any, PlannerContext, Any, Any]  # type: ignore[explicit-any]


# ---------------------------------------------------------------------------
# Helpers: the one place the subgraph is built and invoked
# ---------------------------------------------------------------------------


def _compile(
    fakes: PlannerFakes, planner_model: FakePlannerModel, critic: CritiqueFn | None = None
) -> PlannerGraph:
    """The planner subgraph, compiled once so several turns can share a thread."""
    return build_planner_graph(planner_deps(fakes, planner_model, critic)).compile(
        checkpointer=InMemorySaver(), store=fakes.store
    )


def _invoke(
    graph: PlannerGraph,
    *,
    learner_message: str = LEARNER_MESSAGE,
) -> State:
    """One learner turn on THREAD_ID for USER_ID; returns the final graph state."""
    state: State = graph.invoke(
        {"learner_message": learner_message},
        config={"configurable": {"thread_id": THREAD_ID}},
        context=PlannerContext(user_id=USER_ID),
    )
    return state


def _run(
    fakes: PlannerFakes,
    planner_model: FakePlannerModel,
    *,
    critic: CritiqueFn | None = None,
) -> State:
    return _invoke(_compile(fakes, planner_model, critic))


def _scripted(*turns: AIMessage) -> FakePlannerModel:
    return FakePlannerModel(script=tuple(turns))


def _repeat_forever() -> FakePlannerModel:
    return FakePlannerModel(repeat=tool_call(SUGGEST, "call-forever"))


def _favor_suggested_theme(store: InMemoryStore) -> None:
    """Master every theme except SUGGESTED_THEME."""
    seed_mastery(
        store,
        USER_ID,
        {theme: (0.0 if theme == SUGGESTED_THEME else FULL_MASTERY) for theme in EXAM_QUESTION_COUNTS},
    )


def _teach(call_id: str, theme_slug: str = DEFAULT_THEME.slug) -> AIMessage:
    return tool_call(TEACH, call_id, theme=theme_slug, focus=FOCUS)


def _quiz(call_id: str, n: int, theme_slug: str = DEFAULT_THEME.slug) -> AIMessage:
    return tool_call(GENERATE_QUIZ, call_id, theme=theme_slug, n=n)


def _empty_retrieval(fakes: PlannerFakes) -> PlannerFakes:
    return replace(fakes, retriever=FakeRetriever(section_count=0))


def _assert_fallback_reply(state: State) -> None:
    """The learner got a lesson as the reply, never an error."""
    assert state[LESSON] is not None
    assert state[REPLY] == state[LESSON].text


def _previous_exchange(messages: list[BaseMessage]) -> tuple[str, str]:
    """The (learner, coach) pair placed between the system prompt and the current turn."""
    layout = [type(m).__name__ for m in messages[:3]]
    assert layout == ["SystemMessage", "HumanMessage", "AIMessage"], f"no previous exchange in {layout}"
    _, learner, coach = messages[:3]
    assert isinstance(coach, AIMessage)
    assert not coach.tool_calls
    return str(learner.content), str(coach.content)


def _context_text(messages: list[BaseMessage]) -> str:
    """Every non-system message of one model call, joined."""
    return "\n".join(str(m.content) for m in messages if not isinstance(m, SystemMessage))


def _humans(messages: list[BaseMessage]) -> list[str]:
    return [str(m.content) for m in messages if isinstance(m, HumanMessage)]


# ---------------------------------------------------------------------------
# Tool set offered to the model
# ---------------------------------------------------------------------------


class TestToolSet:
    """The model is offered exactly the planner tools; none lets it pick whose memory it touches."""

    def test_offers_exactly_the_planner_tools(self, planner_fakes: PlannerFakes) -> None:
        model = _scripted(final_message(FINAL_TEXT))

        _run(planner_fakes, model)

        assert set(model.offered_tools()) == PLANNER_TOOLS

    def test_no_tool_takes_a_user_id_or_runtime(self, planner_fakes: PlannerFakes) -> None:
        model = _scripted(final_message(FINAL_TEXT))

        _run(planner_fakes, model)

        for name, parameters in model.offered_tools().items():
            assert not INJECTED_PARAMS & set(parameters.get("properties", {})), name

    @pytest.mark.parametrize("name", [RETRIEVE, TEACH, GENERATE_QUIZ])
    def test_theme_parameter_lists_the_official_slugs(
        self, planner_fakes: PlannerFakes, name: str
    ) -> None:
        model = _scripted(final_message(FINAL_TEXT))

        _run(planner_fakes, model)

        assert set(model.offered_tools()[name]["properties"]["theme"]["enum"]) == set(THEMES_BY_SLUG)


# ---------------------------------------------------------------------------
# Loop: tool trace and termination
# ---------------------------------------------------------------------------


class TestScriptedTurn:
    """suggest_review_theme, teach, final message; run in order and then the loop stops."""

    @pytest.fixture()
    def model(self) -> FakePlannerModel:
        return _scripted(
            tool_call(SUGGEST, "call-suggest"),
            _teach("call-teach"),
            final_message(FINAL_TEXT),
        )

    def test_tools_run_in_order_with_model_supplied_args(
        self, planner_fakes: PlannerFakes, model: FakePlannerModel
    ) -> None:
        state = _run(planner_fakes, model)

        assert state[TOOL_CALLS] == [
            ToolCall(name=SUGGEST, args={}),
            ToolCall(name=TEACH, args={"theme": DEFAULT_THEME.slug, "focus": FOCUS}),
        ]

    def test_final_message_ends_the_turn(
        self, planner_fakes: PlannerFakes, model: FakePlannerModel
    ) -> None:
        state = _run(planner_fakes, model)

        assert state[HALT_REASON] == HaltReason.FINAL_MESSAGE
        assert state[REPLY] == FINAL_TEXT
        assert model.invocation_count == len(model.script)

    def test_lesson_is_grounded_in_the_requested_theme(
        self, planner_fakes: PlannerFakes, model: FakePlannerModel
    ) -> None:
        state = _run(planner_fakes, model)

        lesson = state[LESSON]
        assert lesson is not None
        assert lesson.context_chunks
        assert {c.chunk.theme for c in lesson.context_chunks} == {DEFAULT_THEME}
        assert (FOCUS, DEFAULT_THEME) in planner_fakes.retriever.calls

    def test_model_sees_the_suggested_theme(
        self, planner_fakes: PlannerFakes, model: FakePlannerModel
    ) -> None:
        _favor_suggested_theme(planner_fakes.store)

        _run(planner_fakes, model)

        assert SUGGESTED_THEME.slug in model.tool_output("call-suggest")


class TestStepBound:
    """A model that never stops asking for tools is halted and the learner still gets a lesson."""

    def test_model_is_invoked_exactly_the_bound(self, planner_fakes: PlannerFakes) -> None:
        model = _repeat_forever()

        _run(planner_fakes, model)

        assert model.invocation_count == MAX_PLANNER_STEPS

    def test_halt_is_reported_without_raising(self, planner_fakes: PlannerFakes) -> None:
        state = _run(planner_fakes, _repeat_forever())

        assert state[HALT_REASON] == HaltReason.MAX_STEPS

    def test_reply_is_a_fallback_lesson_on_the_suggested_theme(
        self, planner_fakes: PlannerFakes
    ) -> None:
        _favor_suggested_theme(planner_fakes.store)

        state = _run(planner_fakes, _repeat_forever())

        _assert_fallback_reply(state)
        lesson = state[LESSON]
        assert lesson.context_chunks
        assert {c.chunk.theme for c in lesson.context_chunks} == {SUGGESTED_THEME}
        assert (SUGGESTED_THEME.display_name_fr, SUGGESTED_THEME) in planner_fakes.retriever.calls


class TestLessonLength:
    """teach asks for a full lesson by default and a short answer when the model sets brief."""

    def test_brief_is_an_optional_boolean_of_teach(self, planner_fakes: PlannerFakes) -> None:
        model = _scripted(final_message(FINAL_TEXT))

        _run(planner_fakes, model)

        parameters = model.offered_tools()[TEACH]
        assert parameters["properties"][BRIEF]["type"] == "boolean"
        assert BRIEF not in parameters.get("required", [])

    def test_teach_sends_the_lesson_word_limit(self, planner_fakes: PlannerFakes) -> None:
        _run(planner_fakes, _scripted(_teach("call-teach"), final_message(FINAL_TEXT)))

        assert LESSON_WORD_RULE in planner_fakes.content_model.system_text(0)

    def test_brief_teach_sends_only_the_brief_word_limit(self, planner_fakes: PlannerFakes) -> None:
        brief_teach = tool_call(TEACH, "call-teach", theme=DEFAULT_THEME.slug, focus=FOCUS, brief=True)

        _run(planner_fakes, _scripted(brief_teach, final_message(FINAL_TEXT)))

        system_text = planner_fakes.content_model.system_text(0)
        assert BRIEF_WORD_RULE in system_text
        assert LESSON_WORD_RULE not in system_text

    def test_fallback_lesson_keeps_the_lesson_word_limit(self, planner_fakes: PlannerFakes) -> None:
        _run(planner_fakes, _repeat_forever())

        assert LESSON_WORD_RULE in planner_fakes.content_model.system_text(0)

    def test_teach_sends_the_injected_word_limits(self, planner_fakes: PlannerFakes) -> None:
        brief_teach = tool_call(TEACH, "call-brief", theme=DEFAULT_THEME.slug, focus=FOCUS, brief=True)
        model = _scripted(_teach("call-teach"), brief_teach, final_message(FINAL_TEXT))
        deps = replace(planner_deps(planner_fakes, model), word_limits=INJECTED_WORD_LIMITS)
        graph = build_planner_graph(deps).compile(checkpointer=InMemorySaver(), store=planner_fakes.store)

        _invoke(graph)

        assert INJECTED_LESSON_RULE in planner_fakes.content_model.system_text(0)
        assert INJECTED_BRIEF_RULE in planner_fakes.content_model.system_text(1)


class TestTurnsOnOneThread:
    """Each turn on a thread starts fresh: a full step budget and only the bounded previous exchange."""

    @pytest.fixture()
    def model(self) -> FakePlannerModel:
        return _repeat_forever()

    @pytest.fixture()
    def states(self, planner_fakes: PlannerFakes, model: FakePlannerModel) -> list[State]:
        graph = _compile(planner_fakes, model)
        return [
            _invoke(graph, learner_message=LEARNER_MESSAGE),
            _invoke(graph, learner_message=SECOND_LEARNER_MESSAGE),
        ]

    def test_second_turn_gets_the_full_step_budget(
        self, model: FakePlannerModel, states: list[State]
    ) -> None:
        assert model.invocation_count == 2 * MAX_PLANNER_STEPS
        assert [s[HALT_REASON] for s in states] == [HaltReason.MAX_STEPS, HaltReason.MAX_STEPS]

    def test_second_turn_sees_only_the_previous_exchange_not_its_tool_calls(
        self, model: FakePlannerModel, states: list[State]
    ) -> None:
        for messages in model.received[MAX_PLANNER_STEPS:]:
            learner, _ = _previous_exchange(messages)
            assert learner == LEARNER_MESSAGE
            assert _humans(messages) == [LEARNER_MESSAGE, SECOND_LEARNER_MESSAGE]
            current = messages[3]
            assert isinstance(current, HumanMessage)
            assert current.content == SECOND_LEARNER_MESSAGE

    def test_each_turn_is_traced_on_its_own(
        self, planner_fakes: PlannerFakes, states: list[State]
    ) -> None:
        steps = planner_fakes.trace.steps
        turn_ids = list(dict.fromkeys(s.turn_id for s in steps))

        assert len(turn_ids) == 2
        for turn_id, state in zip(turn_ids, states, strict=True):
            turn = [s for s in steps if s.turn_id == turn_id]
            _assert_one_turn(turn, HaltReason.MAX_STEPS, tool_call_count=len(state[TOOL_CALLS]))


class TestReviewSuggestion:
    """suggest_review_theme and the halt fallback use the weighted review heuristic with its tie-break."""

    def test_tool_weights_mastery_by_exam_share(self, planner_fakes: PlannerFakes) -> None:
        seed_mastery(planner_fakes.store, USER_ID, WEIGHTED_MASTERY)
        model = _scripted(tool_call(SUGGEST, "call-suggest"), final_message(FINAL_TEXT))

        _run(planner_fakes, model)

        assert model.tool_output("call-suggest") == WEIGHTED_CHOICE.slug

    def test_tool_breaks_ties_with_a_mistake(self, planner_fakes: PlannerFakes) -> None:
        put(USER_ID, SAMPLE_KEY, MISTAKE, planner_fakes.store)
        model = _scripted(tool_call(SUGGEST, "call-suggest"), final_message(FINAL_TEXT))

        _run(planner_fakes, model)

        assert model.tool_output("call-suggest") == MISTAKE.theme.slug

    def test_fallback_lesson_uses_the_weighted_heuristic(self, planner_fakes: PlannerFakes) -> None:
        seed_mastery(planner_fakes.store, USER_ID, WEIGHTED_MASTERY)

        state = _run(planner_fakes, _repeat_forever())

        assert state[LESSON] is not None
        assert {c.chunk.theme for c in state[LESSON].context_chunks} == {WEIGHTED_CHOICE}


# ---------------------------------------------------------------------------
# Write guard: record_outcome behind PLANNER_WRITE_ALLOWLIST
# ---------------------------------------------------------------------------


class TestRecordOutcomeGuard:
    """Writes outside the planner allowlist never reach the store; the turn ends in a fallback lesson."""

    @pytest.mark.parametrize("kind", list(FORBIDDEN_WRITES))
    def test_forbidden_kind_halts_with_a_fallback_and_store_is_unchanged(
        self, planner_fakes: PlannerFakes, kind: str
    ) -> None:
        model = _scripted(
            tool_call(RECORD, "call-record", kind=kind, key=SAMPLE_KEY, value=FORBIDDEN_WRITES[kind]),
            final_message(FINAL_TEXT),
        )

        state = _run(planner_fakes, model)

        assert state[HALT_REASON] == HaltReason.FORBIDDEN_WRITE
        _assert_fallback_reply(state)
        assert model.invocation_count == 1
        assert planner_fakes.store.list_namespaces() == []

    def test_topic_mastery_is_refused_to_the_planner_but_allowed_in_process(
        self, planner_fakes: PlannerFakes
    ) -> None:
        model = _scripted(tool_call(RECORD, "call-record", **TOPIC_MASTERY_ARGS), final_message(FINAL_TEXT))
        mastery = TopicMastery(theme=DEFAULT_THEME, mastery=FULL_MASTERY)

        state = _run(planner_fakes, model)
        refused = get(USER_ID, DEFAULT_THEME.slug, TopicMastery, planner_fakes.store)
        put(USER_ID, DEFAULT_THEME.slug, mastery, planner_fakes.store)

        assert state[HALT_REASON] == HaltReason.FORBIDDEN_WRITE
        assert refused is None
        assert get(USER_ID, DEFAULT_THEME.slug, TopicMastery, planner_fakes.store) == mastery

    def test_mistake_episode_is_written_for_the_context_user(self, planner_fakes: PlannerFakes) -> None:
        model = _scripted(
            tool_call(RECORD, "call-record", kind="mistake_episode", key=SAMPLE_KEY, value=MISTAKE_VALUE),
            final_message(FINAL_TEXT),
        )

        state = _run(planner_fakes, model)

        assert state[HALT_REASON] == HaltReason.FINAL_MESSAGE
        assert list_records(USER_ID, MistakeEpisode, planner_fakes.store) == [MISTAKE]


# ---------------------------------------------------------------------------
# Grounding: empty retrieval short-circuits, never a fabricated lesson or quiz
# ---------------------------------------------------------------------------


class TestGroundingGuard:
    """With no retrieved passages, teach and generate_quiz short-circuit without the content model."""

    def test_teach_returns_insufficient_material_lesson(self, planner_fakes: PlannerFakes) -> None:
        fakes = _empty_retrieval(planner_fakes)
        model = _scripted(_teach("call-teach"), final_message(FINAL_TEXT))

        state = _run(fakes, model)

        assert state[LESSON] is not None
        assert state[LESSON].context_chunks == ()
        assert fakes.content_model.invocation_count == 0

    def test_generate_quiz_does_not_fail_the_turn(self, planner_fakes: PlannerFakes) -> None:
        fakes = _empty_retrieval(planner_fakes)
        model = _scripted(_quiz("call-quiz", QUIZ_SIZE), final_message(FINAL_TEXT))

        state = _run(fakes, model)

        assert state[HALT_REASON] == HaltReason.FINAL_MESSAGE
        assert state[REPLY] == FINAL_TEXT
        assert state[QUIZ_QUESTIONS] == []
        assert model.tool_output("call-quiz")
        assert fakes.content_model.invocation_count == 0
        assert fakes.question_store.saved_questions == []


class TestComprehensionCheck:
    """teach then generate_quiz(n=1): a critic that rejects everything ends the turn after the lesson."""

    def test_system_prompt_asks_for_a_check_after_teach(self, planner_fakes: PlannerFakes) -> None:
        """A prompt convention, not a code path: the model is told to quiz once on what it taught."""
        model = _scripted(final_message(FINAL_TEXT))

        _run(planner_fakes, model)

        check = re.compile(rf"\b{TEACH}\b.*\b{GENERATE_QUIZ}\(\w+, n={COMPREHENSION_CHECK_SIZE}\)")
        assert any(check.search(text) for text in model.system_texts())

    def test_model_is_not_asked_to_record_session_summaries(self, planner_fakes: PlannerFakes) -> None:
        """memory_writer owns the summary, so a planner write would only waste a step."""
        model = _scripted(final_message(FINAL_TEXT))

        _run(planner_fakes, model)

        record_tool = next(
            tool for tool in model.bound_tools() if getattr(tool, "name", None) == RECORD
        )
        for text in (*model.system_texts(), str(getattr(record_tool, "description"))):
            assert "summar" not in text.lower()

    def test_rejected_check_still_completes_with_the_lesson(self, planner_fakes: PlannerFakes) -> None:
        model = _scripted(
            _teach("call-teach"),
            _quiz("call-check", COMPREHENSION_CHECK_SIZE),
            final_message(FINAL_TEXT),
        )

        state = _run(planner_fakes, model, critic=reject_all_critic)

        assert state[HALT_REASON] == HaltReason.FINAL_MESSAGE
        assert state[REPLY] == FINAL_TEXT
        assert state[LESSON] is not None
        assert state[LESSON].context_chunks
        assert state[QUIZ_QUESTIONS] == []
        assert model.tool_output("call-check")
        assert planner_fakes.question_store.saved_questions == []


# ---------------------------------------------------------------------------
# generate_quiz success
# ---------------------------------------------------------------------------


class TestGenerateQuiz:
    """A generated quiz's questions are kept in state, but the model never sees which option is correct."""

    @pytest.fixture()
    def model(self) -> FakePlannerModel:
        return _scripted(_quiz("call-quiz", QUIZ_SIZE), final_message(FINAL_TEXT))

    def test_state_keeps_the_requested_questions(
        self, planner_fakes: PlannerFakes, model: FakePlannerModel
    ) -> None:
        state = _run(planner_fakes, model)

        questions = state[QUIZ_QUESTIONS]
        assert len(questions) == QUIZ_SIZE
        assert {q.theme for q in questions} == {DEFAULT_THEME}

    def test_tool_output_hides_the_correct_answer(
        self, planner_fakes: PlannerFakes, model: FakePlannerModel
    ) -> None:
        _run(planner_fakes, model)

        output = model.tool_output("call-quiz")
        assert output
        assert ANSWER_FIELD.search(output) is None, output

    def test_tool_output_hides_the_explanations(
        self, planner_fakes: PlannerFakes, model: FakePlannerModel
    ) -> None:
        """An explanation says why an option is right, so it would reveal the answer."""
        state = _run(planner_fakes, model)

        output = model.tool_output("call-quiz")
        for question in state[QUIZ_QUESTIONS]:
            assert question.text in output
            assert explanation_for(question.text) not in output


class TestQuizSizeBound:
    """A quiz size outside 1..MAX_QUIZ_QUESTIONS is refused before any generation cost."""

    @pytest.mark.parametrize("n", [0, MAX_QUIZ_QUESTIONS + 1])
    def test_out_of_range_size_is_reported_without_generating(
        self, planner_fakes: PlannerFakes, n: int
    ) -> None:
        model = _scripted(_quiz("call-quiz", n), final_message(FINAL_TEXT))

        state = _run(planner_fakes, model)

        assert state[HALT_REASON] == HaltReason.FINAL_MESSAGE
        assert state[QUIZ_QUESTIONS] == []
        assert model.tool_output("call-quiz")
        assert planner_fakes.content_model.invocation_count == 0


# ---------------------------------------------------------------------------
# Bad tool arguments are reported to the model, not raised
# ---------------------------------------------------------------------------


class TestBadToolArguments:
    """An unknown theme slug becomes an error message for the model; the turn carries on."""

    def test_unknown_theme_in_teach_is_reported_and_turn_completes(
        self, planner_fakes: PlannerFakes
    ) -> None:
        model = _scripted(_teach("call-teach", UNKNOWN_THEME_SLUG), final_message(FINAL_TEXT))

        state = _run(planner_fakes, model)

        assert state[HALT_REASON] == HaltReason.FINAL_MESSAGE
        assert state[REPLY] == FINAL_TEXT
        assert state[LESSON] is None
        assert DEFAULT_THEME.slug in model.tool_output("call-teach")
        assert planner_fakes.retriever.calls == []
        assert planner_fakes.content_model.invocation_count == 0


# ---------------------------------------------------------------------------
# Read tools
# ---------------------------------------------------------------------------


class TestReadTools:
    """retrieve_corpus and read_learner_state hand their results back to the model."""

    def test_retrieve_corpus_returns_passages_for_theme_and_query(
        self, planner_fakes: PlannerFakes
    ) -> None:
        model = _scripted(
            tool_call(RETRIEVE, "call-retrieve", theme=DEFAULT_THEME.slug, query=QUERY),
            final_message(FINAL_TEXT),
        )

        _run(planner_fakes, model)

        assert (QUERY, DEFAULT_THEME) in planner_fakes.retriever.calls
        output = model.tool_output("call-retrieve")
        for chunk in planner_fakes.retriever.chunks(DEFAULT_THEME):
            assert chunk.chunk.text in output

    def test_read_learner_state_shows_the_goal(self, planner_fakes: PlannerFakes) -> None:
        put(USER_ID, PROFILE_KEY, LearnerProfile(goal=GOAL), planner_fakes.store)
        model = _scripted(tool_call(READ_STATE, "call-state"), final_message(FINAL_TEXT))

        _run(planner_fakes, model)

        assert GOAL in model.tool_output("call-state")

    def test_read_learner_state_shows_the_last_session(self, planner_fakes: PlannerFakes) -> None:
        put(USER_ID, SUMMARY_KEY, SessionSummary(summary=SUMMARY), planner_fakes.store)
        model = _scripted(tool_call(READ_STATE, "call-state"), final_message(FINAL_TEXT))

        _run(planner_fakes, model)

        assert SUMMARY in model.tool_output("call-state")


# ---------------------------------------------------------------------------
# Prompt-injection boundary
# ---------------------------------------------------------------------------


class TestLearnerMessageBoundary:
    """The untrusted learner message goes in the human turn, never the system prompt."""

    @pytest.fixture()
    def model(self) -> FakePlannerModel:
        return _scripted(tool_call(SUGGEST, "call-suggest"), final_message(FINAL_TEXT))

    def test_absent_from_every_system_message(
        self, planner_fakes: PlannerFakes, model: FakePlannerModel
    ) -> None:
        _run(planner_fakes, model)

        assert model.system_texts()
        assert all(LEARNER_MESSAGE not in text for text in model.system_texts())

    def test_present_in_a_human_message(
        self, planner_fakes: PlannerFakes, model: FakePlannerModel
    ) -> None:
        _run(planner_fakes, model)

        assert any(LEARNER_MESSAGE in text for text in model.human_texts())

    def test_previous_message_stays_out_of_system_messages(self, planner_fakes: PlannerFakes) -> None:
        model = _scripted(final_message(FINAL_TEXT), final_message(SECOND_FINAL_TEXT))
        graph = _compile(planner_fakes, model)

        _invoke(graph)
        _invoke(graph, learner_message=SECOND_LEARNER_MESSAGE)

        assert LEARNER_MESSAGE in model.human_text(1)
        assert all(LEARNER_MESSAGE not in text for text in model.system_texts())


# ---------------------------------------------------------------------------
# Previous exchange: the bounded context a follow-up turn sees
# ---------------------------------------------------------------------------


def _fallback_by_forbidden_write() -> tuple[AIMessage, ...]:
    return (tool_call(RECORD, "call-record", **TOPIC_MASTERY_ARGS),)


def _fallback_by_step_bound() -> tuple[AIMessage, ...]:
    return tuple(tool_call(SUGGEST, f"call-suggest-{i}") for i in range(MAX_PLANNER_STEPS))


class TestPreviousExchange:
    """The next turn's model calls see the last completed exchange, bounded and without answers."""

    def test_next_turn_sees_the_previous_message_and_reply_on_every_call(
        self, planner_fakes: PlannerFakes
    ) -> None:
        model = _scripted(
            final_message(FINAL_TEXT),
            tool_call(SUGGEST, "call-suggest"),
            final_message(SECOND_FINAL_TEXT),
        )
        graph = _compile(planner_fakes, model)

        _invoke(graph)
        _invoke(graph, learner_message=SECOND_LEARNER_MESSAGE)

        for messages in model.received[1:]:
            learner, coach = _previous_exchange(messages)
            assert learner == LEARNER_MESSAGE
            assert FINAL_TEXT in coach
            assert _humans(messages) == [LEARNER_MESSAGE, SECOND_LEARNER_MESSAGE]

    def test_first_turn_on_a_thread_has_no_previous_exchange(self, planner_fakes: PlannerFakes) -> None:
        model = _scripted(final_message(FINAL_TEXT))

        _run(planner_fakes, model)

        (messages,) = model.received
        assert [type(m) for m in messages] == [SystemMessage, HumanMessage]

    def test_quiz_stems_are_numbered_as_displayed_without_options_or_answers(
        self, planner_fakes: PlannerFakes
    ) -> None:
        model = _scripted(
            _quiz("call-quiz", QUIZ_SIZE), final_message(FINAL_TEXT), final_message(SECOND_FINAL_TEXT)
        )
        graph = _compile(planner_fakes, model)

        questions = _invoke(graph)[QUIZ_QUESTIONS]
        _invoke(graph, learner_message=SECOND_LEARNER_MESSAGE)

        messages = model.received[-1]
        _, coach = _previous_exchange(messages)
        assert len(questions) == QUIZ_SIZE
        for number, question in enumerate(questions, start=1):
            assert re.search(rf"\b{number}[.):]\s*{re.escape(question.text)}", coach), coach
        context = _context_text(messages)
        assert all(option not in context for option in SAMPLE_OPTIONS)
        assert all(explanation_for(q.text) not in context for q in questions)
        assert ANSWER_FIELD.search(context) is None, context

    def test_lesson_at_the_limit_is_kept_whole(self, planner_fakes: PlannerFakes) -> None:
        fakes = replace(planner_fakes, content_model=FakeChatModel(canned_text=LESSON_AT_LIMIT))
        model = _scripted(
            _teach("call-teach"), final_message(FINAL_TEXT), final_message(SECOND_FINAL_TEXT)
        )
        graph = _compile(fakes, model)

        _invoke(graph)
        _invoke(graph, learner_message=SECOND_LEARNER_MESSAGE)

        _, coach = _previous_exchange(model.received[-1])
        assert LESSON_AT_LIMIT in coach

    def test_lesson_beyond_the_limit_is_cut(self, planner_fakes: PlannerFakes) -> None:
        long_lesson = LESSON_AT_LIMIT + LESSON_TAIL
        fakes = replace(planner_fakes, content_model=FakeChatModel(canned_text=long_lesson))
        model = _scripted(
            _teach("call-teach"), final_message(FINAL_TEXT), final_message(SECOND_FINAL_TEXT)
        )
        graph = _compile(fakes, model)

        _invoke(graph)
        _invoke(graph, learner_message=SECOND_LEARNER_MESSAGE)

        messages = model.received[-1]
        _, coach = _previous_exchange(messages)
        assert LESSON_AT_LIMIT in coach
        assert LESSON_TAIL not in _context_text(messages)

    def test_failed_turn_leaves_the_last_good_exchange(self, planner_fakes: PlannerFakes) -> None:
        model = ControlledPlannerModel(
            script=(
                final_message(FINAL_TEXT),
                final_message(SECOND_FINAL_TEXT),
                final_message(THIRD_FINAL_TEXT),
            ),
            fail_at_call=1,
        )
        graph = _compile(planner_fakes, model)

        _invoke(graph)
        with pytest.raises(RuntimeError, match=PLANNER_FAILURE_TEXT):
            _invoke(graph, learner_message=SECOND_LEARNER_MESSAGE)
        _invoke(graph, learner_message=THIRD_LEARNER_MESSAGE)

        messages = model.received[-1]
        learner, coach = _previous_exchange(messages)
        assert learner == LEARNER_MESSAGE
        assert FINAL_TEXT in coach
        assert _humans(messages) == [LEARNER_MESSAGE, THIRD_LEARNER_MESSAGE]

    @pytest.mark.parametrize(
        "fallback_turn",
        [_fallback_by_forbidden_write, _fallback_by_step_bound],
        ids=["forbidden-write", "step-bound"],
    )
    def test_fallback_turn_counts_and_its_lesson_appears_once(
        self, planner_fakes: PlannerFakes, fallback_turn: Callable[[], tuple[AIMessage, ...]]
    ) -> None:
        fakes = replace(planner_fakes, content_model=FakeChatModel(canned_text=LESSON_TEXT))
        model = _scripted(*fallback_turn(), final_message(SECOND_FINAL_TEXT))
        graph = _compile(fakes, model)

        _invoke(graph)
        _invoke(graph, learner_message=SECOND_LEARNER_MESSAGE)

        messages = model.received[-1]
        learner, _ = _previous_exchange(messages)
        assert learner == LEARNER_MESSAGE
        assert _context_text(messages).count(LESSON_TEXT) == 1


# ---------------------------------------------------------------------------
# Planner trace: one step row per tool call, then one terminal row
# ---------------------------------------------------------------------------


def _by_step(steps: list[TraceStep]) -> list[TraceStep]:
    """Rows may be written in any order; step_index gives the turn's order."""
    return sorted(steps, key=lambda s: s.step_index)


def _step_calls(steps: list[TraceStep]) -> list[tuple[str | None, dict[str, object] | None]]:
    """(tool_name, tool_args) of every non-terminal row, in step_index order."""
    return [
        (s.tool_name, dict(s.tool_args) if s.tool_args is not None else None)
        for s in _by_step(steps)
        if s.halt_reason is None
    ]


def _assert_one_turn(steps: list[TraceStep], halt_reason: HaltReason, tool_call_count: int) -> None:
    """Every row shares the turn; steps are numbered 0.. and end in exactly one terminal row."""
    ordered = _by_step(steps)
    assert ordered
    assert len({s.turn_id for s in ordered}) == 1
    assert all(s.user_id == USER_ID for s in ordered)
    assert all(s.thread_id == THREAD_ID for s in ordered)
    assert all(s.prompt_version == PLANNER_PROMPTS.version for s in ordered)
    assert [s.step_index for s in ordered] == list(range(tool_call_count + 1))
    terminal = ordered[-1]
    assert terminal.halt_reason == halt_reason
    assert terminal.tool_name is None
    assert terminal.tool_args is None
    assert all(s.halt_reason is None and s.tool_name is not None for s in ordered[:-1])


class TestTraceScriptedTurn:
    """suggest, teach, final: two step rows with the model's args, then a FINAL_MESSAGE row."""

    def test_rows_follow_the_tool_calls_in_order(self, planner_fakes: PlannerFakes) -> None:
        model = _scripted(
            tool_call(SUGGEST, "call-suggest"),
            _teach("call-teach"),
            final_message(FINAL_TEXT),
        )

        _run(planner_fakes, model)

        steps = planner_fakes.trace.steps
        _assert_one_turn(steps, HaltReason.FINAL_MESSAGE, tool_call_count=2)
        assert _step_calls(steps) == [
            (SUGGEST, {}),
            (TEACH, {"theme": DEFAULT_THEME.slug, "focus": FOCUS}),
        ]

    def test_calls_in_one_reply_are_numbered_in_listing_order(
        self, planner_fakes: PlannerFakes
    ) -> None:
        """Calls of one reply may run concurrently; step_index still follows the model's listing."""
        model = _scripted(
            parallel_tool_calls(tool_call(SUGGEST, "call-suggest"), tool_call(READ_STATE, "call-state")),
            _teach("call-teach"),
            final_message(FINAL_TEXT),
        )

        state = _run(planner_fakes, model)

        steps = planner_fakes.trace.steps
        _assert_one_turn(steps, HaltReason.FINAL_MESSAGE, tool_call_count=3)
        assert _step_calls(steps) == [
            (SUGGEST, {}),
            (READ_STATE, {}),
            (TEACH, {"theme": DEFAULT_THEME.slug, "focus": FOCUS}),
        ]
        assert state[TOOL_CALLS] == [
            ToolCall(name=SUGGEST, args={}),
            ToolCall(name=READ_STATE, args={}),
            ToolCall(name=TEACH, args={"theme": DEFAULT_THEME.slug, "focus": FOCUS}),
        ]

    def test_unknown_tool_is_recorded_as_requested(self, planner_fakes: PlannerFakes) -> None:
        model = _scripted(
            tool_call(UNKNOWN_TOOL, "call-unknown", sample_arg=SAMPLE_ARG_VALUE),
            final_message(FINAL_TEXT),
        )

        _run(planner_fakes, model)

        steps = planner_fakes.trace.steps
        _assert_one_turn(steps, HaltReason.FINAL_MESSAGE, tool_call_count=1)
        assert _step_calls(steps) == [(UNKNOWN_TOOL, {"sample_arg": SAMPLE_ARG_VALUE})]

    def test_zero_tool_turn_writes_only_the_terminal_row(self, planner_fakes: PlannerFakes) -> None:
        _run(planner_fakes, _scripted(final_message(FINAL_TEXT)))

        _assert_one_turn(planner_fakes.trace.steps, HaltReason.FINAL_MESSAGE, tool_call_count=0)

    def test_each_turn_on_a_thread_gets_its_own_turn_id(self, planner_fakes: PlannerFakes) -> None:
        graph = _compile(planner_fakes, _scripted(final_message(FINAL_TEXT), final_message(FINAL_TEXT)))

        _invoke(graph)
        _invoke(graph, learner_message=SECOND_LEARNER_MESSAGE)

        first, second = planner_fakes.trace.steps
        assert first.turn_id != second.turn_id


class TestTurnReset:
    """A turn's lesson, quiz and tool calls do not leak into the next turn on the same thread."""

    def test_second_turn_starts_without_the_first_turns_outputs(
        self, planner_fakes: PlannerFakes
    ) -> None:
        model = _scripted(
            _teach("call-teach"),
            _quiz("call-quiz", QUIZ_SIZE),
            final_message(FINAL_TEXT),
            final_message(FINAL_TEXT),
        )
        graph = _compile(planner_fakes, model)

        first = _invoke(graph)
        second = _invoke(graph, learner_message=SECOND_LEARNER_MESSAGE)

        assert first[LESSON] is not None
        assert first[QUIZ_QUESTIONS]
        assert second[LESSON] is None
        assert second[QUIZ_QUESTIONS] == []
        assert second[TOOL_CALLS] == []


class TestTraceStepBound:
    """A turn halted at MAX_PLANNER_STEPS ends in a MAX_STEPS row after the calls actually made."""

    def test_terminal_row_is_max_steps(self, planner_fakes: PlannerFakes) -> None:
        state = _run(planner_fakes, _repeat_forever())

        steps = planner_fakes.trace.steps
        _assert_one_turn(steps, HaltReason.MAX_STEPS, tool_call_count=len(state[TOOL_CALLS]))
        assert _step_calls(steps) == [(c.name, dict(c.args)) for c in state[TOOL_CALLS]]


class TestTraceForbiddenWrite:
    """A refused write is traced as a step, then the turn ends in one FORBIDDEN_WRITE row."""

    def test_forbidden_write_is_traced_then_halts(self, planner_fakes: PlannerFakes) -> None:
        model = _scripted(tool_call(RECORD, "call-record", **TOPIC_MASTERY_ARGS), final_message(FINAL_TEXT))

        _run(planner_fakes, model)

        steps = planner_fakes.trace.steps
        _assert_one_turn(steps, HaltReason.FORBIDDEN_WRITE, tool_call_count=1)
        assert _step_calls(steps) == [(RECORD, TOPIC_MASTERY_ARGS)]


# ---------------------------------------------------------------------------
# Retrieval gaps: every empty-retrieval short-circuit records the theme and query
# ---------------------------------------------------------------------------


class TestRetrievalGaps:
    """Empty retrieval records a gap with the theme and query that failed; anything else records none."""

    def test_empty_teach_records_the_focus(self, planner_fakes: PlannerFakes) -> None:
        fakes = _empty_retrieval(planner_fakes)

        _run(fakes, _scripted(_teach("call-teach"), final_message(FINAL_TEXT)))

        assert fakes.trace.gaps == [RetrievalGap(theme=DEFAULT_THEME, query=FOCUS)]

    def test_empty_fallback_lesson_records_the_theme_name(self, planner_fakes: PlannerFakes) -> None:
        fakes = _empty_retrieval(planner_fakes)
        _favor_suggested_theme(fakes.store)

        _run(fakes, _repeat_forever())

        assert fakes.trace.gaps == [
            RetrievalGap(theme=SUGGESTED_THEME, query=SUGGESTED_THEME.display_name_fr)
        ]

    def test_empty_retrieve_corpus_records_the_query(self, planner_fakes: PlannerFakes) -> None:
        fakes = _empty_retrieval(planner_fakes)
        model = _scripted(
            tool_call(RETRIEVE, "call-retrieve", theme=DEFAULT_THEME.slug, query=QUERY),
            final_message(FINAL_TEXT),
        )

        _run(fakes, model)

        assert fakes.trace.gaps == [RetrievalGap(theme=DEFAULT_THEME, query=QUERY)]

    def test_empty_quiz_records_the_theme_query(self, planner_fakes: PlannerFakes) -> None:
        fakes = _empty_retrieval(planner_fakes)

        _run(fakes, _scripted(_quiz("call-quiz", QUIZ_SIZE), final_message(FINAL_TEXT)))

        assert fakes.trace.gaps == [RetrievalGap(theme=DEFAULT_THEME, query=DEFAULT_THEME.display_name_fr)]

    def test_empty_histoire_quiz_records_each_sub_topic_query(self, planner_fakes: PlannerFakes) -> None:
        fakes = _empty_retrieval(planner_fakes)
        model = _scripted(
            _quiz("call-quiz", QUIZ_SIZE, theme_slug=MULTI_QUERY_THEME.slug), final_message(FINAL_TEXT)
        )

        _run(fakes, model)

        tried = [query for query, _ in fakes.retriever.calls]
        assert len(tried) > 1
        assert fakes.trace.gaps == [RetrievalGap(theme=MULTI_QUERY_THEME, query=query) for query in tried]

    def test_successful_teach_records_no_gap(self, planner_fakes: PlannerFakes) -> None:
        _run(planner_fakes, _scripted(_teach("call-teach"), final_message(FINAL_TEXT)))

        assert planner_fakes.trace.gaps == []

    def test_critic_rejection_records_no_gap(self, planner_fakes: PlannerFakes) -> None:
        model = _scripted(_quiz("call-quiz", QUIZ_SIZE), final_message(FINAL_TEXT))

        state = _run(planner_fakes, model, critic=reject_all_critic)

        assert state[QUIZ_QUESTIONS] == []
        assert planner_fakes.trace.gaps == []
