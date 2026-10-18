"""Unit tests for civica.graph.planner: the bounded, tool-calling planner loop.

The planner model is a scripted fake (tests/support/planner.py); its recorded inputs are the
external LLM boundary. Memory is an in-memory store; retrieval and content generation are fakes.
No network, no database.
"""

import re
from dataclasses import replace

import pytest
from langchain_core.messages import AIMessage
from langgraph.store.memory import InMemoryStore

from civica.assessment.critic import CritiqueFn
from civica.domain.themes import (
    DROITS_ET_DEVOIRS,
    EXAM_QUESTION_COUNTS,
    HISTOIRE_GEOGRAPHIE_ET_CULTURE,
    SYSTEME_INSTITUTIONNEL_ET_POLITIQUE,
    VIVRE_DANS_LA_SOCIETE_FRANCAISE,
)
from civica.graph.planner import (
    MAX_PLANNER_STEPS,
    MAX_QUIZ_QUESTIONS,
    HaltReason,
    PlannerDeps,
    PlannerMode,
    PlannerResult,
    ToolCall,
    read_learner_state,
    run_planner,
)
from civica.graph.prompts import PLANNER_PROMPTS
from civica.graph.trace import RetrievalGap, TraceStep
from civica.memory.records import LearnerProfile, MistakeEpisode, TopicMastery
from civica.memory.writer import MemoryNotAllowed, get, list_records, put
from tests.support.assessment import FakeRetriever, reject_all_critic
from tests.support.memory import (
    FULL_MASTERY,
    HALF_MASTERY,
    WEIGHTED_CHOICE,
    WEIGHTED_MASTERY,
    seed_mastery,
)
from tests.support.planner import (
    PlannerFakes,
    FakePlannerModel,
    final_message,
    tool_call,
)
from tests.support.users import USER_ID

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LEARNER_MESSAGE = "sample-learner-message-001"
THREAD_ID = "sample-thread-001"
FINAL_TEXT = "sample-final-reply"
FOCUS = "sample-focus"
QUERY = "sample-query"
GOAL = "sample-goal"
UNKNOWN_THEME_SLUG = "sample-unknown-theme"
UNKNOWN_KIND = "sample-unknown-kind"
UNKNOWN_TOOL = "sample-unknown-tool"
SAMPLE_ARG_VALUE = "sample-arg-value"
SAMPLE_KEY = "sample-key-001"
SAMPLE_QUESTION_ID = "sample-question-id-001"

# One LearnerProfile per user under a constant key.
PROFILE_KEY = "current"

DEFAULT_THEME = DROITS_ET_DEVOIRS
SUGGESTED_THEME = VIVRE_DANS_LA_SOCIETE_FRANCAISE
MULTI_QUERY_THEME = HISTOIRE_GEOGRAPHIE_ET_CULTURE

QUIZ_SIZE = 2
COMPREHENSION_CHECK_SIZE = 1

# The longest scripted turn below: teach, generate_quiz, then a final message.
LONGEST_SCRIPTED_TURN = 3

# Tool names are the contract with the planner model.
SUGGEST = "suggest_review_theme"
READ_STATE = "read_learner_state"
RETRIEVE = "retrieve_corpus"
TEACH = "teach"
GENERATE_QUIZ = "generate_quiz"
RECORD = "record_outcome"
PLANNER_TOOLS = {SUGGEST, READ_STATE, RETRIEVE, TEACH, GENERATE_QUIZ, RECORD}

# A field that would tell the learner-facing model which option is correct.
ANSWER_FIELD = re.compile(
    r"""["']?\b(correct_index|correct_answer|correct_option|answer_index|answer)\b["']?\s*[:=]""",
    re.IGNORECASE,
)

MISTAKE = MistakeEpisode(theme=DEFAULT_THEME, question_id=SAMPLE_QUESTION_ID)
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _deps(
    fakes: PlannerFakes, planner_model: FakePlannerModel, critic: CritiqueFn | None = None
) -> PlannerDeps:
    return PlannerDeps(
        planner_model=planner_model,
        store=fakes.store,
        content_model=fakes.content_model,
        retriever=fakes.retriever,
        question_store=fakes.question_store,
        answer_logger=fakes.answer_logger,
        critic=critic,
        rejection_store=fakes.rejection_store,
        trace=fakes.trace,
    )


def _run(
    fakes: PlannerFakes,
    planner_model: FakePlannerModel,
    *,
    critic: CritiqueFn | None = None,
    mode: PlannerMode = PlannerMode.TEACH,
) -> PlannerResult:
    return run_planner(
        USER_ID, LEARNER_MESSAGE, mode, _deps(fakes, planner_model, critic), thread_id=THREAD_ID
    )


def _scripted(*turns: AIMessage) -> FakePlannerModel:
    return FakePlannerModel(script=tuple(turns))


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


# ---------------------------------------------------------------------------
# Tool set offered to the model
# ---------------------------------------------------------------------------


class TestToolSet:
    """The model is offered exactly the planner tools; none lets it pick whose memory it touches."""

    def test_offers_exactly_the_planner_tools(self, planner_fakes: PlannerFakes) -> None:
        model = _scripted(final_message(FINAL_TEXT))

        _run(planner_fakes, model)

        assert set(model.offered_tools()) == PLANNER_TOOLS

    def test_no_tool_takes_a_user_id(self, planner_fakes: PlannerFakes) -> None:
        model = _scripted(final_message(FINAL_TEXT))

        _run(planner_fakes, model)

        for name, parameters in model.offered_tools().items():
            assert "user_id" not in parameters.get("properties", {}), name

    def test_step_bound_fits_a_teach_and_check_turn(self) -> None:
        assert MAX_PLANNER_STEPS >= LONGEST_SCRIPTED_TURN


# ---------------------------------------------------------------------------
# Loop: tool trace and termination
# ---------------------------------------------------------------------------


class TestScriptedTurn:
    """suggest_review_theme then teach then a final message: both run in order, then the loop stops."""

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
        result = _run(planner_fakes, model)

        assert result.tool_calls == (
            ToolCall(name=SUGGEST, args={}),
            ToolCall(name=TEACH, args={"theme": DEFAULT_THEME.slug, "focus": FOCUS}),
        )

    def test_final_message_ends_the_turn(
        self, planner_fakes: PlannerFakes, model: FakePlannerModel
    ) -> None:
        result = _run(planner_fakes, model)

        assert result.halt_reason is HaltReason.FINAL_MESSAGE
        assert result.reply == FINAL_TEXT
        assert model.invocation_count == len(model.script)

    def test_lesson_is_grounded_in_the_requested_theme(
        self, planner_fakes: PlannerFakes, model: FakePlannerModel
    ) -> None:
        result = _run(planner_fakes, model)

        assert result.lesson is not None
        assert result.lesson.context_chunks
        assert {c.chunk.theme for c in result.lesson.context_chunks} == {DEFAULT_THEME}
        assert (FOCUS, DEFAULT_THEME) in planner_fakes.retriever.calls

    def test_model_sees_the_suggested_theme(
        self, planner_fakes: PlannerFakes, model: FakePlannerModel
    ) -> None:
        _favor_suggested_theme(planner_fakes.store)

        _run(planner_fakes, model)

        assert SUGGESTED_THEME.slug in model.tool_output("call-suggest")


class TestStepBound:
    """A model that never stops asking for tools is halted and the learner still gets a lesson."""

    @pytest.fixture()
    def model(self) -> FakePlannerModel:
        return FakePlannerModel(repeat=tool_call(SUGGEST, "call-forever"))

    def test_model_is_invoked_exactly_the_bound(
        self, planner_fakes: PlannerFakes, model: FakePlannerModel
    ) -> None:
        _run(planner_fakes, model)

        assert model.invocation_count == MAX_PLANNER_STEPS

    def test_halt_is_reported_without_raising(
        self, planner_fakes: PlannerFakes, model: FakePlannerModel
    ) -> None:
        result = _run(planner_fakes, model)

        assert result.halt_reason is HaltReason.MAX_STEPS

    def test_reply_is_a_fallback_lesson_on_the_suggested_theme(
        self, planner_fakes: PlannerFakes, model: FakePlannerModel
    ) -> None:
        _favor_suggested_theme(planner_fakes.store)

        result = _run(planner_fakes, model)

        assert result.lesson is not None
        assert result.reply == result.lesson.text
        assert result.lesson.context_chunks
        assert {c.chunk.theme for c in result.lesson.context_chunks} == {SUGGESTED_THEME}
        assert (SUGGESTED_THEME.display_name_fr, SUGGESTED_THEME) in planner_fakes.retriever.calls


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
        model = FakePlannerModel(repeat=tool_call(SUGGEST, "call-forever"))

        result = _run(planner_fakes, model)

        assert result.lesson is not None
        assert {c.chunk.theme for c in result.lesson.context_chunks} == {WEIGHTED_CHOICE}


# ---------------------------------------------------------------------------
# Write guard: record_outcome behind PLANNER_WRITE_ALLOWLIST
# ---------------------------------------------------------------------------


class TestRecordOutcomeGuard:
    """Writes outside the planner allowlist surface MemoryNotAllowed and never reach the store."""

    @pytest.mark.parametrize("kind", list(FORBIDDEN_WRITES))
    def test_forbidden_kind_raises_and_store_is_unchanged(
        self, planner_fakes: PlannerFakes, kind: str
    ) -> None:
        model = _scripted(
            tool_call(RECORD, "call-record", kind=kind, key=SAMPLE_KEY, value=FORBIDDEN_WRITES[kind]),
            final_message(FINAL_TEXT),
        )

        with pytest.raises(MemoryNotAllowed):
            _run(planner_fakes, model)

        assert planner_fakes.store.list_namespaces() == []

    def test_topic_mastery_is_refused_to_the_planner_but_allowed_in_process(
        self, planner_fakes: PlannerFakes
    ) -> None:
        model = _scripted(
            tool_call(
                RECORD, "call-record", kind="topic_mastery", key=DEFAULT_THEME.slug,
                value=FORBIDDEN_WRITES["topic_mastery"],
            ),
            final_message(FINAL_TEXT),
        )
        mastery = TopicMastery(theme=DEFAULT_THEME, mastery=FULL_MASTERY)

        with pytest.raises(MemoryNotAllowed):
            _run(planner_fakes, model)
        put(USER_ID, DEFAULT_THEME.slug, mastery, planner_fakes.store)

        assert get(USER_ID, DEFAULT_THEME.slug, TopicMastery, planner_fakes.store) == mastery

    def test_mistake_episode_is_written_for_the_bound_user(self, planner_fakes: PlannerFakes) -> None:
        model = _scripted(
            tool_call(RECORD, "call-record", kind="mistake_episode", key=SAMPLE_KEY, value=MISTAKE_VALUE),
            final_message(FINAL_TEXT),
        )

        result = _run(planner_fakes, model)

        assert result.halt_reason is HaltReason.FINAL_MESSAGE
        assert list_records(USER_ID, MistakeEpisode, planner_fakes.store) == [MISTAKE]


# ---------------------------------------------------------------------------
# Grounding: empty retrieval short-circuits, never a fabricated lesson or quiz
# ---------------------------------------------------------------------------


class TestGroundingGuard:
    """With no retrieved passages, teach and generate_quiz short-circuit without the content model."""

    def test_teach_returns_insufficient_material_lesson(self, planner_fakes: PlannerFakes) -> None:
        fakes = _empty_retrieval(planner_fakes)
        model = _scripted(_teach("call-teach"), final_message(FINAL_TEXT))

        result = _run(fakes, model)

        assert result.lesson is not None
        assert result.lesson.context_chunks == ()
        assert fakes.content_model.invocation_count == 0

    def test_generate_quiz_does_not_fail_the_turn(self, planner_fakes: PlannerFakes) -> None:
        fakes = _empty_retrieval(planner_fakes)
        model = _scripted(_quiz("call-quiz", QUIZ_SIZE), final_message(FINAL_TEXT))

        result = _run(fakes, model)

        assert result.halt_reason is HaltReason.FINAL_MESSAGE
        assert result.reply == FINAL_TEXT
        assert result.quiz is None
        assert model.tool_output("call-quiz")
        assert fakes.content_model.invocation_count == 0
        assert fakes.question_store.saved_questions == []


class TestComprehensionCheck:
    """teach then generate_quiz(n=1): a critic that rejects everything ends the turn after the lesson."""

    @pytest.mark.parametrize("mode", list(PlannerMode))
    def test_system_prompt_asks_for_a_check_after_teach(
        self, planner_fakes: PlannerFakes, mode: PlannerMode
    ) -> None:
        """A prompt convention, not a code path: the model is told to quiz once on what it taught."""
        model = _scripted(final_message(FINAL_TEXT))

        _run(planner_fakes, model, mode=mode)

        check = re.compile(rf"\b{TEACH}\b.*\b{GENERATE_QUIZ}\(\w+, n={COMPREHENSION_CHECK_SIZE}\)")
        assert any(check.search(text) for text in model.system_texts())

    def test_rejected_check_still_completes_with_the_lesson(self, planner_fakes: PlannerFakes) -> None:
        model = _scripted(
            _teach("call-teach"),
            _quiz("call-check", COMPREHENSION_CHECK_SIZE),
            final_message(FINAL_TEXT),
        )

        result = _run(planner_fakes, model, critic=reject_all_critic)

        assert result.halt_reason is HaltReason.FINAL_MESSAGE
        assert result.reply == FINAL_TEXT
        assert result.lesson is not None
        assert result.lesson.context_chunks
        assert result.quiz is None
        assert model.tool_output("call-check")
        assert planner_fakes.question_store.saved_questions == []


# ---------------------------------------------------------------------------
# generate_quiz success
# ---------------------------------------------------------------------------


class TestGenerateQuiz:
    """A generated quiz is returned to the caller, but the model never sees which option is correct."""

    @pytest.fixture()
    def model(self) -> FakePlannerModel:
        return _scripted(_quiz("call-quiz", QUIZ_SIZE), final_message(FINAL_TEXT))

    def test_result_carries_the_requested_quiz(
        self, planner_fakes: PlannerFakes, model: FakePlannerModel
    ) -> None:
        result = _run(planner_fakes, model, mode=PlannerMode.QUIZ)

        assert result.quiz is not None
        assert len(result.quiz.questions) == QUIZ_SIZE
        assert {q.theme for q in result.quiz.questions} == {DEFAULT_THEME}
        assert result.quiz.user_id == USER_ID

    def test_tool_output_hides_the_correct_answer(
        self, planner_fakes: PlannerFakes, model: FakePlannerModel
    ) -> None:
        _run(planner_fakes, model, mode=PlannerMode.QUIZ)

        output = model.tool_output("call-quiz")
        assert output
        assert ANSWER_FIELD.search(output) is None, output


class TestQuizSizeBound:
    """A quiz size outside 1..MAX_QUIZ_QUESTIONS is refused before any generation cost."""

    @pytest.mark.parametrize("n", [0, MAX_QUIZ_QUESTIONS + 1])
    def test_out_of_range_size_is_reported_without_generating(
        self, planner_fakes: PlannerFakes, n: int
    ) -> None:
        model = _scripted(_quiz("call-quiz", n), final_message(FINAL_TEXT))

        result = _run(planner_fakes, model, mode=PlannerMode.QUIZ)

        assert result.halt_reason is HaltReason.FINAL_MESSAGE
        assert result.quiz is None
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

        result = _run(planner_fakes, model)

        assert result.halt_reason is HaltReason.FINAL_MESSAGE
        assert result.reply == FINAL_TEXT
        assert result.lesson is None
        assert model.tool_output("call-teach")
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


# ---------------------------------------------------------------------------
# Prompt-injection boundary
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", list(PlannerMode))
class TestLearnerMessageBoundary:
    """The untrusted learner message goes in the human turn, never the system prompt."""

    @pytest.fixture()
    def model(self) -> FakePlannerModel:
        return _scripted(tool_call(SUGGEST, "call-suggest"), final_message(FINAL_TEXT))

    def test_absent_from_every_system_message(
        self, planner_fakes: PlannerFakes, model: FakePlannerModel, mode: PlannerMode
    ) -> None:
        _run(planner_fakes, model, mode=mode)

        assert model.system_texts()
        assert all(LEARNER_MESSAGE not in text for text in model.system_texts())

    def test_present_in_a_human_message(
        self, planner_fakes: PlannerFakes, model: FakePlannerModel, mode: PlannerMode
    ) -> None:
        _run(planner_fakes, model, mode=mode)

        assert any(LEARNER_MESSAGE in text for text in model.human_texts())


# ---------------------------------------------------------------------------
# read_learner_state (public function)
# ---------------------------------------------------------------------------


class TestReadLearnerState:
    """Mastery per recorded theme, mistake episodes, and the stated goal, for one user."""

    def test_returns_recorded_mastery_mistakes_and_goal(self, planner_fakes: PlannerFakes) -> None:
        store = planner_fakes.store
        recorded = {DEFAULT_THEME: HALF_MASTERY, SYSTEME_INSTITUTIONNEL_ET_POLITIQUE: FULL_MASTERY}
        seed_mastery(store, USER_ID, recorded)
        put(USER_ID, SAMPLE_KEY, MISTAKE, store)
        put(USER_ID, PROFILE_KEY, LearnerProfile(goal=GOAL), store)

        state = read_learner_state(USER_ID, store)

        assert state.mastery == recorded
        assert state.mistakes == (MISTAKE,)
        assert state.goal == GOAL

    def test_empty_store_gives_empty_state(self, planner_fakes: PlannerFakes) -> None:
        state = read_learner_state(USER_ID, planner_fakes.store)

        assert state.mastery == {}
        assert state.mistakes == ()
        assert state.goal is None


# ---------------------------------------------------------------------------
# Planner trace: one step row per tool call, then one terminal row
# ---------------------------------------------------------------------------


def _step_calls(steps: list[TraceStep]) -> list[tuple[str | None, dict[str, object] | None]]:
    """(tool_name, tool_args) of every non-terminal row, in write order."""
    return [
        (s.tool_name, dict(s.tool_args) if s.tool_args is not None else None)
        for s in steps
        if s.halt_reason is None
    ]


def _assert_one_turn(steps: list[TraceStep], halt_reason: HaltReason, tool_call_count: int) -> None:
    """Every row shares the turn; steps are numbered 0.. and end in exactly one terminal row."""
    assert steps
    assert len({s.turn_id for s in steps}) == 1
    assert all(s.user_id == USER_ID for s in steps)
    assert all(s.thread_id == THREAD_ID for s in steps)
    assert all(s.prompt_version == PLANNER_PROMPTS.version for s in steps)
    assert [s.step_index for s in steps] == list(range(tool_call_count + 1))
    terminal = steps[-1]
    assert terminal.halt_reason is halt_reason
    assert terminal.tool_name is None
    assert terminal.tool_args is None
    assert all(s.halt_reason is None and s.tool_name is not None for s in steps[:-1])


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

    def test_each_turn_gets_its_own_turn_id(self, planner_fakes: PlannerFakes) -> None:
        _run(planner_fakes, _scripted(final_message(FINAL_TEXT)))
        _run(planner_fakes, _scripted(final_message(FINAL_TEXT)))

        first, second = planner_fakes.trace.steps
        assert first.turn_id != second.turn_id


class TestTraceStepBound:
    """A turn halted at MAX_PLANNER_STEPS ends in a MAX_STEPS row after the calls actually made."""

    def test_terminal_row_is_max_steps(self, planner_fakes: PlannerFakes) -> None:
        model = FakePlannerModel(repeat=tool_call(SUGGEST, "call-forever"))

        result = _run(planner_fakes, model)

        steps = planner_fakes.trace.steps
        _assert_one_turn(steps, HaltReason.MAX_STEPS, tool_call_count=len(result.tool_calls))
        assert _step_calls(steps) == [(c.name, dict(c.args)) for c in result.tool_calls]


class TestTraceForbiddenWrite:
    """A refused write is traced as a step, the turn ends in FORBIDDEN_WRITE, and the error surfaces."""

    def test_forbidden_write_is_traced_and_still_raises(self, planner_fakes: PlannerFakes) -> None:
        args: dict[str, object] = {
            "kind": "topic_mastery",
            "key": DEFAULT_THEME.slug,
            "value": FORBIDDEN_WRITES["topic_mastery"],
        }
        model = _scripted(tool_call(RECORD, "call-record", **args), final_message(FINAL_TEXT))

        with pytest.raises(MemoryNotAllowed):
            _run(planner_fakes, model)

        steps = planner_fakes.trace.steps
        _assert_one_turn(steps, HaltReason.FORBIDDEN_WRITE, tool_call_count=1)
        assert _step_calls(steps) == [(RECORD, args)]


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

        _run(fakes, FakePlannerModel(repeat=tool_call(SUGGEST, "call-forever")))

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

        result = _run(planner_fakes, model, critic=reject_all_critic)

        assert result.quiz is None
        assert planner_fakes.trace.gaps == []
