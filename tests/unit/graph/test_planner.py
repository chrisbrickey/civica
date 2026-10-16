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
FINAL_TEXT = "sample-final-reply"
FOCUS = "sample-focus"
QUERY = "sample-query"
GOAL = "sample-goal"
UNKNOWN_THEME_SLUG = "sample-unknown-theme"
UNKNOWN_KIND = "sample-unknown-kind"
SAMPLE_KEY = "sample-key-001"
SAMPLE_QUESTION_ID = "sample-question-id-001"
# Plan key convention: one LearnerProfile per user under a constant key.
PROFILE_KEY = "current"

THEME = DROITS_ET_DEVOIRS
# Seeded as the only unmastered theme, so the review heuristic always suggests it.
SUGGESTED_THEME = VIVRE_DANS_LA_SOCIETE_FRANCAISE
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

MISTAKE = MistakeEpisode(theme=THEME, question_id=SAMPLE_QUESTION_ID)
MISTAKE_VALUE: dict[str, object] = {
    "theme": THEME.slug,
    "question_id": SAMPLE_QUESTION_ID,
    "sources": [],
}

# Well-formed values for kinds the planner must never write, so only the allowlist can refuse them.
FORBIDDEN_WRITES: dict[str, dict[str, object]] = {
    "learner_profile": {"goal": GOAL},
    "topic_mastery": {"theme": THEME.slug, "mastery": FULL_MASTERY},
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
    )


def _run(
    fakes: PlannerFakes,
    planner_model: FakePlannerModel,
    *,
    critic: CritiqueFn | None = None,
    mode: PlannerMode = PlannerMode.TEACH,
) -> PlannerResult:
    return run_planner(USER_ID, LEARNER_MESSAGE, mode, _deps(fakes, planner_model, critic))


def _scripted(*turns: AIMessage) -> FakePlannerModel:
    return FakePlannerModel(script=tuple(turns))


def _favor_suggested_theme(store: InMemoryStore) -> None:
    """Master every theme except SUGGESTED_THEME."""
    seed_mastery(
        store,
        USER_ID,
        {theme: (0.0 if theme == SUGGESTED_THEME else FULL_MASTERY) for theme in EXAM_QUESTION_COUNTS},
    )


def _teach(call_id: str, theme_slug: str = THEME.slug) -> AIMessage:
    return tool_call(TEACH, call_id, theme=theme_slug, focus=FOCUS)


def _quiz(call_id: str, n: int, theme_slug: str = THEME.slug) -> AIMessage:
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
            ToolCall(name=TEACH, args={"theme": THEME.slug, "focus": FOCUS}),
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
        assert {c.chunk.theme for c in result.lesson.context_chunks} == {THEME}
        assert (FOCUS, THEME) in planner_fakes.retriever.calls

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
                RECORD, "call-record", kind="topic_mastery", key=THEME.slug,
                value=FORBIDDEN_WRITES["topic_mastery"],
            ),
            final_message(FINAL_TEXT),
        )
        mastery = TopicMastery(theme=THEME, mastery=FULL_MASTERY)

        with pytest.raises(MemoryNotAllowed):
            _run(planner_fakes, model)
        put(USER_ID, THEME.slug, mastery, planner_fakes.store)

        assert get(USER_ID, THEME.slug, TopicMastery, planner_fakes.store) == mastery

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
        assert {q.theme for q in result.quiz.questions} == {THEME}
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
            tool_call(RETRIEVE, "call-retrieve", theme=THEME.slug, query=QUERY),
            final_message(FINAL_TEXT),
        )

        _run(planner_fakes, model)

        assert (QUERY, THEME) in planner_fakes.retriever.calls
        output = model.tool_output("call-retrieve")
        for chunk in planner_fakes.retriever.chunks(THEME):
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
        recorded = {THEME: HALF_MASTERY, SYSTEME_INSTITUTIONNEL_ET_POLITIQUE: FULL_MASTERY}
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
