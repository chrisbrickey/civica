"""Unit tests for the question critic (Step 9B): the grounding and correctness gate.

No network calls. No DB spinup."""

import json
import string
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import PrivateAttr

from civica.assessment import prompts
from civica.assessment.critic import CritiqueFn, LlmCritic, Verdict
from civica.assessment.engine import (
    MAX_REFINE_ATTEMPTS,
    PROMPT_VERSION,
    QuestionGenerationError,
    generate_mock_exam,
    generate_quiz,
)
from civica.assessment.prompts import PROMPTS
from civica.assessment.scoring import MockExam, Quiz
from civica.domain.question import Question, QuestionKind
from civica.domain.source_ref import SourceRef
from civica.domain.themes import (
    EXAM_QUESTION_COUNTS,
    EXAM_SCENARIO_COUNTS,
    SYSTEME_INSTITUTIONNEL_ET_POLITIQUE,
    Theme,
)
from civica.domain.user import UserId
from civica.retrieval.content import ContentChunk
from tests.support.assessment import (
    SAMPLE_OPTIONS,
    RecordingAnswerLogger,
    RecordingQuestionStore,
    RecordingRejectionStore,
    RecordingRetriever,
    make_content_chunk,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_USER_ID = UserId(uuid.UUID(int=42))
_THEME = SYSTEME_INSTITUTIONNEL_ET_POLITIQUE
_EXPECTED_PROMPT_VERSION = "quiz-v003"
_EXAM_SIZE = 40
_SAMPLE_CORRECT_INDEX = 1

# Quiz of three; the retriever yields three sections, so passage group i is chunk i.
_QUIZ_SIZE = 3
_FAILING_POSITION = 1

_FIRST_TEXT = "sample-question-001"
_UNGROUNDED_TEXT = "sample-ungrounded-question"
_THIRD_TEXT = "sample-question-003"
_REGENERATED_TEXT = "sample-regenerated-question"
_UNGROUNDED_REASON = "sample-reason-ungrounded"

_PASS = Verdict(passed=True, reason="")

# Extra check the critic prompt must state verbatim; this string is the spec.
_SELF_CONTAINED_CHECK = (
    "5) la question se comprend seule et porte sur des connaissances civiques, pas sur les fiches."
)

# Generate, critique, then one regenerate and one critique per refine attempt.
_BOUNDED_CALL_COUNT = 1 + 1 + 2 * MAX_REFINE_ATTEMPTS

_EXAM_THEMES = list(EXAM_QUESTION_COUNTS)
_EXAM_FAILING_THEME = _EXAM_THEMES[0]
# The last item of a theme is a scenario question.
_EXAM_FAILING_POSITION = EXAM_QUESTION_COUNTS[_EXAM_FAILING_THEME] - 1
_EXAM_MAX_ITEMS_PER_CALL = max(EXAM_QUESTION_COUNTS.values())

_MALFORMED_CRITIC_REPLIES = {
    "not-json": "this is not json [",
    "object-not-array": json.dumps({"passed": True, "reason": ""}),
    "missing-passed": json.dumps([{"reason": "sample-reason"}, {"reason": "sample-reason"}]),
    "passed-not-bool": json.dumps(
        [{"passed": "sample-text", "reason": ""}, {"passed": "sample-text", "reason": ""}]
    ),
}


def _reason(attempt: int) -> str:
    return f"sample-reason-attempt-{attempt}"


def _sources(index: int) -> tuple[SourceRef, ...]:
    """Sources of passage group `index` from the default fake retriever."""
    return (SourceRef(page_slug=f"sample-page-{index:03d}", section_id=f"section-{index:03d}"),)


def _passage(theme: Theme, index: int) -> str:
    return make_content_chunk(theme, index).chunk.text


# ---------------------------------------------------------------------------
# Scripted replies
# ---------------------------------------------------------------------------


def _generation_reply(*texts: str) -> str:
    return json.dumps(
        [
            {"text": text, "options": SAMPLE_OPTIONS, "correct_index": _SAMPLE_CORRECT_INDEX}
            for text in texts
        ]
    )


def _critic_reply(*verdicts: Verdict) -> str:
    return json.dumps([{"passed": v.passed, "reason": v.reason} for v in verdicts])


def _fail(reason: str) -> Verdict:
    return Verdict(passed=False, reason=reason)


# ---------------------------------------------------------------------------
# Fakes (dependency injection, not monkeypatching)
# ---------------------------------------------------------------------------


class _ScriptedChatModel(BaseChatModel):  # type: ignore[explicit-any]
    """Fake chat model shared by generator and critic: returns `replies` in order.

    Once the script runs out it emits `fallback_items` generated questions with texts unique
    to the call, or fails loudly when `fallback_items` is 0.
    """

    replies: tuple[str, ...] = ()
    fallback_items: int = 0

    _received: list[list[BaseMessage]] = PrivateAttr(default_factory=list)

    @property
    def invocation_count(self) -> int:
        return len(self._received)

    def prompt_text(self, call_index: int) -> str:
        """Every message of one call joined, so assertions do not depend on message layout."""
        return "\n".join(str(m.content) for m in self._received[call_index])

    def _reply_for(self, call_index: int) -> str:
        if call_index < len(self.replies):
            return self.replies[call_index]
        assert self.fallback_items, f"unexpected model call #{call_index + 1}: script exhausted"
        return _generation_reply(
            *(f"sample-generated-question-{call_index:02d}-{i:02d}" for i in range(self.fallback_items))
        )

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        call_index = len(self._received)
        self._received.append(messages)
        message = AIMessage(content=self._reply_for(call_index))
        return ChatResult(generations=[ChatGeneration(message=message)])

    @property
    def _llm_type(self) -> str:
        return "sample-scripted-chat-model"


@dataclass(frozen=True)
class _CritiqueCall:
    theme: Theme
    questions: tuple[Question, ...]
    groups: tuple[tuple[ContentChunk, ...], ...]


@dataclass
class _ScriptedCritic:
    """Fake CritiqueFn: fails chosen positions on a theme's first pass, then passes or keeps failing."""

    fail_first_pass: dict[Theme, set[int]] = field(default_factory=dict)
    keep_failing: bool = False
    calls: list[_CritiqueCall] = field(default_factory=list)

    def __call__(
        self,
        theme: Theme,
        questions: Sequence[Question],
        groups: Sequence[Sequence[ContentChunk]],
    ) -> list[Verdict]:
        is_first_pass = not self.calls_for(theme)
        self.calls.append(
            _CritiqueCall(theme, tuple(questions), tuple(tuple(g) for g in groups))
        )
        attempt = len(self.calls_for(theme))
        failing = self.fail_first_pass.get(theme, set())
        verdicts = []
        for position in range(len(questions)):
            fails = position in failing if is_first_pass else bool(failing) and self.keep_failing
            verdicts.append(_fail(_reason(attempt)) if fails else _PASS)
        return verdicts

    def calls_for(self, theme: Theme) -> list[_CritiqueCall]:
        return [call for call in self.calls if call.theme == theme]


@dataclass(frozen=True)
class _Fakes:
    retriever: RecordingRetriever
    question_store: RecordingQuestionStore
    rejection_store: RecordingRejectionStore
    answer_logger: RecordingAnswerLogger


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def fakes(answer_logger: RecordingAnswerLogger) -> _Fakes:
    return _Fakes(
        retriever=RecordingRetriever(),
        question_store=RecordingQuestionStore(),
        rejection_store=RecordingRejectionStore(),
        answer_logger=answer_logger,
    )


@pytest.fixture()
def recovering_chat() -> _ScriptedChatModel:
    """Generation with one ungrounded item, critic fails it, regeneration fixes it, critic passes."""
    return _ScriptedChatModel(
        replies=(
            _generation_reply(_FIRST_TEXT, _UNGROUNDED_TEXT, _THIRD_TEXT),
            _critic_reply(_PASS, _fail(_UNGROUNDED_REASON), _PASS),
            _generation_reply(_REGENERATED_TEXT),
            _critic_reply(_PASS),
        )
    )


@pytest.fixture()
def always_failing_chat() -> _ScriptedChatModel:
    """The middle item fails every critic pass; extra script proves the loop stops on its own."""
    regenerations: list[str] = []
    for attempt in range(2, MAX_REFINE_ATTEMPTS + 4):
        regenerations += [
            _generation_reply(f"{_REGENERATED_TEXT}-{attempt}"),
            _critic_reply(_fail(_reason(attempt))),
        ]
    return _ScriptedChatModel(
        replies=(
            _generation_reply(_FIRST_TEXT, _UNGROUNDED_TEXT, _THIRD_TEXT),
            _critic_reply(_PASS, _fail(_reason(1)), _PASS),
            *regenerations,
        )
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _quiz(fakes: _Fakes, chat: _ScriptedChatModel, critic: CritiqueFn | None) -> Quiz:
    return generate_quiz(
        _USER_ID,
        _THEME,
        _QUIZ_SIZE,
        retriever=fakes.retriever,
        chat_client=chat,
        question_store=fakes.question_store,
        answer_logger=fakes.answer_logger,
        critic=critic,
        rejection_store=fakes.rejection_store,
    )


def _exam(fakes: _Fakes, critic: CritiqueFn) -> MockExam:
    return generate_mock_exam(
        _USER_ID,
        retriever=fakes.retriever,
        chat_client=_ScriptedChatModel(fallback_items=_EXAM_MAX_ITEMS_PER_CALL),
        question_store=fakes.question_store,
        answer_logger=fakes.answer_logger,
        critic=critic,
        rejection_store=fakes.rejection_store,
    )


def _sample_question(label: str, correct_index: int = 0) -> Question:
    """A question whose text and options are all unique to `label`."""
    options = tuple(f"sample-option-{label}-{i}" for i in range(len(SAMPLE_OPTIONS)))
    return Question(
        theme=_THEME,
        kind=QuestionKind.KNOWLEDGE,
        text=f"sample-critic-question-{label}",
        options=options,
        correct_index=correct_index,
        sources=_sources(0),
    )


def _critique_prompt(questions: Sequence[Question]) -> str:
    """Run LlmCritic once over `questions` (group i is chunk i) and return its prompt text."""
    chat = _ScriptedChatModel(replies=(_critic_reply(*[_PASS] * len(questions)),))
    groups = [[make_content_chunk(_THEME, i)] for i in range(len(questions))]
    LlmCritic(chat)(_THEME, questions, groups)
    return chat.prompt_text(0)


# ---------------------------------------------------------------------------
# Tests: prompt version
# ---------------------------------------------------------------------------


class TestPromptVersion:
    """Every prompt change moves the version stamp on."""

    def test_prompt_version_is_bumped(self) -> None:
        assert PROMPT_VERSION == _EXPECTED_PROMPT_VERSION
        assert prompts.PROMPT_VERSION == _EXPECTED_PROMPT_VERSION


# ---------------------------------------------------------------------------
# Tests: quiz through the LLM critic
# ---------------------------------------------------------------------------


class TestMalformedItemIsReplaced:
    """An item the critic fails is regenerated once and never reaches the learner."""

    def test_malformed_item_never_returned(
        self, fakes: _Fakes, recovering_chat: _ScriptedChatModel
    ) -> None:
        quiz = _quiz(fakes, recovering_chat, LlmCritic(recovering_chat))

        assert _UNGROUNDED_TEXT not in [q.text for q in quiz.questions]

    def test_returns_only_critic_passed_questions_in_group_order(
        self, fakes: _Fakes, recovering_chat: _ScriptedChatModel
    ) -> None:
        quiz = _quiz(fakes, recovering_chat, LlmCritic(recovering_chat))

        assert [q.text for q in quiz.questions] == [_FIRST_TEXT, _REGENERATED_TEXT, _THIRD_TEXT]

    def test_model_called_for_generate_critique_regenerate_critique(
        self, fakes: _Fakes, recovering_chat: _ScriptedChatModel
    ) -> None:
        _quiz(fakes, recovering_chat, LlmCritic(recovering_chat))

        assert recovering_chat.invocation_count == 4

    def test_regenerated_question_keeps_original_kind_and_sources(
        self, fakes: _Fakes, recovering_chat: _ScriptedChatModel
    ) -> None:
        quiz = _quiz(fakes, recovering_chat, LlmCritic(recovering_chat))

        regenerated = quiz.questions[_FAILING_POSITION]
        assert regenerated.kind is QuestionKind.KNOWLEDGE
        assert regenerated.sources == _sources(_FAILING_POSITION)

    def test_regeneration_prompt_holds_only_the_failed_group(
        self, fakes: _Fakes, recovering_chat: _ScriptedChatModel
    ) -> None:
        _quiz(fakes, recovering_chat, LlmCritic(recovering_chat))

        regeneration_prompt = recovering_chat.prompt_text(2)
        assert _passage(_THEME, _FAILING_POSITION) in regeneration_prompt
        for position in range(_QUIZ_SIZE):
            if position != _FAILING_POSITION:
                assert _passage(_THEME, position) not in regeneration_prompt

    def test_only_passed_questions_are_saved(
        self, fakes: _Fakes, recovering_chat: _ScriptedChatModel
    ) -> None:
        quiz = _quiz(fakes, recovering_chat, LlmCritic(recovering_chat))

        saved = fakes.question_store.saved_questions
        assert [q.question_id for q in saved] == [q.question_id for q in quiz.questions]
        assert _UNGROUNDED_TEXT not in [q.text for q in saved]
        assert fakes.question_store.saved_prompt_versions == {PROMPT_VERSION}

    def test_rejection_recorded_with_theme_reason_attempt_and_version(
        self, fakes: _Fakes, recovering_chat: _ScriptedChatModel
    ) -> None:
        _quiz(fakes, recovering_chat, LlmCritic(recovering_chat))

        [rejection] = fakes.rejection_store.rejections
        assert rejection.prompt_version == PROMPT_VERSION
        assert rejection.theme == _THEME
        assert rejection.reason == _UNGROUNDED_REASON
        assert rejection.attempt == 1


class TestPersistentlyFailingItemIsDropped:
    """An item that fails every pass is dropped after MAX_REFINE_ATTEMPTS; a short quiz is fine."""

    def test_model_calls_are_bounded(
        self, fakes: _Fakes, always_failing_chat: _ScriptedChatModel
    ) -> None:
        _quiz(fakes, always_failing_chat, LlmCritic(always_failing_chat))

        assert always_failing_chat.invocation_count == _BOUNDED_CALL_COUNT

    def test_quiz_is_short_and_keeps_only_passing_items(
        self, fakes: _Fakes, always_failing_chat: _ScriptedChatModel
    ) -> None:
        quiz = _quiz(fakes, always_failing_chat, LlmCritic(always_failing_chat))

        assert [q.text for q in quiz.questions] == [_FIRST_TEXT, _THIRD_TEXT]

    def test_dropped_item_is_not_saved(
        self, fakes: _Fakes, always_failing_chat: _ScriptedChatModel
    ) -> None:
        quiz = _quiz(fakes, always_failing_chat, LlmCritic(always_failing_chat))

        saved = fakes.question_store.saved_questions
        assert [q.question_id for q in saved] == [q.question_id for q in quiz.questions]
        assert [q.text for q in saved] == [_FIRST_TEXT, _THIRD_TEXT]

    def test_every_failed_pass_is_recorded_with_its_attempt(
        self, fakes: _Fakes, always_failing_chat: _ScriptedChatModel
    ) -> None:
        _quiz(fakes, always_failing_chat, LlmCritic(always_failing_chat))

        attempts = range(1, MAX_REFINE_ATTEMPTS + 2)
        rejections = fakes.rejection_store.rejections
        assert [(r.attempt, r.reason) for r in rejections] == [(a, _reason(a)) for a in attempts]
        assert {r.theme for r in rejections} == {_THEME}
        assert {r.prompt_version for r in rejections} == {PROMPT_VERSION}


class TestCriticDisabled:
    """critic=None keeps Step 9 behavior for raw-vs-critiqued comparison runs."""

    def test_returns_and_saves_every_generated_question_without_rejections(
        self, fakes: _Fakes
    ) -> None:
        chat = _ScriptedChatModel(
            replies=(_generation_reply(_FIRST_TEXT, _UNGROUNDED_TEXT, _THIRD_TEXT),)
        )

        quiz = _quiz(fakes, chat, None)

        assert [q.text for q in quiz.questions] == [_FIRST_TEXT, _UNGROUNDED_TEXT, _THIRD_TEXT]
        assert [q.text for q in fakes.question_store.saved_questions] == [
            q.text for q in quiz.questions
        ]
        assert fakes.rejection_store.rejections == []
        assert chat.invocation_count == 1


# ---------------------------------------------------------------------------
# Tests: mock exam through a scripted critic
# ---------------------------------------------------------------------------


class TestMockExamCritique:
    """A mock exam must fill every theme with critic-passed questions or save nothing."""

    def test_all_passing_returns_forty_questions_without_rejections(self, fakes: _Fakes) -> None:
        exam = _exam(fakes, _ScriptedCritic())

        assert len(exam.questions) == _EXAM_SIZE
        assert len(fakes.question_store.saved_questions) == _EXAM_SIZE
        assert fakes.rejection_store.rejections == []

    def test_theme_that_cannot_be_filled_raises_and_saves_nothing(self, fakes: _Fakes) -> None:
        critic = _ScriptedCritic(
            fail_first_pass={_EXAM_FAILING_THEME: {_EXAM_FAILING_POSITION}}, keep_failing=True
        )

        with pytest.raises(QuestionGenerationError):
            _exam(fakes, critic)

        assert fakes.question_store.saved_questions == []
        rejections = [r for r in fakes.rejection_store.rejections if r.theme == _EXAM_FAILING_THEME]
        assert [r.attempt for r in rejections] == list(range(1, MAX_REFINE_ATTEMPTS + 2))
        assert {r.prompt_version for r in rejections} == {PROMPT_VERSION}

    def test_recovered_scenario_item_keeps_position_kind_and_sources(self, fakes: _Fakes) -> None:
        critic = _ScriptedCritic(fail_first_pass={_EXAM_FAILING_THEME: {_EXAM_FAILING_POSITION}})

        exam = _exam(fakes, critic)

        first_pass, second_pass = critic.calls_for(_EXAM_FAILING_THEME)
        rejected = first_pass.questions[_EXAM_FAILING_POSITION]
        [regenerated] = second_pass.questions
        assert second_pass.groups == (first_pass.groups[_EXAM_FAILING_POSITION],)

        theme_questions = [q for q in exam.questions if q.theme == _EXAM_FAILING_THEME]
        assert theme_questions[_EXAM_FAILING_POSITION] == regenerated
        assert regenerated.kind is QuestionKind.SCENARIO
        assert regenerated.sources == rejected.sources
        assert rejected not in exam.questions
        assert len(exam.questions) == _EXAM_SIZE
        assert sum(q.kind is QuestionKind.SCENARIO for q in exam.questions) == sum(
            EXAM_SCENARIO_COUNTS.values()
        )


# ---------------------------------------------------------------------------
# Tests: LlmCritic
# ---------------------------------------------------------------------------


class TestLlmCriticVerdicts:
    """One chat call per batch returns one verdict per question, in order."""

    @pytest.mark.parametrize("fenced", [False, True], ids=["plain", "code-fence"])
    def test_parses_one_verdict_per_question(self, fenced: bool) -> None:
        verdicts = [_PASS, _fail(_UNGROUNDED_REASON)]
        reply = _critic_reply(*verdicts)
        if fenced:
            reply = f"```json\n{reply}\n```"
        chat = _ScriptedChatModel(replies=(reply,))
        questions = [_sample_question("a"), _sample_question("b")]
        groups = [[make_content_chunk(_THEME, 0)], [make_content_chunk(_THEME, 1)]]

        assert LlmCritic(chat)(_THEME, questions, groups) == verdicts
        assert chat.invocation_count == 1

    def test_wrong_verdict_count_raises(self) -> None:
        chat = _ScriptedChatModel(replies=(_critic_reply(_PASS),))
        questions = [_sample_question("a"), _sample_question("b")]
        groups = [[make_content_chunk(_THEME, 0)], [make_content_chunk(_THEME, 1)]]

        with pytest.raises(QuestionGenerationError):
            LlmCritic(chat)(_THEME, questions, groups)

    @pytest.mark.parametrize(
        "reply", _MALFORMED_CRITIC_REPLIES.values(), ids=_MALFORMED_CRITIC_REPLIES.keys()
    )
    def test_malformed_reply_raises(self, reply: str) -> None:
        chat = _ScriptedChatModel(replies=(reply,))
        questions = [_sample_question("a"), _sample_question("b")]
        groups = [[make_content_chunk(_THEME, 0)], [make_content_chunk(_THEME, 1)]]

        with pytest.raises(QuestionGenerationError):
            LlmCritic(chat)(_THEME, questions, groups)

    def test_constructs_without_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The default client is built lazily, so wiring the critic needs no credentials."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        LlmCritic()


class TestLlmCriticPrompt:
    """The critic prompt is built from PROMPTS and carries everything needed to judge each item."""

    def test_critic_prompts_are_declared(self) -> None:
        assert PROMPTS["critic_system"].strip()
        assert PROMPTS["critic_human"].strip()

    def test_prompt_is_built_from_the_critic_templates(self) -> None:
        prompt = _critique_prompt([_sample_question("a")])

        assert PROMPTS["critic_system"] in prompt
        literals = [text.strip() for text, *_ in string.Formatter().parse(PROMPTS["critic_human"])]
        for literal in literals:
            assert literal in prompt

    def test_prompt_contains_theme_questions_options_and_passages(self) -> None:
        questions = [_sample_question("a"), _sample_question("b")]

        prompt = _critique_prompt(questions)

        assert _THEME.display_name_fr in prompt
        for index, question in enumerate(questions):
            assert question.text in prompt
            for option in question.options:
                assert option in prompt
            assert _passage(_THEME, index) in prompt

    def test_prompt_checks_question_stands_alone_on_civic_knowledge(self) -> None:
        prompt = _critique_prompt([_sample_question("a")])

        assert _SELF_CONTAINED_CHECK in prompt

    def test_prompt_marks_the_answer(self) -> None:
        """Changing only the marked answer changes the prompt the critic sees."""
        first = _critique_prompt([_sample_question("a", correct_index=0)])
        second = _critique_prompt([_sample_question("a", correct_index=2)])

        assert first != second
