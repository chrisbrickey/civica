"""Unit tests for civica.domain.question: Question validation and content-hash identity."""

import hashlib

import pytest
from pydantic import ValidationError

from civica.domain.question import Question, QuestionKind
from civica.domain.source_ref import SourceRef
from civica.domain.themes import DROITS_ET_DEVOIRS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_THEME = DROITS_ET_DEVOIRS
_TEXT = "sample-question-text"
_OTHER_TEXT = "other-sample-question-text"
_FOUR_OPTIONS = ("option-a", "option-b", "option-c", "option-d")
_THREE_OPTIONS = ("option-a", "option-b", "option-c")
_FIVE_OPTIONS = ("option-a", "option-b", "option-c", "option-d", "option-e")
_VALID_INDEX = 2
_OUT_OF_RANGE_INDEX = 4
_NEGATIVE_INDEX = -1
_SOURCES = (SourceRef(page_slug="sample-page", section_id="section-001"),)
_EXPLANATION = "sample-explanation"
_OTHER_EXPLANATION = "other-sample-explanation"


def _make_question(
    *,
    text: str = _TEXT,
    options: tuple[str, ...] = _FOUR_OPTIONS,
    correct_index: int = _VALID_INDEX,
    explanation: str = _EXPLANATION,
) -> Question:
    return Question(
        theme=_THEME,
        kind=QuestionKind.KNOWLEDGE,
        text=text,
        options=options,  # type: ignore[arg-type]
        correct_index=correct_index,
        sources=_SOURCES,
        explanation=explanation,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestValidQuestion:
    """A well-formed question keeps every field it was given."""

    def test_fields_round_trip(self) -> None:
        question = _make_question()

        assert question.theme == _THEME
        assert question.kind is QuestionKind.KNOWLEDGE
        assert question.text == _TEXT
        assert question.options == _FOUR_OPTIONS
        assert question.correct_index == _VALID_INDEX
        assert question.sources == _SOURCES
        assert question.explanation == _EXPLANATION

    def test_kind_values_are_the_persisted_strings(self) -> None:
        assert QuestionKind.KNOWLEDGE == "knowledge"
        assert QuestionKind.SCENARIO == "scenario"


class TestExplanation:
    """A question says why its correct option is right; older saved questions have none."""

    def test_question_saved_without_explanation_loads_with_empty_one(self) -> None:
        saved = _make_question().model_dump(exclude={"explanation"})

        assert Question.model_validate(saved).explanation == ""


class TestOptionCount:
    """Every question has exactly four options, as on the official exam."""

    @pytest.mark.parametrize("options", [_THREE_OPTIONS, _FIVE_OPTIONS])
    def test_wrong_option_count_rejected(self, options: tuple[str, ...]) -> None:
        with pytest.raises(ValidationError):
            _make_question(options=options)


class TestCorrectIndexRange:
    """correct_index must point at one of the four options."""

    @pytest.mark.parametrize("index", [_OUT_OF_RANGE_INDEX, _NEGATIVE_INDEX])
    def test_out_of_range_index_rejected(self, index: int) -> None:
        with pytest.raises(ValidationError):
            _make_question(correct_index=index)


class TestEmptyText:
    def test_empty_text_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _make_question(text="")


class TestQuestionId:
    """question_id is the sha256 of the text, so identical questions share a key."""

    def test_equals_sha256_of_text(self) -> None:
        expected = hashlib.sha256(_TEXT.encode("utf-8")).hexdigest()

        assert _make_question().question_id == expected

    def test_is_deterministic(self) -> None:
        assert _make_question().question_id == _make_question().question_id

    def test_differs_for_different_text(self) -> None:
        assert _make_question().question_id != _make_question(text=_OTHER_TEXT).question_id

    def test_ignores_the_explanation(self) -> None:
        other = _make_question(explanation=_OTHER_EXPLANATION)

        assert _make_question().question_id == other.question_id
