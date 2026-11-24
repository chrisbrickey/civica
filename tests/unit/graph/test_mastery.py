"""Unit tests for civica.graph.mastery, the derived mastery formula."""

import pytest

from civica.domain.source_ref import SourceRef
from civica.graph.mastery import mastery_from_answers
from civica.progress.quiz_log import SourcedAnswer

_A = SourceRef(page_slug="sample-page", section_id="section-a")
_B = SourceRef(page_slug="sample-page", section_id="section-b")
_C = SourceRef(page_slug="sample-page", section_id="section-c")
_D = SourceRef(page_slug="sample-page", section_id="section-d")
_OUTSIDE = SourceRef(page_slug="other-page", section_id="section-x")

_SECTION_SIZE = 100
# Four sections of equal size, so each is a quarter of the theme.
_EQUAL_SECTIONS = {source: _SECTION_SIZE for source in (_A, _B, _C, _D)}


def _right(*sources: SourceRef) -> SourcedAnswer:
    return SourcedAnswer(is_correct=True, sources=list(sources))


def _wrong(*sources: SourceRef) -> SourcedAnswer:
    return SourcedAnswer(is_correct=False, sources=list(sources))


def test_a_right_answer_earns_its_section_share_of_the_theme() -> None:
    assert mastery_from_answers([_right(_A)], _EQUAL_SECTIONS) == pytest.approx(0.25)


def test_credit_is_split_equally_across_the_cited_sections() -> None:
    # 1/3 credit to each of three sections, out of four sections.
    assert mastery_from_answers([_right(_A, _B, _C)], _EQUAL_SECTIONS) == pytest.approx(0.25)


def test_sections_are_weighted_by_their_content_size() -> None:
    sections = {_A: 100, _B: 300}

    assert mastery_from_answers([_right(_B)], sections) == pytest.approx(0.75)
    assert mastery_from_answers([_right(_A)], sections) == pytest.approx(0.25)


@pytest.mark.parametrize(
    ("answer_count", "expected"),
    [(1, 0.25), (2, 0.375), (3, 0.125), (4, 0.375)],
)
def test_a_wrong_answer_subtracts_the_share_a_right_answer_would_add(answer_count: int, expected: float) -> None:
    answers = [_right(_A), _right(_A, _B), _wrong(_A, _B), _right(_B)]

    assert mastery_from_answers(answers[:answer_count], _EQUAL_SECTIONS) == pytest.approx(expected)


def test_a_section_caps_at_full_credit_so_extra_credit_is_not_banked() -> None:
    answers = [_right(_A), _right(_A), _wrong(_A)]

    assert mastery_from_answers(answers, _EQUAL_SECTIONS) == pytest.approx(0.0)


def test_a_section_never_drops_below_zero() -> None:
    answers = [_wrong(_A), _wrong(_A), _right(_A)]

    assert mastery_from_answers(answers, _EQUAL_SECTIONS) == pytest.approx(0.25)


def test_full_mastery_needs_full_credit_in_every_section() -> None:
    answers = [_right(_A), _right(_B), _right(_C)]

    assert mastery_from_answers(answers, _EQUAL_SECTIONS) == pytest.approx(0.75)
    assert mastery_from_answers([*answers, _right(_D)], _EQUAL_SECTIONS) == pytest.approx(1.0)


def test_a_cited_section_outside_the_theme_keeps_its_share_but_earns_nothing() -> None:
    assert mastery_from_answers([_right(_A, _OUTSIDE)], _EQUAL_SECTIONS) == pytest.approx(0.125)


def test_no_answers_give_no_mastery() -> None:
    assert mastery_from_answers([], _EQUAL_SECTIONS) is None


def test_a_theme_with_no_sections_gives_no_mastery() -> None:
    assert mastery_from_answers([_right(_A)], {}) is None
