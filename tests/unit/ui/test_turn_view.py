"""Unit tests for civica.ui.turn_view. Final states are built by hand: no graph, no database."""

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest
from langgraph.types import StateSnapshot

from civica.domain.chunk import Chunk
from civica.domain.question import Question
from civica.embeddings.embedder import Embedder
from civica.explanation.engine import Explanation
from civica.graph.graph import PLANNER, InputKind
from civica.graph.planner import HaltReason, ToolCall
from civica.ingestion.chunker import CHUNK_SIZE
from civica.retrieval.content import ContentChunk
from civica.scripts.ingest_corpus import collect_pending_chunks
from civica.ui.turn_view import (
    QuizItemView,
    SourcePage,
    SourceView,
    TurnView,
    build_turn_view,
    group_by_page,
)
from tests.support.assessment import SAMPLE_OPTIONS, make_content_chunk, make_question
from tests.support.planner import (
    DEFAULT_THEME,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_LEARNER_MESSAGE = "sample-learner-message-001"
_REPLY = "sample-final-reply"
_LESSON_TEXT = "sample-lesson-text"
_INSUFFICIENT_TEXT = "sample-insufficient-material-text"
_FALLBACK_TEXT = "sample-fallback-lesson-text"
_FIRST_QUESTION = "sample-question-text-001"
_SECOND_QUESTION = "sample-question-text-002"

_UNFINISHED_TEXT = "Your last request didn't finish. Please try again."

_PAGE_TITLE = "sample-page-title"
_HEADING = "sample-section-heading"
_BODY = "sample-passage-body"
_OTHER_BODY = "sample-other-passage-body"
_UNTITLED_SOURCE = "Official source"

_TEACH = "teach"

# Deliberately unsorted, so a view that re-ranks the passages is caught.
_SIMILARITIES = (0.7, 0.9, 0.8)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chunks() -> tuple[ContentChunk, ...]:
    return tuple(
        make_content_chunk(DEFAULT_THEME, index).model_copy(update={"similarity": similarity})
        for index, similarity in enumerate(_SIMILARITIES)
    )


def _lesson(text: str = _LESSON_TEXT) -> Explanation:
    return Explanation(text=text, context_chunks=_chunks())


def _state(**overrides: object) -> dict[str, object]:
    """The graph's values at the end of a turn that only sent a reply."""
    state: dict[str, object] = {
        "kind": InputKind.MESSAGE,
        "learner_message": _LEARNER_MESSAGE,
        "reply": _REPLY,
        "lesson": None,
        "quiz_questions": [],
        "tool_calls": [],
        "halt_reason": HaltReason.FINAL_MESSAGE,
    }
    return state | overrides


def _snapshot(values: Mapping[str, object], next_nodes: Sequence[str] = ()) -> StateSnapshot:
    return StateSnapshot(
        values=dict(values),
        next=tuple(next_nodes),
        config={},
        metadata=None,
        created_at=None,
        parent_config=None,
        tasks=(),
        interrupts=(),
    )


def _view(**overrides: object) -> TurnView:
    view = build_turn_view(_snapshot(_state(**overrides)))
    assert view is not None
    return view


def _quiz_item(question: Question) -> QuizItemView:
    return _view(quiz_questions=[question]).quiz_items[0]


# ---------------------------------------------------------------------------
# Finished turns
# ---------------------------------------------------------------------------


def test_turn_with_no_lesson_or_quiz_shows_the_learner_message_and_reply() -> None:
    view = _view()

    assert view.learner_message == _LEARNER_MESSAGE
    assert view.reply == _REPLY


def test_turn_without_a_lesson_has_no_lesson_text_or_sources() -> None:
    view = _view()

    assert view.lesson_text is None
    assert view.sources == ()


def test_grounded_lesson_shows_its_text() -> None:
    assert _view(lesson=_lesson()).lesson_text == _LESSON_TEXT


# The learner already sees the lesson and quiz cards, so the planner's reply would only repeat them.
@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"lesson": _lesson()}, id="grounded-lesson"),
        pytest.param({"lesson": Explanation(text=_INSUFFICIENT_TEXT, context_chunks=())}, id="insufficient-material"),
        pytest.param({"quiz_questions": [make_question()]}, id="quiz-only"),
        pytest.param({"lesson": _lesson(), "quiz_questions": [make_question()]}, id="lesson-and-quiz"),
    ],
)
def test_turn_with_a_lesson_or_quiz_hides_the_planner_reply(overrides: dict[str, object]) -> None:
    assert _view(**overrides).reply == ""


def test_insufficient_material_lesson_shows_its_text_with_no_sources() -> None:
    view = _view(lesson=Explanation(text=_INSUFFICIENT_TEXT, context_chunks=()))

    assert view.lesson_text == _INSUFFICIENT_TEXT
    assert view.sources == ()


def test_quiz_questions_map_to_items_in_order_with_their_options() -> None:
    questions = [make_question(text=_FIRST_QUESTION), make_question(text=_SECOND_QUESTION)]

    view = _view(quiz_questions=questions)

    assert view.quiz_items == (
        QuizItemView(text=_FIRST_QUESTION, options=SAMPLE_OPTIONS),
        QuizItemView(text=_SECOND_QUESTION, options=SAMPLE_OPTIONS),
    )


def test_quiz_item_never_reveals_the_correct_answer() -> None:
    first_option_correct = make_question(correct_index=0)
    last_option_correct = make_question(correct_index=len(SAMPLE_OPTIONS) - 1)

    assert _quiz_item(first_option_correct) == _quiz_item(last_option_correct)


@pytest.mark.parametrize("halt_reason", [HaltReason.MAX_STEPS, HaltReason.FORBIDDEN_WRITE])
def test_turn_that_ended_early_is_flagged_and_shows_the_fallback_text_once(
    halt_reason: HaltReason,
) -> None:
    view = _view(lesson=_lesson(_FALLBACK_TEXT), reply=_FALLBACK_TEXT, halt_reason=halt_reason)

    assert view.is_fallback
    assert [view.lesson_text, view.reply].count(_FALLBACK_TEXT) == 1


def test_turn_ended_by_a_final_message_is_not_a_fallback() -> None:
    assert not _view(lesson=_lesson()).is_fallback


def test_finished_turn_has_no_notice() -> None:
    assert _view().notice is None


# ---------------------------------------------------------------------------
# Empty and unfinished threads
# ---------------------------------------------------------------------------


def test_empty_thread_yields_no_turn_view() -> None:
    assert build_turn_view(_snapshot({})) is None


def test_unfinished_turn_shows_the_notice_and_none_of_the_stale_turn() -> None:
    stale = _state(
        lesson=_lesson(),
        quiz_questions=[make_question()],
        tool_calls=[ToolCall(name=_TEACH, args={})],
        halt_reason=HaltReason.MAX_STEPS,
    )

    view = build_turn_view(_snapshot(stale, next_nodes=[PLANNER]))

    assert view is not None
    assert view.notice == _UNFINISHED_TEXT
    assert view.lesson_text is None
    assert not view.reply
    assert view.sources == ()
    assert view.quiz_items == ()
    assert not view.is_fallback


def test_first_turn_that_never_finished_shows_the_notice() -> None:
    # Only the input is saved when the very first turn of a thread dies.
    partial = {"kind": InputKind.MESSAGE, "learner_message": _LEARNER_MESSAGE}

    view = build_turn_view(_snapshot(partial, next_nodes=[PLANNER]))

    assert view is not None
    assert view.notice == _UNFINISHED_TEXT


# ---------------------------------------------------------------------------
# Learner sources
# ---------------------------------------------------------------------------


def _titled_chunk(
    body: str, *, heading: str = _HEADING, title: str = _PAGE_TITLE, section: int = 0, chunk_index: int = 0
) -> ContentChunk:
    """A retrieved chunk whose text starts with its title and heading, as ingest writes it."""
    chunk = make_content_chunk(DEFAULT_THEME, section, chunk_index)
    prefix = " - ".join(part for part in (title, heading) if part)
    text = f"{prefix}\n\n{body}" if prefix else body
    return chunk.model_copy(update={"chunk": chunk.chunk.model_copy(update={"text": text})})


def _sources(*chunks: ContentChunk) -> tuple[SourceView, ...]:
    return _view(lesson=Explanation(text=_LESSON_TEXT, context_chunks=chunks)).sources


def test_source_shows_the_page_title_heading_and_passage() -> None:
    assert _sources(_titled_chunk(_BODY)) == (SourceView(_PAGE_TITLE, _HEADING, _BODY),)


def test_source_title_may_itself_contain_the_separator() -> None:
    title = f"{_PAGE_TITLE} - sample-subtitle"

    assert _sources(_titled_chunk(_BODY, title=title)) == (SourceView(title, _HEADING, _BODY),)


def test_source_with_only_one_label_uses_it_as_the_title() -> None:
    assert _sources(_titled_chunk(_BODY, heading="")) == (SourceView(_PAGE_TITLE, "", _BODY),)


def test_source_with_no_labels_gets_a_plain_title_and_never_a_slug() -> None:
    source = _sources(_titled_chunk(_BODY, title="", heading=""))[0]

    assert source.page_title == _UNTITLED_SOURCE
    assert source.passage == _BODY


def test_two_passages_from_one_section_are_one_source() -> None:
    chunks = (_titled_chunk(_BODY), _titled_chunk(_OTHER_BODY, chunk_index=3))

    view = _view(lesson=Explanation(text=_LESSON_TEXT, context_chunks=chunks))

    assert len(view.sources) == 1


def test_sources_keep_retrieval_order() -> None:
    chunks = (
        _titled_chunk(_BODY, heading="sample-heading-b", section=1),
        _titled_chunk(_BODY, heading="sample-heading-a", section=2),
        _titled_chunk(_OTHER_BODY, heading="sample-heading-b", section=1, chunk_index=2),
    )

    assert [source.section_heading for source in _sources(*chunks)] == ["sample-heading-b", "sample-heading-a"]


def test_repeated_passage_appears_once_in_its_source() -> None:
    assert _sources(_titled_chunk(_BODY), _titled_chunk(_BODY))[0].passage == _BODY


def test_passage_starting_partway_through_a_section_is_marked() -> None:
    assert _sources(_titled_chunk(_BODY, chunk_index=1))[0].passage == f"…{_BODY}"


def test_passage_cut_at_the_window_size_is_marked() -> None:
    body = "x" * CHUNK_SIZE

    assert _sources(_titled_chunk(body))[0].passage == f"{body}…"


def test_passages_with_a_gap_between_them_are_separated() -> None:
    chunks = (_titled_chunk(_BODY), _titled_chunk(_OTHER_BODY, chunk_index=2))

    assert _sources(*chunks)[0].passage == f"{_BODY}…{_OTHER_BODY}"


def test_sources_are_grouped_by_page_in_first_seen_order() -> None:
    first = SourceView("sample-page-a", "sample-heading-a", _BODY)
    second = SourceView("sample-page-b", "sample-heading-b", _BODY)
    third = SourceView("sample-page-a", "sample-heading-c", _OTHER_BODY)

    assert group_by_page((first, second, third)) == (
        SourcePage("sample-page-a", (first, third)),
        SourcePage("sample-page-b", (second,)),
    )


def test_parts_of_one_page_group_under_the_page_title_in_part_order() -> None:
    part_two = SourceView(f"{_PAGE_TITLE} (2/2)", "sample-heading-b", _BODY)
    other = SourceView("sample-page-b", "sample-heading-c", _BODY)
    part_one = SourceView(f"{_PAGE_TITLE} (1/2)", "sample-heading-a", _OTHER_BODY)

    assert group_by_page((part_two, other, part_one)) == (
        SourcePage(_PAGE_TITLE, (part_one, part_two)),
        SourcePage("sample-page-b", (other,)),
    )


def test_title_ending_in_other_brackets_is_kept_whole() -> None:
    source = SourceView(f"{_PAGE_TITLE} (sample-note)", _HEADING, _BODY)

    assert group_by_page((source,)) == (SourcePage(source.page_title, (source,)),)


def test_no_sources_make_no_pages() -> None:
    assert group_by_page(()) == ()


def test_insufficient_material_has_no_sources() -> None:
    assert _view(lesson=Explanation(text=_INSUFFICIENT_TEXT, context_chunks=())).sources == ()


def test_unfinished_turn_has_no_sources() -> None:
    view = build_turn_view(_snapshot(_state(lesson=_lesson()), next_nodes=[PLANNER]))

    assert view is not None
    assert view.sources == ()


def test_sources_read_back_exactly_what_ingest_wrote(tmp_path: Path) -> None:
    # Goes through real ingest, so a change to its text format or windows fails here.
    section_text = "".join(f"word-{index} " for index in range(CHUNK_SIZE))
    title = f"{_PAGE_TITLE} - sample-subtitle"
    page = {
        "theme": DEFAULT_THEME.slug,
        "slug": "sample-page",
        "title": title,
        "sections": [{"id": "sample-section", "heading": _HEADING, "text": section_text}],
    }
    (tmp_path / "sample-page.json").write_text(json.dumps(page), encoding="utf-8")
    embedder = Embedder(model="sample-embedding-model", embed=lambda _: [], embed_query=lambda _: [])
    pending = collect_pending_chunks(tmp_path, embedder=embedder)
    assert len(pending) > 1

    chunks = tuple(
        ContentChunk(
            chunk=Chunk(
                theme=DEFAULT_THEME, page_slug=row.page_slug, section_id=row.section_id,
                chunk_index=row.chunk_index, content_hash=row.content_hash, text=row.text,
            ),
            similarity=0.9,
        )
        for row in reversed(pending)
    )

    assert _sources(*chunks) == (SourceView(title, _HEADING, section_text),)
