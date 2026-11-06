"""Shapes a thread's saved graph state into what the coach page shows. No NiceGUI import."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from itertools import pairwise
from typing import Final

from langgraph.types import StateSnapshot

from civica.graph.planner import HaltReason
from civica.ingestion.chunker import CHUNK_SIZE, OVERLAP
from civica.retrieval.content import ContentChunk

UNFINISHED_TEXT: Final = "Your last request didn't finish. Please try again."

_FALLBACK_REASONS: Final = frozenset({HaltReason.MAX_STEPS, HaltReason.FORBIDDEN_WRITE})

# Ingest writes each chunk as "<title> - <heading>\n\n<passage>". Titles may contain the separator.
_LABEL_SEPARATOR: Final = " - "
_LABEL_END: Final = "\n\n"
_UNTITLED_SOURCE: Final = "Official source"
_CUT_MARK: Final = "…"


@dataclass(frozen=True)
class SourceView:
    """One section the lesson drew on, as the learner sees it: no slugs or scores."""

    page_title: str
    section_heading: str
    passage: str


@dataclass(frozen=True)
class SourcePage:
    """One page's sources, so the learner sees each page title once."""

    page_title: str
    sources: tuple[SourceView, ...]


@dataclass(frozen=True)
class QuizItemView:
    """A question as the learner sees it: the correct answer is never carried."""

    text: str
    options: tuple[str, ...]


@dataclass(frozen=True)
class TurnView:
    learner_message: str
    reply: str
    lesson_text: str | None
    sources: tuple[SourceView, ...]
    quiz_items: tuple[QuizItemView, ...]
    is_fallback: bool
    notice: str | None


def _unfinished(learner_message: str) -> TurnView:
    return TurnView(
        learner_message=learner_message, reply="", lesson_text=None, sources=(), quiz_items=(),
        is_fallback=False, notice=UNFINISHED_TEXT,
    )


def _split_labels(text: str) -> tuple[str, str, str]:
    """A chunk's page title, section heading, and passage."""
    labels, end, passage = text.partition(_LABEL_END)
    if not end:
        return _UNTITLED_SOURCE, "", text
    title, separator, heading = labels.rpartition(_LABEL_SEPARATOR)
    return (title, heading, passage) if separator else (labels, "", passage)


def _joined(windows: Mapping[int, str]) -> str:
    """Rebuild a section from its retrieved windows, marking where text was left out."""
    indexes = sorted(windows)
    text = (_CUT_MARK if indexes[0] > 0 else "") + windows[indexes[0]]
    for previous, index in pairwise(indexes):
        adjacent = index == previous + 1
        text += windows[index][OVERLAP:] if adjacent else _CUT_MARK + windows[index]
    return text + _CUT_MARK if len(windows[indexes[-1]]) == CHUNK_SIZE else text


def _sources(chunks: Iterable[ContentChunk]) -> tuple[SourceView, ...]:
    """One source per section, in retrieval order."""
    labels: dict[tuple[str, str], tuple[str, str]] = {}
    windows: dict[tuple[str, str], dict[int, str]] = {}
    for item in chunks:
        key = (item.chunk.page_slug, item.chunk.section_id)
        title, heading, passage = _split_labels(item.chunk.text)
        labels.setdefault(key, (title, heading))
        windows.setdefault(key, {})[item.chunk.chunk_index] = passage
    return tuple(SourceView(*labels[key], _joined(parts)) for key, parts in windows.items())


def group_by_page(sources: Iterable[SourceView]) -> tuple[SourcePage, ...]:
    """Sources under their page title, pages in the order first seen. Titles are what the learner tells apart."""
    pages: dict[str, list[SourceView]] = {}
    for source in sources:
        pages.setdefault(source.page_title, []).append(source)
    return tuple(SourcePage(title, tuple(items)) for title, items in pages.items())


def build_turn_view(snapshot: StateSnapshot) -> TurnView | None:  # type: ignore[explicit-any]
    """The turn saved on the thread, or None for an empty thread.

    A pending node means the turn never finished, and its values may hold the previous turn's output.
    """
    values = snapshot.values
    if not values:
        return None
    if snapshot.next:
        return _unfinished(values.get("learner_message", ""))

    lesson = values["lesson"]
    reply = values["reply"]
    quiz_questions = values["quiz_questions"]
    is_fallback = values["halt_reason"] in _FALLBACK_REASONS
    chunks = lesson.context_chunks if lesson is not None else ()
    return TurnView(
        learner_message=values["learner_message"],
        # The lesson and quiz are shown on their own, so the planner reply is hidden then.
        reply="" if lesson is not None or quiz_questions else reply,
        lesson_text=lesson.text if lesson is not None else None,
        sources=_sources(chunks),
        quiz_items=tuple(QuizItemView(q.text, q.options) for q in quiz_questions),
        is_fallback=is_fallback,
        notice=None,
    )
