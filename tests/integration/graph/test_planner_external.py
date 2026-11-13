"""External contract test on a real agent using the planner's tools.

Requires ANTHROPIC_API_KEY and network access.
The planner model is real. Content generation, retrieval, and storage are faked.
"""

import json
from typing import Any

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.memory import InMemoryStore

from civica.domain.themes import SYSTEME_INSTITUTIONNEL_ET_POLITIQUE
from civica.graph.graph import InputKind, build_graph
from civica.graph.nodes import PlannerContext
from civica.graph.planner import HaltReason, PlannerDeps, ToolCall, build_planner_tools
from civica.llm.client import PLANNER_PROFILE, get_chat_model
from tests.support.assessment import (
    FakeAnswerLogger,
    FakeChatModel,
    FakeQuestionStore,
    FakeRejectionStore,
    FakeRetriever,
    grouped_generation_reply,
)
from tests.support.planner import FakeTraceWriter, LessonOrQuizModel
from tests.support.users import USER_ID

_LEARNER_MESSAGE = "Teach me about the rights and duties of citizens."
_CONFIG: RunnableConfig = {"configurable": {"thread_id": "sample-thread-external"}}

# Not the review suggestion for an empty store, so a turn without context would drift elsewhere.
_FOLLOW_UP_THEME = SYSTEME_INSTITUTIONNEL_ET_POLITIQUE
_LESSON_REQUEST = "Teach me about the French institutional and political system."
_LESSON_FOLLOW_UPS = ("Give me another example.", "Explain that more simply.")
_FOLLOW_UP_LESSON = (
    "Le Parlement has two chambers, l'Assemblée nationale and le Sénat. Together they vote the laws."
)
_FOLLOW_UP_QUIZ = grouped_generation_reply((1, "Which two chambers make up le Parlement?"))

_QUIZ_REQUEST = "Quiz me with 3 questions on the principles and values of the Republic."
_EXPLAIN_SECOND = "Explain question 2."
# Names none of the stems' keywords, so only the previous exchange can point the model at question 2.
_VALUES_LESSON = "La Constitution sets out the principles of la République française."
# Each stem with words only an answer about that stem would use.
_STEM_KEYWORDS: dict[str, tuple[str, ...]] = {
    "Which colours make up the French flag?": ("flag", "drapeau", "tricolore"),
    "What is the motto of the French Republic?": ("motto", "devise", "fraternit"),
    "Which song is the French national anthem?": ("anthem", "hymne", "marseillaise"),
}

_QUIZ_TOOL = "generate_quiz"
# Tools whose arguments name the theme being taught or quizzed.
_THEMED_TOOLS = {"teach", "retrieve_corpus", _QUIZ_TOOL}

_Graph = CompiledStateGraph[Any, PlannerContext, Any, Any]  # type: ignore[explicit-any]
_State = dict[str, Any]  # type: ignore[explicit-any]


def _deps(content_model: BaseChatModel) -> PlannerDeps:
    """A real planner model over faked content, retrieval, and storage."""
    return PlannerDeps(
        planner_model=get_chat_model(PLANNER_PROFILE),
        store=InMemoryStore(),
        question_store=FakeQuestionStore(),
        answer_logger=FakeAnswerLogger(),
        rejection_store=FakeRejectionStore(),
        trace=FakeTraceWriter(),
        content_model=content_model,
        retriever=FakeRetriever(),
    )


def _turn(graph: _Graph, message: str) -> _State:
    state: _State = graph.invoke(
        {"kind": InputKind.MESSAGE, "learner_message": message},
        config=_CONFIG,
        context=PlannerContext(user_id=USER_ID),
    )
    return state


def _themes(calls: list[ToolCall]) -> set[object]:
    return {call.args.get("theme") for call in calls if call.name in _THEMED_TOOLS}


def _names(calls: list[ToolCall]) -> set[str]:
    return {call.name for call in calls}


def _args_text(calls: list[ToolCall]) -> str:
    return json.dumps([dict(call.args) for call in calls], ensure_ascii=False).lower()


def _mentions(text: str, keywords: tuple[str, ...]) -> bool:
    return any(keyword in text for keyword in keywords)


@pytest.mark.external
def test_real_model_calls_tools_within_their_schemas_and_ends_the_turn() -> None:
    deps = _deps(FakeChatModel(items_per_call=1))
    schemas = {
        function["name"]: function.get("parameters", {})
        for function in (convert_to_openai_tool(t)["function"] for t in build_planner_tools(deps))
    }

    state = _turn(build_graph(deps, checkpointer=InMemorySaver()), _LEARNER_MESSAGE)

    assert state["tool_calls"]
    for call in state["tool_calls"]:
        schema = schemas[call.name]
        assert set(schema.get("required", [])) <= set(call.args)
        assert set(call.args) <= set(schema.get("properties", {}))
    assert state["halt_reason"] == HaltReason.FINAL_MESSAGE
    assert state["reply"].strip()


@pytest.mark.external
@pytest.mark.parametrize("follow_up", _LESSON_FOLLOW_UPS)
def test_lesson_follow_up_stays_on_theme_without_a_new_check(follow_up: str) -> None:
    content_model = LessonOrQuizModel(lesson=_FOLLOW_UP_LESSON, quiz_reply=_FOLLOW_UP_QUIZ)
    graph = build_graph(_deps(content_model), checkpointer=InMemorySaver())

    first = _turn(graph, _LESSON_REQUEST)
    second = _turn(graph, follow_up)

    assert _themes(first["tool_calls"]) == {_FOLLOW_UP_THEME.slug}
    assert _themes(second["tool_calls"]) == {_FOLLOW_UP_THEME.slug}
    assert _QUIZ_TOOL not in _names(second["tool_calls"])


@pytest.mark.external
def test_explain_question_2_is_about_the_second_displayed_question() -> None:
    quiz = grouped_generation_reply(*enumerate(_STEM_KEYWORDS, start=1))
    content_model = LessonOrQuizModel(lesson=_VALUES_LESSON, quiz_reply=quiz)
    graph = build_graph(_deps(content_model), checkpointer=InMemorySaver())

    questions = _turn(graph, _QUIZ_REQUEST)["quiz_questions"]
    second = _turn(graph, _EXPLAIN_SECOND)

    assert len(questions) >= 2
    target = questions[1].text
    others = [q.text for i, q in enumerate(questions) if i != 1]
    args = _args_text(second["tool_calls"])
    assert _mentions(f"{second['reply'].lower()}\n{args}", _STEM_KEYWORDS[target])
    assert not any(_mentions(args, _STEM_KEYWORDS[other]) for other in others)
    assert _QUIZ_TOOL not in _names(second["tool_calls"])
