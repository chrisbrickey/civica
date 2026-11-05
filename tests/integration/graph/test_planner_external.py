"""External contract test on a real agent using the planner's tools.

Requires ANTHROPIC_API_KEY and network access.
The planner model is real. Content generation, retrieval, and storage are faked.
"""

import pytest
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from civica.graph.graph import InputKind, build_graph
from civica.graph.nodes import PlannerContext
from civica.graph.planner import HaltReason, PlannerDeps, build_planner_tools
from civica.llm.client import PLANNER_PROFILE, get_chat_model
from tests.support.assessment import (
    FakeAnswerLogger,
    FakeChatModel,
    FakeQuestionStore,
    FakeRejectionStore,
    FakeRetriever,
)
from tests.support.planner import FakeTraceWriter
from tests.support.users import USER_ID

_LEARNER_MESSAGE = "Teach me about the rights and duties of citizens."
_CONFIG = {"configurable": {"thread_id": "sample-thread-external"}}


@pytest.mark.external
def test_real_model_calls_tools_within_their_schemas_and_ends_the_turn() -> None:
    deps = PlannerDeps(
        planner_model=get_chat_model(PLANNER_PROFILE),
        store=InMemoryStore(),
        question_store=FakeQuestionStore(),
        answer_logger=FakeAnswerLogger(),
        rejection_store=FakeRejectionStore(),
        trace=FakeTraceWriter(),
        content_model=FakeChatModel(items_per_call=1),
        retriever=FakeRetriever(),
    )
    schemas = {
        function["name"]: function.get("parameters", {})
        for function in (convert_to_openai_tool(t)["function"] for t in build_planner_tools(deps))
    }

    state = build_graph(deps, checkpointer=InMemorySaver()).invoke(
        {"kind": InputKind.MESSAGE, "learner_message": _LEARNER_MESSAGE},
        config=_CONFIG,
        context=PlannerContext(user_id=USER_ID),
    )

    assert state["tool_calls"]
    for call in state["tool_calls"]:
        schema = schemas[call.name]
        assert set(schema.get("required", [])) <= set(call.args)
        assert set(call.args) <= set(schema.get("properties", {}))
    assert state["halt_reason"] == HaltReason.FINAL_MESSAGE
    assert state["reply"].strip()
