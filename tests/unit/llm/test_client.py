"""Unit tests for civica.llm.client. Builds real clients; no request is sent."""

import pytest
from langchain_anthropic import ChatAnthropic

from civica.llm.client import EXPLANATION_PROFILE, PLANNER_PROFILE, QUIZ_PROFILE, AnthropicChat

_SAMPLE_API_KEY = "sample-api-key"


@pytest.mark.parametrize(
    "profile",
    [
        pytest.param(EXPLANATION_PROFILE, id="explanation"),
        pytest.param(QUIZ_PROFILE, id="quiz"),
        pytest.param(PLANNER_PROFILE, id="planner"),
    ],
)
def test_each_profile_builds_a_model_with_a_request_timeout(
    profile: AnthropicChat, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", _SAMPLE_API_KEY)

    model = profile.build()

    assert isinstance(model, ChatAnthropic)
    assert model.default_request_timeout is not None
    assert model.default_request_timeout > 0
