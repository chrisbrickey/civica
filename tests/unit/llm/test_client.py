"""Unit tests for civica.llm.client. Builds real clients; no request is sent."""

import pytest
from langchain_anthropic import ChatAnthropic

from civica.llm.client import CRITIC_PROFILE, EXPLANATION_PROFILE, PLANNER_PROFILE, QUIZ_PROFILE, AnthropicChat
from tests.support.llm import CRITIC_TAG, EXPLANATION_TAG, PLANNER_TAG, QUIZ_TAG

_SAMPLE_API_KEY = "sample-api-key"


@pytest.mark.parametrize(
    "profile",
    [
        pytest.param(EXPLANATION_PROFILE, id="explanation"),
        pytest.param(QUIZ_PROFILE, id="quiz"),
        pytest.param(PLANNER_PROFILE, id="planner"),
        pytest.param(CRITIC_PROFILE, id="critic"),
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


@pytest.mark.parametrize(
    ("profile", "caller"),
    [
        pytest.param(EXPLANATION_PROFILE, EXPLANATION_TAG, id="explanation"),
        pytest.param(QUIZ_PROFILE, QUIZ_TAG, id="quiz"),
        pytest.param(PLANNER_PROFILE, PLANNER_TAG, id="planner"),
        pytest.param(CRITIC_PROFILE, CRITIC_TAG, id="critic"),
    ],
)
def test_each_profile_builds_a_model_tagged_with_its_caller(
    profile: AnthropicChat, caller: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", _SAMPLE_API_KEY)

    model = profile.build()

    assert model.tags == [caller]


def test_critic_profile_uses_the_quiz_model_and_cap() -> None:
    """Splitting the critic's tokens out must not change how its model is called."""
    assert CRITIC_PROFILE.model == QUIZ_PROFILE.model
    assert CRITIC_PROFILE.max_tokens == QUIZ_PROFILE.max_tokens


# Largest single planner call in the Step 16 token measurement (2026-11-21), reasoning included.
_LARGEST_MEASURED_PLANNER_OUTPUT = 770


def test_planner_cap_leaves_twice_the_largest_measured_call(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", _SAMPLE_API_KEY)

    model = PLANNER_PROFILE.build()

    assert isinstance(model, ChatAnthropic)
    assert model.max_tokens >= 2 * _LARGEST_MEASURED_PLANNER_OUTPUT
