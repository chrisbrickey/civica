"""Unit tests for the typed prompt sets: version pins, template placeholders, and render methods.

No network calls. No DB spinup."""

import inspect
import string
from dataclasses import fields, replace
from typing import TYPE_CHECKING

import pytest

from civica.assessment.prompts import QUIZ_PROMPTS
from civica.explanation.prompts import EXPLANATION_PROMPTS
from civica.graph.prompts import PLANNER_PROMPTS
from tests.support.prompts import prompt_text_hash

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

# Editing a prompt means bumping its set's version and re-pinning the hash here, together.
_PINS: list[tuple["DataclassInstance", str, str]] = [
    (QUIZ_PROMPTS, "quiz-v004", "92d4cd828fe7ae437dc1eb5ee57abdd4e5549b269794e6ac0cb84495e3727702"),
    (PLANNER_PROMPTS, "planner-v001", "84385104135bff4151589a95fe18ea0a20d6ef8f0a4b0344dae0cf976ee32b1e"),
    (EXPLANATION_PROMPTS, "explanation-v001", "4c7a28ecf0f4646688feb4ff83016ede2f399b962c1cbfaf414a7da53f055da4"),
]
_PIN_IDS = ["quiz", "planner", "explanation"]

_TEXT_FIELDS = [
    (prompt_set, f.name)
    for prompt_set, *_ in _PINS
    for f in fields(prompt_set)
    if f.name != "version"
]

# Each `<name>_template` field is filled by the render method `<name>`.
_TEMPLATE_SUFFIX = "_template"
_TEMPLATE_FIELDS = [pair for pair in _TEXT_FIELDS if pair[1].endswith(_TEMPLATE_SUFFIX)]
_EXTRA_PLACEHOLDER = "sample_extra"

_SAMPLE_THEME = "sample-theme"
_SAMPLE_COUNT = 42
_SAMPLE_BODY = "sample-body"
_SAMPLE_MODE = "sample-mode"
_SAMPLE_QUESTION = "sample-question"


class TestVersionPins:
    """A prompt text change without a version bump fails here."""

    @pytest.mark.parametrize(("prompt_set", "version", "text_hash"), _PINS, ids=_PIN_IDS)
    def test_version_and_text_are_pinned_together(
        self, prompt_set: "DataclassInstance", version: str, text_hash: str
    ) -> None:
        assert (getattr(prompt_set, "version"), prompt_text_hash(prompt_set)) == (version, text_hash)

    @pytest.mark.parametrize(("prompt_set", "field_name"), _TEXT_FIELDS)
    def test_every_prompt_field_moves_the_hash(
        self, prompt_set: "DataclassInstance", field_name: str
    ) -> None:
        edited = replace(prompt_set, **{field_name: getattr(prompt_set, field_name) + " edited"})  # type: ignore[type-var]

        assert prompt_text_hash(edited) != prompt_text_hash(prompt_set)

    def test_prompt_sets_are_frozen(self) -> None:
        with pytest.raises(AttributeError):
            QUIZ_PROMPTS.version = "sample-version"  # type: ignore[misc]


def _placeholder_mismatch(prompt_set: "DataclassInstance", field_name: str) -> set[str]:
    """Names used by the template but not accepted by its render method, or the reverse."""
    template: str = getattr(prompt_set, field_name)
    placeholders = {name for _, name, _, _ in string.Formatter().parse(template) if name is not None}
    render = getattr(prompt_set, field_name.removesuffix(_TEMPLATE_SUFFIX))
    return placeholders ^ set(inspect.signature(render).parameters)


class TestTemplatePlaceholders:
    """A template's placeholders match its render method's arguments exactly."""

    def test_every_prompt_set_has_a_template(self) -> None:
        assert {id(prompt_set) for prompt_set, _ in _TEMPLATE_FIELDS} == {
            id(prompt_set) for prompt_set, *_ in _PINS
        }

    @pytest.mark.parametrize(("prompt_set", "field_name"), _TEMPLATE_FIELDS)
    def test_placeholders_match_render_arguments(
        self, prompt_set: "DataclassInstance", field_name: str
    ) -> None:
        assert _placeholder_mismatch(prompt_set, field_name) == set()

    @pytest.mark.parametrize(("prompt_set", "field_name"), _TEMPLATE_FIELDS)
    def test_unknown_placeholder_is_caught(
        self, prompt_set: "DataclassInstance", field_name: str
    ) -> None:
        template = getattr(prompt_set, field_name) + "{" + _EXTRA_PLACEHOLDER + "}"
        edited = replace(prompt_set, **{field_name: template})  # type: ignore[type-var]

        assert _placeholder_mismatch(edited, field_name) == {_EXTRA_PLACEHOLDER}


class TestRenderMethods:
    """Templated prompts are filled through typed methods."""

    def test_quiz_human_carries_its_arguments(self) -> None:
        text = QUIZ_PROMPTS.human(theme=_SAMPLE_THEME, count=_SAMPLE_COUNT, groups=_SAMPLE_BODY)

        assert all(value in text for value in (_SAMPLE_THEME, str(_SAMPLE_COUNT), _SAMPLE_BODY))
        assert "{" not in text

    def test_quiz_critic_human_carries_its_arguments(self) -> None:
        text = QUIZ_PROMPTS.critic_human(theme=_SAMPLE_THEME, count=_SAMPLE_COUNT, items=_SAMPLE_BODY)

        assert all(value in text for value in (_SAMPLE_THEME, str(_SAMPLE_COUNT), _SAMPLE_BODY))
        assert "{" not in text

    def test_planner_system_carries_the_mode(self) -> None:
        text = PLANNER_PROMPTS.system(mode=_SAMPLE_MODE)

        assert _SAMPLE_MODE in text
        assert "{" not in text

    def test_explanation_human_carries_its_arguments(self) -> None:
        text = EXPLANATION_PROMPTS.human(question=_SAMPLE_QUESTION, passages=_SAMPLE_BODY)

        assert _SAMPLE_QUESTION in text and _SAMPLE_BODY in text
        assert "{" not in text
