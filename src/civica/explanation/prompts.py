"""LLM prompts for the explanation engine.

A prompt's language reflects the desired language of the LLM output."""

from dataclasses import dataclass

from civica.llm.style import WRITING_STYLE


@dataclass(frozen=True)
class ExplanationPrompts:
    """The explanation engine's prompts; the human prompt is templated on question and passages."""

    version: str
    system: str
    human_template: str

    def human(self, *, question: str, passages: str) -> str:
        return self.human_template.format(question=question, passages=passages)


# Bump version whenever any prompt here changes.
EXPLANATION_PROMPTS = ExplanationPrompts(
    version="explanation-v002",
    system=(
        "You are a coach helping a learner prepare for the French naturalization civics exam. "
        "Use only the provided passages. Do not draw on outside knowledge. "
        "Respond in English, retaining key French vocabulary the learner will need to recognize on the exam. "
        f"{WRITING_STYLE}"
    ),
    human_template="Learner question: {question}\n\nOfficial passages:\n{passages}",
)
