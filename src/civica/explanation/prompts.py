"""LLM prompts for the explanation engine.

A prompt's language reflects the desired language of the LLM output."""

from dataclasses import dataclass

from civica.llm.style import WRITING_STYLE


@dataclass(frozen=True)
class WordLimits:
    """Most words the model may write for a full lesson and for a brief answer."""

    lesson: int = 150
    brief: int = 80


DEFAULT_WORD_LIMITS = WordLimits()


@dataclass(frozen=True)
class ExplanationPrompts:
    """The explanation engine's prompts; the human prompt is templated on question and passages."""

    version: str
    system_template: str
    human_template: str

    def system(self, *, word_limit: int) -> str:
        return self.system_template.format(word_limit=word_limit)

    def human(self, *, question: str, passages: str) -> str:
        return self.human_template.format(question=question, passages=passages)


# Bump version whenever any prompt here changes.
EXPLANATION_PROMPTS = ExplanationPrompts(
    version="explanation-v003",
    system_template=(
        "You are a coach helping a learner prepare for the French naturalization civics exam. "
        "Use only the provided passages. Do not draw on outside knowledge. "
        "Respond in English, retaining key French vocabulary the learner will need to recognize on the exam. "
        "Write at most {word_limit} words. Start with the key point. Use at most three bullets "
        "and no more than one heading. Add no closing summary. Do not repeat facts. "
        f"{WRITING_STYLE}"
    ),
    human_template="Learner question: {question}\n\nOfficial passages:\n{passages}",
)
