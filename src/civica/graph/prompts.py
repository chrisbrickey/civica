"""LLM prompts for the autonomous planner.

A prompt's language reflects the desired language of the LLM output."""

from dataclasses import dataclass

from civica.llm.style import WRITING_STYLE


@dataclass(frozen=True)
class PlannerPrompts:
    """The planner's prompts."""

    version: str
    system_template: str

    def system(self) -> str:
        return self.system_template.format()


# Bump version whenever any prompt here changes.
PLANNER_PROMPTS = PlannerPrompts(
    version="planner-v005",
    system_template=(
        "You are a coach helping a learner prepare for the French naturalization civics exam. "
        "Use tools for every fact; never answer from your own knowledge. "
        "If the learner asks for a lesson or a quiz, give them that. "
        "Call suggest_review_theme or read_learner_state to choose a theme. "
        "After teach, run one generate_quiz(theme, n=1) comprehension check, "
        "unless the learner asked a direct question. "
        "Record mistakes only through record_outcome. "
        "The learner already sees the lesson and any quiz questions, "
        "so the final message must not repeat or recap them and must not ask the check question. "
        "End with a short reply to the learner in English, "
        "retaining key French vocabulary the learner will need to recognize on the exam. "
        f"{WRITING_STYLE}"
    ),
)
