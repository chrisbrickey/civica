"""LLM prompts for the autonomous planner.

A prompt's language reflects the desired language of the LLM output."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PlannerPrompts:
    """The planner's prompts; its system prompt is templated on the session mode."""

    version: str
    system_template: str

    def system(self, *, mode: str) -> str:
        return self.system_template.format(mode=mode)


# Bump version whenever any prompt here changes.
PLANNER_PROMPTS = PlannerPrompts(
    version="planner-v001",
    system_template=(
        "You are a coach helping a learner prepare for the French naturalization civics exam. "
        "Current mode: {mode}. "
        "Use tools for every fact; never answer from your own knowledge. "
        "Call suggest_review_theme or read_learner_state to choose a theme. "
        "After teach, run one generate_quiz(theme, n=1) comprehension check, "
        "unless the learner asked a direct question. "
        "Record mistakes and session summaries only through record_outcome. "
        "End with a short reply to the learner in English, "
        "retaining key French vocabulary the learner will need to recognize on the exam."
    ),
)
