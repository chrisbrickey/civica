"""Unit tests for civica.graph.planner outside the loop: read_learner_state and the step bound.

The loop itself runs as a graph and is tested in test_planner_graph.py. Memory is an in-memory
store. No network, no database.
"""

from civica.domain.themes import SYSTEME_INSTITUTIONNEL_ET_POLITIQUE
from civica.graph.planner import MAX_PLANNER_STEPS, PROFILE_KEY, read_learner_state
from civica.memory.records import LearnerProfile
from civica.memory.writer import put
from tests.support.memory import FULL_MASTERY, HALF_MASTERY, seed_mastery
from tests.support.planner import DEFAULT_THEME, GOAL, MISTAKE, SAMPLE_KEY, PlannerFakes
from tests.support.users import USER_ID

# The longest scripted turn in test_planner_graph.py: three model replies, the last a final message.
LONGEST_SCRIPTED_TURN = 3


def test_step_bound_fits_a_teach_and_check_turn() -> None:
    assert MAX_PLANNER_STEPS >= LONGEST_SCRIPTED_TURN


class TestReadLearnerState:
    """Mastery per recorded theme, mistake episodes, and the stated goal, for one user."""

    def test_returns_recorded_mastery_mistakes_and_goal(self, planner_fakes: PlannerFakes) -> None:
        store = planner_fakes.store
        recorded = {DEFAULT_THEME: HALF_MASTERY, SYSTEME_INSTITUTIONNEL_ET_POLITIQUE: FULL_MASTERY}
        seed_mastery(store, USER_ID, recorded)
        put(USER_ID, SAMPLE_KEY, MISTAKE, store)
        put(USER_ID, PROFILE_KEY, LearnerProfile(goal=GOAL), store)

        state = read_learner_state(USER_ID, store)

        assert state.mastery == recorded
        assert state.mistakes == (MISTAKE,)
        assert state.goal == GOAL

    def test_empty_store_gives_empty_state(self, planner_fakes: PlannerFakes) -> None:
        state = read_learner_state(USER_ID, planner_fakes.store)

        assert state.mastery == {}
        assert state.mistakes == ()
        assert state.goal is None
