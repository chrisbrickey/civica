"""Unit tests for civica.domain.user: UserId domain type."""

from uuid import UUID

from civica.domain.user import UserId

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SAMPLE_UUID = UUID("00000000-0000-0000-0000-000000000001")

# ---------------------------------------------------------------------------
# UserId: wraps a UUID and round-trips unchanged
# ---------------------------------------------------------------------------


def test_user_id_round_trips_uuid_instance() -> None:
    user_id = UserId(SAMPLE_UUID)
    assert user_id == SAMPLE_UUID
    assert user_id is SAMPLE_UUID
    assert isinstance(user_id, UUID)
