"""
Integration tests for civica.users.service.

Exercises methods against a real (isolated, per-test) Postgres
schema via the db_schema fixture. No network calls.
"""

from collections.abc import Callable

import psycopg
import psycopg.rows
import pytest

from civica.domain.user import UserId
from civica.users.service import UsernameTaken, register, verify

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_USERNAME = "test-user"
_SECRET = "sample-secret"
_WRONG_SECRET = "wrong-secret"

# Same account as _USERNAME once normalized (casefold + strip).
_USERNAME_CASE_VARIANT = "TEST-USER"
_USERNAME_WHITESPACE_VARIANT = "  test-user  "

_UNKNOWN_USERNAME = "nobody-registered"

# bcrypt truncates/rejects secrets whose UTF-8 encoding exceeds 72 bytes;
# 73 single-byte ASCII characters is one byte past that limit.
_OVER_LENGTH_SECRET = "a" * 73
_OVER_LENGTH_USERNAME = "test-user-overlength"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _count_users_with_username(
    conn: psycopg.Connection[psycopg.rows.TupleRow], username: str
) -> int:
    """Direct row count on the users table, bypassing the service layer.

    Used to prove register() wrote no row when it raised, rather than
    trusting the service's own return value.
    """
    result = conn.execute(
        "SELECT COUNT(*) FROM users WHERE username = %s", (username,)
    ).fetchone()
    assert result is not None
    return int(result[0])


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# bcrypt cost factor:
# In production this value is close to library default of 12.
# Each hash consumes roughly ~0.24s as of 2026 so we lower it in tests to avoid wasteful cycles.
_TEST_BCRYPT_ROUNDS = 4


@pytest.fixture()
def do_register(
    db_schema: psycopg.Connection[psycopg.rows.TupleRow],
) -> Callable[[str, str], UserId]:
    """Register a user against the per-test schema connection.

    Injects the lowered bcrypt cost factor to improve test velocity.
    """

    def _register(username: str, secret: str) -> UserId:
        return register(username, secret, conn=db_schema, rounds=_TEST_BCRYPT_ROUNDS)

    return _register


@pytest.fixture()
def do_verify(
    db_schema: psycopg.Connection[psycopg.rows.TupleRow],
) -> Callable[[str, str], UserId | None]:
    """Verify a user against the per-test schema connection."""

    def _verify(username: str, secret: str) -> UserId | None:
        return verify(username, secret, conn=db_schema)

    return _verify


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRegisterThenVerify:
    """Happy path: registering then verifying resolves to the same user."""

    def test_verify_returns_same_user_id_as_register(
        self,
        do_register: Callable[[str, str], UserId],
        do_verify: Callable[[str, str], UserId | None],
    ) -> None:
        registered_id = do_register(_USERNAME, _SECRET)

        verified_id = do_verify(_USERNAME, _SECRET)

        assert verified_id == registered_id


class TestDuplicateUsername:
    """Rejects an already-taken username (post-normalization)."""

    def test_exact_duplicate_username_raises_username_taken(
        self,
        do_register: Callable[[str, str], UserId],
    ) -> None:
        do_register(_USERNAME, _SECRET)

        with pytest.raises(UsernameTaken):
            do_register(_USERNAME, _SECRET)

    def test_case_variant_of_existing_username_raises_username_taken(
        self,
        do_register: Callable[[str, str], UserId],
    ) -> None:
        do_register(_USERNAME, _SECRET)

        with pytest.raises(UsernameTaken):
            do_register(_USERNAME_CASE_VARIANT, _SECRET)

    def test_whitespace_variant_of_existing_username_raises_username_taken(
        self,
        do_register: Callable[[str, str], UserId],
    ) -> None:
        do_register(_USERNAME, _SECRET)

        with pytest.raises(UsernameTaken):
            do_register(_USERNAME_WHITESPACE_VARIANT, _SECRET)


class TestVerifyNormalization:
    """Normalizes the supplied username identically to register()."""

    def test_verify_succeeds_with_case_variant_of_registered_username(
        self,
        do_register: Callable[[str, str], UserId],
        do_verify: Callable[[str, str], UserId | None],
    ) -> None:
        registered_id = do_register(_USERNAME, _SECRET)

        verified_id = do_verify(_USERNAME_CASE_VARIANT, _SECRET)

        assert verified_id == registered_id

    def test_verify_succeeds_with_whitespace_variant_of_registered_username(
        self,
        do_register: Callable[[str, str], UserId],
        do_verify: Callable[[str, str], UserId | None],
    ) -> None:
        registered_id = do_register(_USERNAME, _SECRET)

        verified_id = do_verify(_USERNAME_WHITESPACE_VARIANT, _SECRET)

        assert verified_id == registered_id


class TestVerifyFailureCases:
    """Wrong secret and unknown username are both rejected, indistinguishably."""

    def test_wrong_secret_returns_none(
        self,
        do_register: Callable[[str, str], UserId],
        do_verify: Callable[[str, str], UserId | None],
    ) -> None:
        do_register(_USERNAME, _SECRET)

        result = do_verify(_USERNAME, _WRONG_SECRET)

        assert result is None

    def test_unknown_username_returns_none(
        self,
        do_verify: Callable[[str, str], UserId | None],
    ) -> None:
        result = do_verify(_UNKNOWN_USERNAME, _SECRET)

        assert result is None


class TestOverLengthSecret:
    """Rejects a secret past bcrypt's 72-byte limit before writing."""

    def test_over_length_secret_raises_value_error_and_writes_no_row(
        self,
        do_register: Callable[[str, str], UserId],
        db_schema: psycopg.Connection[psycopg.rows.TupleRow],
    ) -> None:
        with pytest.raises(ValueError):
            do_register(_OVER_LENGTH_USERNAME, _OVER_LENGTH_SECRET)

        assert _count_users_with_username(db_schema, _OVER_LENGTH_USERNAME) == 0
