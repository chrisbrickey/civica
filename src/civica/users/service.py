"""Username + local-secret authentication service."""

from uuid import UUID, uuid4

import bcrypt
import psycopg
import psycopg.rows
from pydantic import BaseModel, ConfigDict

from civica.db.session import run_on_connection
from civica.domain.user import UserId
from civica.users.rules import MAX_SECRET_BYTES, validate_secret, validate_username

# bcrypt cost factor: Library default is 12. Raise over time as hardware improves.
# Each hash consumes roughly ~0.24s as of 2026.
# Tests override this to avoid unnecessary drag on test suite performance.
BCRYPT_ROUNDS = 12

_INSERT_USER_SQL = """
INSERT INTO users (user_id, username, secret_hash)
VALUES (%s, %s, %s)
"""

_SELECT_USER_SQL = """
SELECT user_id, secret_hash FROM users WHERE username = %s
"""


class UsernameTaken(Exception):
    """Raised when registering a username that is already in use."""


class _UserRow(BaseModel):  # type: ignore[explicit-any]
    """One raw users row as returned by the verify() lookup query."""

    model_config = ConfigDict(frozen=True)

    user_id: UUID
    secret_hash: str


def normalize_username(username: str) -> str:
    """The stored form of a username, trimmed and casefolded."""
    return username.strip().casefold()


def _register_on_connection(
    username: str,
    secret: str,
    rounds: int,
    conn: psycopg.Connection[psycopg.rows.TupleRow],
) -> UserId:
    normalized_username = normalize_username(username)

    validate_username(normalized_username)
    validate_secret(secret, normalized_username)

    secret_hash = bcrypt.hashpw(
        secret.encode("utf-8"), bcrypt.gensalt(rounds)
    ).decode("utf-8")
    generated_id = uuid4()

    try:
        conn.execute(_INSERT_USER_SQL, (generated_id, normalized_username, secret_hash))
    except psycopg.errors.UniqueViolation as err:
        raise UsernameTaken(f"Username '{normalized_username}' is already taken.") from err

    return UserId(generated_id)


def register(
    username: str,
    secret: str,
    conn: psycopg.Connection[psycopg.rows.TupleRow] | None = None,
    *,
    rounds: int = BCRYPT_ROUNDS,
) -> UserId:
    """Register a new user with a username and local secret (aka password).

    Normalizes the username. Validates, encrypts, and persists the secret.
    Encryption rounds parameter is injected for override to improve test velocity.

    Raises InvalidUsername or InvalidSecret (both ValueError) when an input breaks rules.
    Raises UsernameTaken when the normalized username is already registered.
    """
    return run_on_connection(
        lambda connection: _register_on_connection(username, secret, rounds, connection),
        conn,
    )


def _verify_on_connection(
    username: str,
    secret: str,
    conn: psycopg.Connection[psycopg.rows.TupleRow],
) -> UserId | None:
    normalized_username = normalize_username(username)

    with conn.cursor(row_factory=psycopg.rows.class_row(_UserRow)) as cursor:
        cursor.execute(_SELECT_USER_SQL, (normalized_username,))
        row = cursor.fetchone()

    if row is None:
        return None

    secret_bytes = secret.encode("utf-8")
    if len(secret_bytes) > MAX_SECRET_BYTES:  # bcrypt rejects over-length input
        return None

    if bcrypt.checkpw(secret_bytes, row.secret_hash.encode("utf-8")):
        return UserId(row.user_id)

    return None


def verify(
    username: str,
    secret: str,
    conn: psycopg.Connection[psycopg.rows.TupleRow] | None = None,
) -> UserId | None:
    """Verify a username and secret, returning the UserId on success.

    Normalizes the username.
    Returns None when the username is unknown or the secret does not match
    (including over-length secrets).
    """
    return run_on_connection(
        lambda connection: _verify_on_connection(username, secret, connection), conn
    )
