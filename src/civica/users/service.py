"""Username + local-secret authentication service."""

from uuid import UUID, uuid4

import bcrypt
import psycopg
import psycopg.rows
from pydantic import BaseModel, ConfigDict

from civica.db.session import run_on_connection
from civica.domain.user import UserId

# bcrypt silently truncates (or rejects in new releases) secrets whose UTF-8 encoding exceeds this many bytes.
# Reject over-length secrets with a clear error instead of hashing a truncated value.
_BCRYPT_MAX_SECRET_BYTES = 72

# bcrypt cost factor: Library default is 12. Raise over time as hardware improves.
# Each hash consumes roughly ~0.24s as of 2026.
# Tests override this to avoid unnecessary drag on test suite performance.
_BCRYPT_ROUNDS = 12

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


def _normalize_username(username: str) -> str:
    return username.strip().casefold()


def _register_on_connection(
    username: str,
    secret: str,
    rounds: int,
    conn: psycopg.Connection[psycopg.rows.TupleRow],
) -> UserId:
    normalized_username = _normalize_username(username)

    if len(secret.encode("utf-8")) > _BCRYPT_MAX_SECRET_BYTES:
        raise ValueError(
            f"Secret exceeds the maximum length of {_BCRYPT_MAX_SECRET_BYTES} bytes."
        )

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
    rounds: int = _BCRYPT_ROUNDS,
) -> UserId:
    """Register a new user with a username and local secret.

    Normalizes the username (trim + casefold), validates the secret length
    before any hashing or database write, hashes the secret with bcrypt, and
    inserts the row with an application-generated UUID.

    Pass in a value or rounds to override the production default
    (e.g., to improve velocity in tests).

    Raises ValueError when the secret exceeds bcrypt's 72-byte limit.
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
    normalized_username = _normalize_username(username)

    with conn.cursor(row_factory=psycopg.rows.class_row(_UserRow)) as cursor:
        cursor.execute(_SELECT_USER_SQL, (normalized_username,))
        row = cursor.fetchone()

    if row is None:
        return None

    if bcrypt.checkpw(secret.encode("utf-8"), row.secret_hash.encode("utf-8")):
        return UserId(row.user_id)

    return None


def verify(
    username: str,
    secret: str,
    conn: psycopg.Connection[psycopg.rows.TupleRow] | None = None,
) -> UserId | None:
    """Verify a username and secret, returning the UserId on success.

    Normalizes the username.
    Returns None when the username is unknown or the secret does not match.
    """
    return run_on_connection(
        lambda connection: _verify_on_connection(username, secret, connection), conn
    )
