"""Dependencies the web app's pages use, injected so tests can supply their own."""

from dataclasses import dataclass

import psycopg
import psycopg.rows

from civica.users.service import BCRYPT_ROUNDS


@dataclass(frozen=True)
class AppDeps:
    """Every seam the pages use. A None conn means users.service checks one out from the shared pool."""

    conn: psycopg.Connection[psycopg.rows.TupleRow] | None = None
    bcrypt_rounds: int = BCRYPT_ROUNDS


def default_app_deps() -> AppDeps:
    """Production wiring."""
    return AppDeps()
