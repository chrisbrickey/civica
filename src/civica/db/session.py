"""Connection-acquisition dispatch shared by the database-backed services.

Every service function can follow the same optional-connection convention:
- If connection argument provided, run on that connection (e.g., tests and independent transactions).
- If no connection provided, check one out from the shared pool.
"""

from collections.abc import Callable
from typing import TypeVar

import psycopg
import psycopg.rows

from civica.db.pool import PoolProvider, get_pool

T = TypeVar("T")


def run_on_connection(
    operation: Callable[[psycopg.Connection[psycopg.rows.TupleRow]], T],
    conn: psycopg.Connection[psycopg.rows.TupleRow] | None,
    *,
    pool_provider: PoolProvider = get_pool,
) -> T:
    """Run operation on a live connection, sourcing one from the pool if needed.

    When conn is provided, operation runs on that connection directly (e.g., tests and independent transactions).
    When conn is None, a connection is checked out from the shared pool for duration of the call and returned afterward.
    """
    if conn is not None:
        return operation(conn)

    pool = pool_provider()
    with pool.connection() as pool_conn:
        return operation(pool_conn)
