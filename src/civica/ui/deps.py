"""Dependencies the web app's pages use, injected so tests can supply their own."""

from dataclasses import dataclass

import psycopg
import psycopg.rows

from civica.graph.graph import build_graph
from civica.ui.turns import CoachGraph, TurnRunner
from civica.users.service import BCRYPT_ROUNDS


@dataclass(frozen=True)
class AppDeps:
    """Every seam the pages use. A None conn means users.service checks one out from the shared pool."""

    graph: CoachGraph
    turn_runner: TurnRunner
    conn: psycopg.Connection[psycopg.rows.TupleRow] | None = None
    bcrypt_rounds: int = BCRYPT_ROUNDS


def default_app_deps() -> AppDeps:
    """Production wiring. Build it once per process: it holds the chat models and the pool."""
    graph = build_graph()
    return AppDeps(graph=graph, turn_runner=TurnRunner(graph))
