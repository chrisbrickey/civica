# Project Plan: Civica MVP

## Context

Civica is a memory-aware learning coach for the French naturalization written civicg exam. 
This plan covers the MVP: a single Streamlit app that offers three terminal modes (teach, quiz, mock exam), backed by 
generative AI for questions/responses, embeddings for content retrieval, and data store for the official corpus and agent memory. 
See `docs/mvp-overview.md` for full scope.

## Prerequisites (manual)

- **Docker Desktop** (or another OCI runtime) for local Postgres+pgvector.
- **`uv`** installed on the host (`brew install uv` or `curl -LsSf https://astral.sh/uv/install.sh | sh`) with interpreter Python 3.12. 
- **API keys:**
  - `ANTHROPIC_API_KEY` for text generation
  - `OPENAI_API_KEY` for `text-embedding-3-large` for one-time embedding of source material

## Test fixture conventions

- **Corpus fixtures:** two or three short, hand-crafted French thematic paragraphs stored as strings/JSON in `tests/fixtures/thematic_sheets/`. Themes should span at least two of the five official themes so ingest and retrieval can be tested cross-theme. Keep each paragraph under ~200 words.
- **HTML fixtures:** one small `<html>` file per capture-test scenario (minimal, real-shaped enough to exercise the normalizer). Store under `tests/fixtures/html/`.
- **Question fixtures:** small hand-written multiple-choice questions in French (2–3 total) with an obviously fake correct-answer index. Never copy real ministry questions into tests.
- **Learner fixtures:** synthetic `user_id`s like `test-user-alex`, `test-user-jordan`. Never use real names.
- **DB fixtures:** the test DB is a dedicated `civica_test` database (separate from `civica`), reached via `TEST_DATABASE_URL`. Tests that write to real tables use a function-scoped, opt-in `db_schema` fixture defined in `tests/conftest.py`. It creates a fresh Postgres schema per test (e.g. `test_<uuid>`), sets it as the connection's `search_path`, applies `schema.sql` into it, and drops the schema on teardown. Tests that only need a live connection (e.g. Step 1's pool tests) do not request the fixture and pay no isolation cost. The fixture itself is built in Step 4, when the first table (`content_chunks`) makes isolation necessary. External tests (marked `@pytest.mark.external`) may hit real Anthropic/OpenAI endpoints; default `pytest` skips them.

## Schema convention

The database schema is captured in a single file: `src/civica/db/schema.sql`. 
Every statement is idempotent (`CREATE EXTENSION IF NOT EXISTS`, `CREATE TABLE IF NOT EXISTS`, `CREATE INDEX IF NOT EXISTS`), so re-running is always safe.

- Whenever a step extends `schema.sql`, a developer applies the change locally by running:
  ```
  uv run python -m civica.scripts.migrate
  ```
- The migrate script and its accompanying `db.migrate.apply_schema()` function are created in Step 1. Later steps only append to `schema.sql`.

---

## ✅ Step 1: Project scaffolding + Postgres

**Goal:** Empty project → runnable `uv`-managed Python project with a local Postgres+pgvector container and a shared connection pool.

- Python package:
  - Install and pin Python 3.12 for the project:
    - `uv python install 3.12`
    - `uv python pin 3.12` (writes `.python-version` so every `uv run` in this repo uses 3.12).
  - Run `uv init --package civica --python 3.12` and adopt `src/`-layout so the package lives at `src/civica/`.
  - Create `src/civica/` with `__init__.py`.
  - Configure `pyproject.toml`:
    - `requires-python = ">=3.12,<3.13"` under `[project]` (pins the supported interpreter range).
    - `[tool.pytest.ini_options]`: `addopts = "-m 'not external'"` and register the `external` marker.
    - `[tool.mypy]`: `python_version = "3.12"`, `strict = true`, `disallow_any_explicit = true`, `disallow_untyped_defs = true`.

- Dependencies:
  - Add dependencies (`uv add`): `psycopg[binary,pool]`, `pgvector`, `python-dotenv`.
  - Add dev dependencies (`uv add --dev`): `pytest`, `pytest-cov`, `mypy`.

- Environment variables:
  - Create `.env.example` with `DATABASE_URL=`, `TEST_DATABASE_URL=`, `ANTHROPIC_API_KEY=`, `OPENAI_API_KEY=`.
  - Create `.env` by copying `.env.example` and filling in the values; e.g., `DATABASE_URL=postgresql://civica:civica@localhost:5432/civica`
  - Also set `TEST_DATABASE_URL=postgresql://civica:civica@localhost:5432/civica_test` in both `.env.example` and `.env`. Tests require a separate database to prevent corruption of development data.
  - Add `.env` to `.gitignore` with header `#Environment`.

- **Check In:** Stop and confirm with user that scaffolding and `.env` handling are satisfactory before continuing.

- Data store:
  - Create `docker-compose.yml` at repo root using `pgvector/pgvector:pg16`, exposing 5432 (mount a named volume for data persistence)
  - Initialize DB name `civica` with user `civica`.
  - **Tests:** `tests/db/test_pool.py`
    - Assert `get_pool()` returns a working pool by executing `SELECT 1` through a checked-out connection (against the local test DB).
    - Assert `enable_pgvector` is idempotent (call twice, no error).
    - Only invoke the public functions; do not import module-private state.
  - Create `src/civica/db/__init__.py` and `src/civica/db/pool.py`:
    - `get_pool() -> ConnectionPool` - reads `DATABASE_URL` from env, returns a singleton `psycopg_pool.ConnectionPool` (autocommit).
    - `enable_pgvector(conn) -> None` - `CREATE EXTENSION IF NOT EXISTS vector`.

- Migrations:
  - Create `src/civica/db/schema.sql` as the single source of truth for the database schema. It starts empty; later steps append `CREATE TABLE IF NOT EXISTS` / `CREATE INDEX IF NOT EXISTS` statements. Every statement must be idempotent so re-running is a no-op.
  - **Tests:** `tests/db/test_migrate.py`
    - Assert `apply_schema()` runs cleanly against a fresh database (extension enabled, `schema.sql` executed).
    - Assert `apply_schema()` is idempotent: calling it twice against the same database does not error and leaves the schema unchanged.
    - Only invoke public functions.
  - Create `src/civica/db/migrate.py`:
    - `apply_schema() -> None` - checks out a connection from the pool, calls `enable_pgvector`, then executes the contents of `schema.sql`.
  - Create `src/civica/scripts/migrate.py`:
    - Entrypoint: `main() -> None` (runnable via `uv run python -m civica.scripts.migrate`). Calls `db.migrate.apply_schema()` and prints a short confirmation.

- **Check In:** Stop and confirm with user that Postgres connectivity and migrations are satisfactory before continuing.

- **README:**

    - Add Technology table within the **Architecture** section: 
  >   ### Technology
  >
  > | Dependency             | Purpose                                       |
  > |------------------------|-----------------------------------------------|
  > | `uv`                   | Python package + venv manager                 |
  > | `python` (3.12)        | Runtime                                       |
  > | `docker`               | Container runtime for local Postgres+pgvector |
  > | `pgvector` (extension) | Vector similarity search in Postgres          |
  > | `psycopg[binary,pool]` | Postgres driver + connection pool             |
  > | `pytest`, `mypy`       | Tests and type checking                       |


    - Add the following instructions under **Setup** section:

    >   ### 1. Clone the repo
    >
    >   ```
    >   git clone <repo-url>
    >   cd civica
    >   ```
    >
    >   ### 2. Install dependencies
    >
    >   ```
    >   uv sync
    >   ```
    >
    >   ### 3. Set environment variables 
    >   Copy `.env.example` to `.env` and fill in `DATABASE_URL`, `ANTHROPIC_API_KEY`, and `OPENAI_API_KEY`.
    >
    >   ### 4. Start Postgres (with pgvector)
    >   ```
    >   docker compose up -d
    >   ```
    >   This launches `pgvector/pgvector:pg16` locally.
    >
    >   ### 5. Apply the database schema
    >
    >   ```
    >   uv run python -m civica.scripts.migrate
    >   ```
    >   Idempotent. This command is safe to run any time.  If the object already exists, PostgreSQL silently skips that statement. Existing rows are untouched.
    >   The word "migrate" here does not operate like Django/Rails frameworks, where it can mean  to reset to declared schema state (potential for dropping columns).
    >   But schema.sql is pure idempotent DDL. The impact is more like: "make sure these tables exist".

    - Add testing instructions under **Development** section.
  >   ### Run the test suite
  >
  >   - Unit + integration tests (fast, no network): `uv run pytest`
  >   - External tests (real API/DB calls): `uv run pytest -m external`
  >   - Type checker: `uv run mypy src`
  
    - Add the top-level `### Project Structure` to the **Development** section.
    ```
    civica/
      ├── data/                     # git-ignored; raw + normalized corpus
      ├── docs/                     # design docs and plans
      ├── docker-compose.yml        # local Postgres+pgvector
      │ 
      ├── src/civica/               # application package
      │
      └── tests/                    # unit + integration tests
    ```

- **Update this plan:** After the step ships, prefix the header with `✅` and add below notes on any divertions from the plan.
  - Added test that runs mypy as part of the standard test suite to avoid regressions. 
  - Added autoflake dependency to detect unused imports and variables. Added a test to run it as part of standard test suite.
  - Updated `src/civica/db/pool.py` to pass `open=True` explicitly to `psycopg_pool.ConnectionPool(...)` to enforce strictness, which addressed the psycopg_pool 4.x deprecation warning about the default changing to `False`.
  - Added init script (`docker/postgres/init/01-create-test-db.sql`) to create test database (`civica_test`) on first boot of Postgres database container.
  - Deferred per-test schema isolation fixture (`db_schema` in `tests/conftest.py`) to Step 4 becuase nothing built in step 1 exercises this fixture.
  - Made various improvements to setup and development instructions in `README.md`.

---

## Step 2: Capture ministry thematic sheets (`data/raw/`)

**Goal:** One-shot crawl of `formation-civique.interieur.gouv.fr/fiches-par-thematiques/` that persists every visited page as HTML under `data/raw/thematic_sheets/<slug>/index.html`.

- Add dependencies: `httpx`, `beautifulsoup4`.
- **Tests:** 
  - Unit test (`tests/scripts/test_capture_thematic_sheets.py`) with `httpx.MockTransport` (no network). Stub the index page + two child pages. Assert files are created at expected paths. Assert idempotency: running twice does not re-download unchanged pages.
  - An integration test that is tagged as `external` to confirm the contract with the real ministry site. Though the crawl script is intended to be run manually only one time, it's important to know if the ministry may have updated the official material.
- Create `src/civica/scripts/__init__.py` and `src/civica/scripts/capture_thematic_sheets.py`:
  - Entrypoint: `main() -> None` (runnable via `uv run python -m civica.scripts.capture_thematic_sheets`).
  - Start at the thematic index URL. Discover links within the `formation-civique.interieur.gouv.fr/fiches-par-thematiques/` path only.
  - Persist each HTML response verbatim to `data/raw/thematic_sheets/<url-slug>/index.html`.
  - Idempotent: if the file exists and the response body matches, skip; otherwise overwrite.
  - Rate-limit to 1 request/second and set a descriptive `User-Agent`.

- **Check In:** Stop and confirm with user that the crawler correctly discovers and stores pages before moving to normalization.

- **README:** 
  - Add to project structure diagram: `├── scripts/                  # one-shot ingestion scripts`
  - Add to **Setup** section:
  > ### 6. Capture and ingest the corpus (one-time, manual)
  > Because `data/` is git-ignored, each developer must rebuild the corpus locally.
  > ```
  > uv run python -m civica.scripts.capture_thematic_sheets     # Ministry HTML → data/raw/
  > ```
  
- **Update this plan:** After the step ships, prefix the header with `✅` and add below notes on any divertions from the plan.

---

## Step 3: Normalize captured HTML → `data/corpus/`

**Goal:** Turn each `data/raw/thematic_sheets/<slug>/index.html` into a clean JSON record (`{theme, slug, title, sections: [...]}`) at `data/corpus/thematic_sheets/<theme-slug>/<page-slug>.json`.

- Create `src/civica/domain/themes.py` with the `Theme` enum for the five official theme slugs: `principes-et-valeurs-de-la-republique`, `droits-et-devoirs`, `histoire-geographie-et-culture`, `systeme-institutionnel-et-politique`, `vivre-dans-la-societe-francaise`. This module is depended on by ingestion and assessment later.
- **Test:** `tests/scripts/test_normalize_thematic_sheets.py`
  - Feed a small fixture HTML file, assert the produced JSON matches an expected fixture JSON (byte-for-byte after re-encoding).
  - Assert unknown themes raise a clear error rather than being silently written to a wrong bucket.
- Create `src/civica/scripts/normalize_thematic_sheets.py`:
  - Entrypoint: `main() -> None`.
  - For each captured HTML file, extract the page title, section headings, and body prose using `BeautifulSoup`. Drop navigation, cookie banners, `<script>`, and repeated header/footer nodes.
  - Bucket each page under one of the five theme slugs based on URL path or explicit mapping.
  - Write deterministic JSON (stable key order, UTF-8, no BOM).

- **Check In:** Stop and confirm with user that the normalized JSON shape is what downstream ingestion expects.

- **README:** 
  - Add the `normalize_thematic_sheets` command to the **Setup → 6. Capture and ingest** flow: `uv run python -m civica.scripts.normalize_thematic_sheets   # HTML → data/corpus/`
  - Add `data/corpus/thematic_sheets/` to the project structure diagram (with inline comment).

- **Update this plan:** After the step ships, prefix the header with `✅` and add below notes on any divertions from the plan.

---

## Step 4: Chunk + embed + ingest corpus into pgvector

**Goal:** Read `data/corpus/thematic_sheets/**/*.json`, chunk each section into ~800-character passages, embed with OpenAI `text-embedding-3-large`, and upsert into a `content_chunks` table with a pgvector index. Re-running is a no-op unless the source changed.

- Add dependency: `langchain-openai`.
- Extend `src/civica/db/schema.sql` (see [Schema convention](#schema-convention); developers apply with `uv run python -m civica.scripts.migrate`):
  ```sql
  CREATE TABLE IF NOT EXISTS content_chunks (
    content_hash TEXT PRIMARY KEY,
    theme TEXT NOT NULL,
    page_slug TEXT NOT NULL,
    section_id TEXT NOT NULL,
    text TEXT NOT NULL,
    embedding VECTOR(3072) NOT NULL
  );
  CREATE INDEX IF NOT EXISTS content_chunks_theme_idx ON content_chunks(theme);
  CREATE INDEX IF NOT EXISTS content_chunks_embedding_idx
    ON content_chunks USING hnsw (embedding vector_cosine_ops);
  ```
  
- **Check In:** Stop and confirm with user that implementation is satisfactory. Prompt the user on how to manually test the structure of the database.

- **Test:** `tests/ingest/test_chunker.py`
  - Assert chunk boundaries are stable and deterministic.
  - Assert chunks respect the size/overlap invariants.
- Create `src/civica/ingest/__init__.py` and `src/civica/ingest/chunker.py`:
  - `chunker.chunk_section(text: str) -> list[str]` - deterministic char-window chunker (~800 chars, ~100 char overlap).

- **Check In:** Stop and confirm with user that implementation is satisfactory.

- Extend `tests/conftest.py` with the per-test schema fixture (see [Test fixture conventions](#test-fixture-conventions)):
  - `db_schema` fixture: function-scoped, opt-in. On setup: opens a connection outside the shared pool, creates schema `test_<uuid>`, sets `search_path` to it, calls `apply_schema()`, yields the connection. On teardown: `DROP SCHEMA ... CASCADE` and closes the connection.
  - Any test that inserts into tables requests `db_schema` and uses its yielded connection instead of `get_pool()`. This keeps the shared pool's connections at `search_path = public` so parallel/other tests do not see the transient schema.
- **Tests:** 
    - `tests/ingest/test_repository.py`
      - Uses the `db_schema` fixture.
      - Integration test with fake embeddings (fixed vectors), real Postgres schema. Insert then re-insert same content_hash - assert exactly one row exists.
      - Assert `theme` filter query works.
    - `tests/ingest/test_embedder_external.py`: one `@pytest.mark.external` test that embeds a real string and asserts the returned vector length equals 3072

- Create `src/civica/ingest/embedder.py` and `src/civica/ingest/repository.py`:
  - `embedder.embed(texts: list[str]) -> list[list[float]]` - thin wrapper around `OpenAIEmbeddings(model="text-embedding-3-large")`.
  - `repository.upsert_chunks(rows: Iterable[ChunkRow]) -> None` - `ChunkRow` is a dataclass with `theme`, `page_slug`, `section_id`, `content_hash`, `text`, `embedding`. Upserts by `content_hash` primary key so re-ingest is idempotent.

- Create `src/civica/scripts/ingest_corpus.py`:
  - Entrypoint reads all JSON under `data/corpus/thematic_sheets/`, chunks, embeds (batched), upserts.

- **Check In:** Stop and confirm with user that implementation is satisfactory and that end-to-end ingestion works against a real corpus sample before going forward.

- **README:** 
  - Add to technology table: | `langchain-openai`     | Embeddings client (`text-embedding-3-large`) |
  - Add `ingest_corpus` to **Setup → 6**: `uv run python -m civica.scripts.ingest_corpus               # chunk + embed → pgvector`
  - Add `src/civica/ingest/` to the project structure diagram.

- **Update this plan:** After the step ships, prefix the header with `✅` and add below notes on any divertions from the plan.

---

## Step 5: Content retriever

**Goal:** Semantic-search API used by `teach` and `evaluate` nodes.

- **Test:** `tests/retrieval/test_content.py`
  - Integration test: seed the DB with a few chunks whose embeddings are hand-picked fake vectors, then assert `search` returns them in the expected order for a stubbed query embedding.
  - Assert `theme` filter narrows results.
- Create `src/civica/retrieval/content.py`:
  - `search(query: str, theme: Theme | None, k: int = 5) -> list[ContentChunk]`.
  - Embeds the query with the same embedder as ingestion, runs cosine similarity via pgvector, optionally filters by `theme`.
  - `ContentChunk` is a small dataclass exposing `text`, `theme`, `page_slug`, `section_id`, `similarity`.

- **Check In:** Stop and confirm with user that implementation is satisfactory and that retrieval quality on a small manual query before continuing.

- **README:** Consider if anything should be added based on the changes in this section.

- **Update this plan:** After the step ships, prefix the header with `✅` and add below notes on any divertions from the plan.

---

## Step 6: Users + auth (username + local secret)

**Goal:** A `users` table plus register/verify flows. Every subsequent memory/quiz-log call is namespaced by `user_id`.

- Add dependency: `bcrypt`.

- Extend `src/civica/db/schema.sql` (see [Schema convention](#schema-convention)):
  ```sql
  CREATE TABLE IF NOT EXISTS users (
    user_id UUID PRIMARY KEY,
    username TEXT UNIQUE NOT NULL,
    secret_hash TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
  );
  ```

- **Test:** `tests/users/test_service.py`
  - Register → verify: happy path returns a `UserId`.
  - Register duplicate username → raises `UsernameTaken`.
  - Verify with wrong secret → returns `None`.
  - Verify with unknown username → returns `None`.
- Create `src/civica/users/__init__.py` and `src/civica/users/service.py`:
  - `register(username: str, secret: str) -> UserId` - hashes secret with bcrypt, inserts row, raises `UsernameTaken` on conflict.
  - `verify(username: str, secret: str) -> UserId | None` - returns `UserId` on success, `None` on bad credentials.
  - `UserId` is a `NewType[UUID]`.
  
- **Check In:** Stop and confirm with user that implementation is satisfactory and that the auth surface is what the UI will consume.

- **README:** 
  - In **Usage → Streamlit app**, note that first-time users register with a username + local secret. 
  - Add to technology table: | `bcrypt`               | Local-secret hashing for username auth       |
  - No new dirs in the project structure diagram (`src/civica/users/` is a sub-package).

- **Update this plan:** After the step ships, prefix the header with `✅` and add below notes on any divertions from the plan.

---

## Step 7: Memory layer (LangGraph checkpointer + store + record helpers + memory_writer + quiz log)

**Goal:** All persistent learner state in one cohesive layer: LangGraph short-term thread state, LangGraph long-term namespaced store with four record types, an allowlist-guarded `memory_writer` helper, and a raw quiz-answer append-only table outside LangGraph state.

- Add dependency: `langgraph`, `langgraph-checkpoint-postgres` (or the equivalent monorepo package that ships `PostgresSaver` + `PostgresStore`).
- Extend `src/civica/db/schema.sql` (see [Schema convention](#schema-convention)):
  ```sql
  CREATE TABLE IF NOT EXISTS quiz_answers (
    id BIGSERIAL PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES users(user_id),
    theme TEXT NOT NULL,
    question_id TEXT NOT NULL,
    chosen_index INT NOT NULL,
    correct_index INT NOT NULL,
    is_correct BOOLEAN NOT NULL,
    answered_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
  );
  CREATE INDEX IF NOT EXISTS quiz_answers_user_theme_idx ON quiz_answers(user_id, theme);
  ```

- **Test:** `tests/memory/test_checkpointer.py`
  - A checkpoint saved for thread `t1` can be restored; unrelated threads return `None`.
- Create `src/civica/memory/__init__.py`, `src/civica/memory/checkpointer.py`, and `src/civica/memory/store.py`:
  - `checkpointer.get_saver() -> PostgresSaver` - singleton bound to the shared connection pool. Calls `setup()` once.
  - `store.get_store() -> PostgresStore` - singleton bound to the shared pool. Calls `setup()` once.

- **Check In:** Stop and confirm with user that the implemenation is satisfactory.

- **Test:** `tests/memory/test_writer.py`
  - `writer.put` with an allowed kind succeeds and is readable back via `writer.get`.
  - `writer.put` with a disallowed kind raises `MemoryNotAllowed`.
  - Reads/writes are scoped per `user_id` - a value written for `alex` is not returned for `jordan`.
- Create `src/civica/memory/records.py` and `src/civica/memory/writer.py`:
  - `records.py` - typed dataclasses for the four record types: `LearnerProfile`, `TopicMastery`, `MistakeEpisode`, `SessionSummary`. Each has an obvious `Theme`-typed field where relevant. No hidden state.
  - `writer.MEMORY_WRITE_ALLOWLIST: set[str] = {"learner_profile", "topic_mastery", "mistake_episode", "session_summary"}`.
  - `writer.put(user_id: UserId, namespace_kind: str, key: str, value: Mapping[str, object]) -> None` - raises `MemoryNotAllowed` if `namespace_kind` is not in the allowlist; otherwise calls `store.put(("<kind>", str(user_id)), key, value)`.
  - `writer.get(user_id, namespace_kind, key) -> Mapping[str, object] | None` - thin read passthrough (no allowlist on reads).

- **Check In:** Stop and confirm with user that the implemenation is satisfactory.

- **Test:** `tests/progress/test_quiz_log.py`
  - Log 5 answers, `recent_mistakes(user_id, limit=3)` returns the 3 most recent incorrect ones in reverse-chronological order.
  - `recent_mistakes` for a fresh user returns an empty list.
- Create `src/civica/progress/quiz_log.py`:
  - `log_answer(user_id, theme, question_id, chosen_index, correct_index, is_correct) -> None`.
  - `recent_mistakes(user_id, limit=20) -> list[QuizAnswerRow]`.

- **Check In:** Stop and confirm with user that implementation is satisfactory and that the memory surface (record shapes, allowlist, quiz-log schema) is what the graph will consume.

- **README:** 
  - Add `src/civica/memory/` and `src/civica/progress/` to the project structure.
  - Add to technology table: | `langgraph`            | Graph orchestration, checkpointer, store     |

- **Update this plan:** After the step ships, prefix the header with `✅` and add below notes on any divertions from the plan.

---

## Step 8: Explanation engine (Claude + retrieval)

**Goal:** Given a theme and a learner question, produce a concise English explanation (with French vocabulary) grounded in retrieved French corpus chunks.

- Add dependency: `langchain-anthropic`.
- **Test:** `tests/explain/test_engine.py`
  - Unit test with a fake retriever (returns fixed chunks) and a fake `ChatAnthropic` (records the prompt, returns a canned response). Assert:
    - The final prompt contains all retrieved passages verbatim.
    - The system message forbids drawing on outside knowledge (assert the exact substring is present).
    - Returned `Explanation.citations` matches the fake retriever's output.
- Create `src/civica/explain/engine.py`:
  - `explain(user_question: str, theme: Theme) -> Explanation` - calls `retrieval.content.search`, packs top-k passages into a Claude prompt with a fixed system message that mandates using only the provided passages and translating French → English, calls `ChatAnthropic(model="claude-...")`, returns an `Explanation` dataclass with `text`, `citations: list[ContentChunk]`.
  - Prompt lives in a `PROMPTS` dict at module top so it can be tested without invoking the LLM.
- **Test:** `tests/explain/test_engine_external.py` - one `@pytest.mark.external` test that calls Claude with a tiny stub context and asserts a non-empty response.

- **Check In:** Stop and confirm the implementation with the user. Eyeball the explanation quality on a real theme before moving on.

- **README:**
  - Add to technology table: | `langchain-anthropic`  | Claude LLM client                            |

- **Update this plan:** After the step ships, prefix the header with `✅` and add below notes on any divertions from the plan.

---

## Step 9: Assessment engine (quiz + mock exam assembly)

**Goal:** Assemble quiz batches and mock exams that mirror the official structure.

- **Test:** `tests/assessment/test_engine.py`
  - Fake Claude client returns a deterministic JSON payload of MCQ items. Assert:
    - `generate_quiz` returns exactly `n` well-typed questions from the requested theme.
    - `generate_mock_exam` returns exactly 40 questions with the required per-theme counts.
    - `MockExamResult.passed` reflects the 80% threshold at boundaries (31 = fail, 32 = pass, 40 = pass).
- Create `src/civica/assessment/engine.py`:
  - `generate_quiz(user_id: UserId, theme: Theme, n: int = 5) -> list[Question]` - retrieves corpus passages for the theme, prompts Claude to produce `n` MCQ questions in French (4 options, exactly one correct), returns typed `Question` records.
  - `generate_mock_exam(user_id: UserId) -> MockExam` - produces exactly 40 questions in the fixed theme distribution (11, 11, 8, 6, 4), also mirroring 28 knowledge + 12 scenario if separable via prompt. `MockExam` exposes `questions`, `time_limit_seconds = 45 * 60`, and a `score(answers) -> MockExamResult` method.
  - `MockExamResult` exposes `total_correct`, `passed` (>= 32/40), and `per_theme_scores: dict[Theme, int]`.

- **Check In:** Stop and confirm with user that the question quality is appropriate on a small manual run of `generate_quiz`.

- **README:** Consider if anything should be added based on the changes in this section.

- **Update this plan:** After the step ships, prefix the header with `✅` and add below notes on any divertions from the plan.

---

## Step 10: Router + LangGraph graph wiring

**Goal:** Deterministic priority router + a single LangGraph that terminates in `teach`, `quiz`, or `mock_exam` and always runs `memory_writer` at the end.

- **Test:** `tests/graph/test_router.py`
  - Given synthetic mastery values and no mistakes, assert the router picks the highest-priority theme.
  - With ties, assert the theme with an active mistake episode wins.
- Create `src/civica/graph/router.py`:
  - `pick_next_theme(user_id: UserId) -> Theme` - implements `priority(theme) = weight(theme) × (1 − mastery(theme))`. Reads `topic_mastery` for the user via `memory.writer.get`; missing mastery treated as 0. Ties broken by presence of any `mistake_episode` for the theme (present = higher priority). Weights come from the official theme distribution.

- **Check In:** Stop and confirm with user that the implementation is satisfactory.

- **Test:** `tests/graph/test_graph.py`
  - Integration test using real `explain`/`assessment` code but fake Claude + fake retriever. Run the graph through the `quiz` mode end-to-end for a test user and assert:
    - A quiz answer written through the graph appears in the raw `quiz_answers` table.
    - `topic_mastery` for the answered theme is updated (via `memory_writer_node`, respecting the allowlist).
    - The mock exam mode produces 40 questions and reports pass/fail.
- Create `src/civica/graph/nodes.py`:
  - `retrieve_content_node`, `teach_node`, `evaluate_node`, `quiz_node`, `retrieve_question_node`, `mock_exam_node`, `update_mastery_node`, `save_session_summary_node`, `memory_writer_node`.
  - Each node is a plain function `(state) -> state_update`. Business logic lives in `explain`, `assessment`, `memory`; nodes only orchestrate.
- Create `src/civica/graph/graph.py`:
  - `build_graph() -> CompiledGraph` - compiles the graph with the shared `PostgresSaver` as checkpointer and the shared `PostgresStore` as store.
  - Terminal modes: `teach`, `quiz`, `mock_exam`. `memory_writer` is a required edge before any terminal write.

- **Check In:** Stop and confirm with user that the implementation is satisfactory and that the graph runs end-to-end from the CLI before layering the UI.

- **README:** Add `src/civica/graph/` to the project structure diagram.

- **Update this plan:** After the step ships, prefix the header with `✅` and add below notes on any divertions from the plan.

---

## Step 11: Streamlit UI

**Goal:** A single Streamlit app that handles login, mode selection, and the chat/quiz/mock-exam interactions by driving the compiled LangGraph.

- **Test:** `tests/ui/test_chat_ui.py`
  - Use `streamlit.testing.v1.AppTest` to render the app, simulate register + login, select **Teach**, submit a prompt (with the compiled graph replaced by a fake), and assert the response appears in the rendered output.
  - Assert an unauthenticated user cannot access the mode picker.
- Create `chat_ui.py` at the repo root:
  - Login screen: username + secret text field. Calls `users.service.register` for new usernames or `.verify` for existing ones. Stores `user_id` in `st.session_state`.
  - Mode picker: **Teach**, **Quiz**, **Mock Exam**.
  - **Teach**: free-text input → `graph.invoke({..., "mode": "teach"})` → renders the explanation + citations.
  - **Quiz**: renders the next MCQ question, captures the user's choice, submits, renders correct/incorrect + short explanation.
  - **Mock Exam**: countdown timer (client-side using `st.empty()` refresh), 40-question progression, final pass/fail + per-theme breakdown.
  - Each Streamlit "turn" invokes the graph with a `thread_id = f"{user_id}:{mode}"` so `PostgresSaver` restores state.

- **Check In:** Stop and confirm with the user that the implementation is satisfactory. Manually exercise the full loop (register → quiz a theme → mock exam) end-to-end before declaring MVP done.

- **README:** 
  - Fill in the **Usage → Streamlit app** section with the exact `uv run streamlit run chat_ui.py` command and a screenshot placeholder. 
  - Add `chat_ui.py` to the project structure diagram (with inline comment: `# Streamlit entrypoint`).
  - Add to **Technology** table: | `streamlit`            | Web UI                                       |
  - Add to project structure:    ├── chat_ui.py                # streamlit entrypoint
  - Add under **Usage** section:
  > ### Streamlit app
  > ```
  > uv run streamlit run chat_ui.py
  > ```
  > Sign in with a username and local secret (created on first use), then pick a mode: **Teach**, **Quiz**, or **Mock Exam**.
  

- **Update this plan:** After the step ships, prefix the header with `✅` and add below notes on any divertions from the plan.

---

## Implementation Order

1. **Step 1 - Scaffolding + Postgres:** everything else needs a package layout, `uv` env, and a working DB connection
2. **Step 2 - Capture:** corpus acquisition has zero DB dependency; ship it first so a real snapshot exists for later ingestion tests
3. **Step 3 - Normalize:** depends on Step 2 for input HTML; produces the deterministic corpus consumed by ingest
4. **Step 4 - Ingest:** depends on Step 3's JSON shape and Step 1's pgvector setup
5. **Step 5 - Retriever:** depends on Step 4's `content_chunks` table
6. **Step 6 - Users + auth:** required to namespace memory in Step 7; independent of retrieval
7. **Step 7 - Memory layer:** needs Step 6's `user_id` and Step 1's pool; blocks the graph
8. **Step 8 - Explanation engine:** needs Step 5 (retrieval); independent of memory
9. **Step 9 - Assessment engine:** needs Step 5 (retrieval) and Step 7 (persist quiz log); independent of the graph
10. **Step 10 - Router + graph:** needs Steps 7, 8, 9; last piece of pure back end
11. **Step 11 - Streamlit UI:** needs Steps 6 and 10; delivers the user-facing MVP

---

## Verification (end-to-end)

Run these against a fresh clone with `data/` empty:

1. `docker compose up -d`
2. `uv sync`
3. `cp .env.example .env` - fill in `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`.
4. `uv run python -m civica.scripts.migrate` - verify schema created (extension `vector` present, all tables in place).
5. `uv run python -m civica.scripts.capture_thematic_sheets` - verify `data/raw/thematic_sheets/` populates.
6. `uv run python -m civica.scripts.normalize_thematic_sheets` - verify `data/corpus/thematic_sheets/` populates.
7. `uv run python -m civica.scripts.ingest_corpus` - verify `content_chunks` count in Postgres > 0.
8. `uv run pytest` - full unit/integration suite green, mypy strict green.
9. `uv run streamlit run chat_ui.py`:
   - Register `test-alex` + secret; verify user row created.
   - **Teach → "Marianne"** returns an English explanation citing at least one French corpus passage.
   - **Quiz → Principes et valeurs**: answer 3 questions; verify `quiz_answers` rows appear and `topic_mastery` updates.
   - **Mock Exam**: verify 40 questions, 45-minute countdown, and a per-theme pass/fail breakdown at completion.

---

## Project Structure

```
civica/
  ├── chat_ui.py                # streamlit entrypoint
  ├── data/                     # git-ignored; raw + normalized corpus
  ├── docs/                     # design docs and plans
  ├── docker-compose.yml        # local Postgres+pgvector
  ├── scripts/                  # one-shot ingestion scripts
  │ 
  ├── src/civica/               # application package
  │   └── ...                   # populated as steps are implemented
  │
  └── tests/                    # unit + integration tests
```
