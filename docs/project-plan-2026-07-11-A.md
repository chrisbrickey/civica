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

- **Update this plan:** After the step ships, prefix the header with `✅` and add below notes on any diversions from the plan.
  - Added test that runs mypy as part of the standard test suite to avoid regressions. 
  - Added autoflake dependency to detect unused imports and variables. Added a test to run it as part of standard test suite.
  - Updated `src/civica/db/pool.py` to pass `open=True` explicitly to `psycopg_pool.ConnectionPool(...)` to enforce strictness, which addressed the psycopg_pool 4.x deprecation warning about the default changing to `False`.
  - Added init script (`docker/postgres/init/01-create-test-db.sql`) to create test database (`civica_test`) on first boot of Postgres database container.
  - Deferred per-test schema isolation fixture (`db_schema` in `tests/conftest.py`) to Step 4 becuase nothing built in step 1 exercises this fixture.
  - Made various improvements to setup and development instructions in `README.md`.

---

## ✅ Step 2: Capture ministry thematic sheets (`data/raw/`)

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
  > ### 7. Capture and ingest the corpus (one-time, manual)
  > Because `data/` is git-ignored, each developer must rebuild the corpus locally.
  > ```
  > uv run python -m civica.scripts.capture_thematic_sheets     # Ministry HTML → data/raw/
  > ```
  
- **Update this plan:** After the step ships, prefix the header with `✅` and add below notes on any diversions from the plan.
  - Bifurcated the test suite into `unit/` and `integration/` directories.
  - Added proper logging. Replaced print statements on the migrate script (added in Step 1).
  - Discovered that the top-level website provides a sitemap so that was incorporated into the script. Implemented a hybrid approach: 
    - The script's first pass uses the sitemap as the source of truth, but silently misses pages if the sitemap is stale.
    - The script's second pass searches all encountered links to catch new pages that may not be listed in the sitemap. A small amount of additional complexity to gain a comprehensive result.

---

## ✅ Step 3: Normalize captured HTML → `data/corpus/`

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

- **Update this plan:** After the step ships, prefix the header with `✅` and add below notes on any diversions from the plan.
  - Added an additional verification phase (`verify_corpus_coverage.py`) that cross-checks the normalized json against the raw html to ensure no critical information was dropped.
  - Transformed official themes into pydantic model for easier validation and extension (additional properties will be required for subsequent steps).
  - README: Added detail bullets about the scripts. Added civica package directories to the project structure.

---

## ✅ Step 4: Chunk + embed + ingest corpus into pgvector

**Goals:**
- Read `data/corpus/thematic_sheets/**/*.json`. 
- Chunk each section into ~800-character passages.
- Embed with OpenAI `text-embedding-3-large`.
- Upsert into a `content_chunks` table with a pgvector index. 
- Re-running is a no-op unless the source changed.
- If source pages or the chunk text format changed, re-embedding will be triggered selectively. If rows disappeared from the corpus, old chunks will be pruned to avoid accumulation of stale chunks. After pruning, `content_chunks` is an exact mirror of the current `data/corpus/`.

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

- **Tests:** 
  - Extend `tests/conftest.py` with the per-test schema fixture (see [Test fixture conventions](#test-fixture-conventions)):
    - `db_schema` fixture: function-scoped, opt-in. On setup: opens a connection outside the shared pool, creates schema `test_<uuid>`, sets `search_path` to it, calls `apply_schema()`, yields the connection. On teardown: `DROP SCHEMA ... CASCADE` and closes the connection.
    - Any test that inserts into tables requests `db_schema` and uses its yielded connection instead of `get_pool()`. This keeps the shared pool's connections at `search_path = public` so parallel/other tests do not see the transient schema.
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

- **Check In:** 
  - Stop and confirm with user that implementation is satisfactory and that end-to-end ingestion works against a real corpus sample before going forward.
  - Ask the user run the following query after running the ingestion script to confirm that the chunks are embedded as expected.
  ```
  docker compose exec postgres psql -U civica -d civica -c "
  SELECT theme,
         COUNT(*)                    AS chunks,
         COUNT(DISTINCT page_slug)   AS pages,
         MIN(vector_dims(embedding)) AS min_dims,
         MAX(vector_dims(embedding)) AS max_dims,
         ROUND(AVG(vector_norm(embedding))::numeric, 4) AS avg_norm
  FROM content_chunks
  GROUP BY ROLLUP(theme)
  ORDER BY theme NULLS LAST;"
  ```

- **Test:** extend `tests/integration/ingest/test_repository.py` (uses the `db_schema` fixture):
  - Upsert three rows, then call `delete_chunks_not_in` keeping two of the three hashes - assert exactly those two rows remain and the returned deleted-count is 1.
  - Call `delete_chunks_not_in` keeping all existing hashes - assert no rows are deleted and the returned count is 0.
  - Call `delete_chunks_not_in` with an empty keep-set - assert it raises `ValueError` and deletes nothing (a full wipe must never happen implicitly; resetting the table is a manual operation).
  
- Extend `src/civica/ingest/repository.py`:
  - `delete_chunks_not_in(keep_hashes: Collection[str], conn: psycopg.Connection | None = None) -> int` - deletes every `content_chunks` row whose `content_hash` is not in `keep_hashes` and returns the number of rows deleted. Raises `ValueError` on an empty `keep_hashes`. Same connection convention as `upsert_chunks` (pool by default, explicit connection in tests).
  
- Extend `src/civica/scripts/ingest_corpus.py`:
  - After the embed/upsert phase, call `delete_chunks_not_in` with the hashes of ALL chunks in the current corpus (not just the newly embedded ones) and log the deleted count.
  - Safety guard: if the corpus yields zero chunks (e.g. `data/corpus/` missing or empty), skip pruning and log a warning instead of emptying the table.
  - Because the prune keys on `content_hash`, it also removes obsolete rows when the chunk text format evolves (e.g. a change to the title/heading prefix), since reformatted chunks re-ingest under new hashes.

- **Check In:** Stop and confirm with user that pruning is correct: modify or remove one corpus JSON locally, re-run `ingest_corpus`, and verify the stale rows disappear while the row count matches the current corpus.
  ```
  # 1. Row count before
  docker compose exec postgres psql -U civica -d civica -c "SELECT COUNT(*) FROM content_chunks;"

  # 2. Temporarily remove one corpus page (pick any)
  mv data/corpus/thematic_sheets/droits-et-devoirs/<some-page>.json /tmp/

  # 3. Re-run ingestion: expect "Pruned N stale row(s)" and 0 new embeddings (no API cost)
  uv run python -m civica.scripts.ingest_corpus

  # 4. Row count should have dropped by exactly N; then restore and re-run
  mv /tmp/<some-page>.json data/corpus/thematic_sheets/droits-et-devoirs/
  uv run python -m civica.scripts.ingest_corpus   # re-embeds just that page's chunks, prunes 0
  ```

- **README:** 
  - Add to technology table: | `langchain-openai`     | Embeddings client (`text-embedding-3-large`) |
  - Add `ingest_corpus` to **Setup → 6**: `uv run python -m civica.scripts.ingest_corpus               # chunk + embed → pgvector`
  - Add `src/civica/ingest/` to the project structure diagram.

- **Update this plan:** After the step ships, prefix the header with `✅` and add below notes on any diversions from the plan.
  - pgvector's HNSW index caps the `vector` type at 2000 dimensions so I built the index on a `halfvec(3072)` cast which is half-precision and indexable up to 4000 dimensions. Step 5 was updated so retrieval queries use the same cast to hit the index.
  - Added `chunk_index` column so overlapping chunks within a section have a stable positional identity for citations and re-ingest diffs.
  - Added a `<page title> - <section heading>` prefix (part of the hashed text) to each chunk prior to embedding so that chunk retrieval stays anchored to the official topic. How? Adding the prefix encourages a co-location of chunks which come from the same topic in the official study materials within the vector DB. The goal is to minimize the chance that responses stray from the official material when studying a given topic. I would not have enforced this kind of categorization on a more general LLM applicationm, but it's appropriate for this narrowly focused exam prep app.
  - Used NFC-normalized UTF-8 on `content_hash` so that accent-encoding drift will not duplicate rows.
  - Nested the `db_schema` fixture within `tests/integration/conftest.py` following the logic of the unit/integration test suite split. `apply_schema()` accepts an optional connection so tests can target a per-test schema that is not visible to concurrent tests.
  
---

## ✅ Step 5: Content retriever

**Goal:** Semantic-search API used by `teach` and `evaluate` nodes.

- **Move `embedder.py` to a neutral package** (so retrieval never grows a dependency on `ingest/`):
  - Move `src/civica/ingest/embedder.py` → `src/civica/embeddings/embedder.py` (create `src/civica/embeddings/__init__.py`). No behavior change.
  - Rationale: ingestion and retrieval must share one embedder (model + dimensions) to be correct. Doing the move before `retrieval/` exists means retrieval imports the neutral package from day one instead of importing `ingest.embedder` and unwinding it later.
  - Update all imports: `db/migrate.py` (imports `EMBEDDING_DIMENSIONS`), `scripts/ingest_corpus.py`, and `tests/integration/ingest/test_embedder_external.py` (relocate to `tests/integration/embeddings/test_embedder_external.py` to mirror the source tree).
  - Verification: full suite green (`uv run pytest`), mypy strict green

- Extend `src/civica/embeddings/embedder.py`:
  - `embed_query(text: str) -> list[float]`: one-line wrapper around the client's `embed_query` so retrieval uses the same embedder as ingestion without awkward `embed([query])[0]` calls.

- **Check In:** Stop and confirm with user that implementation is satisfactory.

- **Test:** `tests/integration/retrieval/test_content.py` (integration test becuase it seeds real DB rows)
  - Seed the DB with a few chunks whose embeddings are hand-picked fake vectors, then assert `search` returns them for a stubbed query embedding, ordered by descending `similarity`.
  - Stub the query embedding by monkeypatching the embed-function reference inside `civica.retrieval.content` (no network; this must not become an `external` test).
  - Pass the fixture's yielded connection via the `conn` parameter (see below): the fixture's per-test schema is only on that private connection's `search_path`, so a pool-based read would not see the seeded rows.
  - Assert `theme` filter narrows results.
  - Assert `similarity` is `1 - cosine_distance` (seed with vectors whose expected distances are hand-computable, e.g. orthogonal unit vectors).

- Create `src/civica/retrieval/__init__.py` and `src/civica/retrieval/content.py`:
  - `search(query: str, theme: Theme | None = None, k: int = 5, conn: psycopg.Connection | None = None) -> list[ContentChunk]`: the optional-connection convention matches `upsert_chunks` and `apply_schema` (pool by default, explicit connection in tests).
  - Embeds the query with `embedder.embed_query`, runs cosine similarity via pgvector, optionally filters by `theme` (compare the `theme` column against `theme.slug`; `Theme` is a pydantic model per the Step 3 diversion, not an enum).
  - The similarity query must order by `embedding::halfvec(N) <=> <query>::halfvec(N)`: the HNSW index is built on that cast (see Step 4 notes), so a plain `embedding <=> <query>` would fall back to a sequential scan. Build `N` into the SQL from `embedder.EMBEDDING_DIMENSIONS` (same constant `migrate.py` templates into `schema.sql`); never hardcode 3072.
  - Call `register_vector(conn)` before passing the query vector as a parameter, same as `repository.py` does.
  - `<=>` returns cosine *distance*; expose `similarity = 1 - distance` so the field means what its name says.
  - `ContentChunk` is a small dataclass exposing `text`, `theme: Theme` (hydrated via `Theme.from_slug`), `page_slug`, `section_id`, `chunk_index`, `content_hash`, `similarity`. `chunk_index` and `content_hash` give citations a stable identity (the reason `chunk_index` was added in Step 4).
  - NB: Chunks were embedded with the `<page title> - <section heading>` prefix but queries are embedded raw without a standard prefix. This asymmetry is acceptable for now. 
        The order of the two elements in the prefix should be neglible in terms of embedding location.
        Subsequent steps have been updated to prefix queries as appropriate when the category is known (e.g. tutoring or quizzing within an official topic).

- **Check In:** Stop and confirm with user that implementation is satisfactory and retrieval quality on a small manual query before continuing. 
  - Instruct user to perform manual verification: Verify the HNSW index is *usable* with `EXPLAIN` after `SET enable_seqscan = off`. At MVP corpus size the planner may correctly prefer a sequential scan with seqscan enabled, so an automated index-usage assertion would be flaky by design.
  - NB: With a `theme` filter, HNSW gathers candidates first and post-filters. So filtered searches may return fewer than `k` rows on larger corpora (pgvector 0.8's iterative scans address this). Not worth engineering around at MVP size.

- **README:** Consider if anything should be added based on the changes in this section. Suggestions:
  - Add `src/civica/retrieval/` and `src/civica/embeddings/` to the project structure diagram
  - Update any text that references `ingest/embedder.py`.

- **Update this plan:** After the step ships, prefix the header with `✅` and add below notes on any diversions from the plan.
  - Similarity assertions in the integration tests use a 1e-2 tolerance (instead of exact floats) because the distances are computed through the `halfvec` cast (half precision) to hit the HNSW index. Using equivalency to an exact float would be flaky.
  - Beyond the planned cases, tests also cover the `k` limit and full `ContentChunk` field round-trip (theme hydrated as a `Theme` instance for citation identity).
  - Added smoke test to ensure database container is up before running subsequent database tests (that will all fail after wasting 30 seconds of timeout). I also DRYed a few of the existing test files that had a lot of hard-coded repetition.
  - Renamed `ingest/` to `ingestion/` to align with best practices. Modules are named for the thing or subsystem they contain (nouns). Functions are named for actions (verbs).

---

## ✅ Step 5B: Extract a validated `Chunk` domain model (refactor)

**Goals:** 
- Extract the shared core of a `Chunk` that is used by both ingestion and retreival. It represents the data persisted in the vector database.
- Enforce field invariants at the ingestion boundary to prevent inappropriate values from being persisted.
- No impact to behavior. So all existing tests must still pass at the step boundary (updated only where validation or field access changes).

**Rationale:**
- `ChunkRow` and `ContentChunk` share six persistent fields (`theme`, `page_slug`, `section_id`, `chunk_index`, `content_hash`, `text`) but each carries one field that the other must not: `ChunkRow.embedding` and `ContentChunk.similarity` which should not be shared.
- Composition (as opposed to merger) is desireable: a shared core `Chunk` pydantic model plus a thin write wrapper pydantic model (for ingestion) and a thin read wrapper pydantic model (for retrieval). All three are pydantic so validation and the frozen-model convention are uniform across the chunk types.
- Use pydantic (as opposed to dataclasses) aligning with the existing `Theme` precedent.
  - The benefit is fail-fast invariant enforcement during ingestion when parsed/derived data is persisted. On the read path (retreival) the DB schema already guarantees these values (no need for additional validation at that boundary).
  - NB: With raw psycopg, pydantic validates on construction. It sits beside the SQL mapping. It is not the DB (de)serializer.
  - Most of the new pydantic models added in this section will not be shared broadly so it is not justifiable at this time to co-locate them (for example, in a `schemas` file or directory).
- It is NOT desireable to merge these two models into one flat model because that would force `embedding` and `similarity` to be optional fields. The result would be loss of the type-level guarantee of which is populated.
- It is NOT desirable to introduce an ORM (e.g. SQLAlchemy), which would only map persisted columns. It would not own the derived `similarity` (a separate scored-result type is still required) and it would fight the hand-tuned `embedding::halfvec(N) <=> ...::halfvec(N)` cast the HNSW index depends.

**Guardrails (apply to all parts):**
- `Chunk` is corpus-side. It must never be JSON-serialized into the LangGraph memory store (the strict corpus/memory separation rule). This composition is for the retrieval/ingest paths only.
- Each new `BaseModel` (`Chunk`, `EmbeddedChunk`, `ContentChunk`) inherits the `# type: ignore[explicit-any]` mypy-strict workaround already used on `Theme`. Expected cost, not a surprise.
- Run `uv run pytest` (unit + integration + mypy strict) green at the step boundary; this is a refactor, so no behavior regresses.

**Part 1: Shared core `Chunk` domain model.** A neutral, pydantic model so ingestion and retrieval both depend on it, not on each other.

- **Test:** `tests/unit/domain/test_chunk.py` (new; unit, no DB)
  - Constructing a `Chunk` with an empty `text`, `page_slug`, `section_id`, or `content_hash`, or a negative `chunk_index`, raises `ValidationError`.
  - A valid `Chunk` is frozen: assigning to any field after construction raises.
  - `theme` accepts a `Theme` instance and round-trips unchanged (guards against a slug string sneaking in).
- Create `src/civica/domain/chunk.py`:
  - `Chunk` (pydantic `BaseModel`, `model_config = ConfigDict(frozen=True)`) with fields:
    - `theme: Theme` (the domain object, not a slug string)
    - `page_slug: str = Field(min_length=1)`
    - `section_id: str = Field(min_length=1)`
    - `chunk_index: int = Field(ge=0)`
    - `content_hash: str = Field(min_length=1)`
    - `text: str = Field(min_length=1)`
  - Imports only `domain/themes.py` (no embeddings/db imports), keeping the domain layer dependency-light.
- **Check In:** Stop and confirm the core model's fields and validation rules. Check in with the user before going forward. 

**Part 2: Write wrapper + `theme` unification (ingestion).** 

This closes the existing inconsistency where the write model typed `theme` as a slug string while the read model typed it as `Theme`. 
The result should be that both carry `Theme` and the slug appears only at the SQL boundary. No bare theme-slug strings should live on either model.

- **Test:** extend `tests/integration/ingestion/test_repository.py` (uses the `db_schema` fixture)
  - Constructing an `EmbeddedChunk` whose `embedding` length != `EMBEDDING_DIMENSIONS` raises `ValidationError` before any DB write.
  - A correctly sized `EmbeddedChunk` upserts as before; existing idempotency and `theme`-filter assertions stay green.
- Extend `src/civica/ingestion/repository.py`:
  - Replace the `ChunkRow` dataclass with `EmbeddedChunk` (pydantic `BaseModel`, `model_config = ConfigDict(frozen=True)`): `chunk: Chunk`, `embedding: list[float]`.
  - Validate the embedding dimension with a pydantic `Field` (the single most valuable guard): `embedding: list[float] = Field(min_length=EMBEDDING_DIMENSIONS, max_length=EMBEDDING_DIMENSIONS)`. Import `EMBEDDING_DIMENSIONS` from `embeddings/embedder` (the same constant `migrate.py` templates into `schema.sql`; never hardcode 3072). This catches a silent, nasty class of ingest bug before it reaches Postgres. Keep this coupling in the ingestion layer (it already imports the embedder); do not push it down into `domain/`.
  - Update `_upsert_on_connection` / `upsert_chunks` to accept `Iterable[EmbeddedChunk]` and build the SQL params from `row.chunk.theme.slug`, `row.chunk.page_slug`, `row.chunk.section_id`, `row.chunk.chunk_index`, `row.chunk.content_hash`, `row.chunk.text`, `row.embedding` (serialize the slug here, at the SQL boundary only).
- Extend `src/civica/scripts/ingest_corpus.py`:
  - Build each row as `EmbeddedChunk(chunk=Chunk(theme=Theme.from_slug(<source slug>), page_slug=..., section_id=..., chunk_index=..., content_hash=..., text=...), embedding=embedding)` instead of the flat `ChunkRow(...)`.
- **Check In:** Stop and confirm the ingestion path and the embedding-dimension guard. Check in with the user before going forward.

**Part 3: Read wrapper (retrieval).**

- **Test:** update `tests/integration/retrieval/test_content.py`
  - Adjust for the composed `ContentChunk` shape. The field round-trip still asserts `theme` is hydrated as a `Theme`; `similarity` is still `1 - cosine_distance` (existing 1e-2 tolerance from the Step 5 notes preserved).
  - Existing `theme`-filter and `k`-limit assertions stay green.
- Update `src/civica/retrieval/content.py`:
  - `ContentChunk` becomes a pydantic `BaseModel` (`model_config = ConfigDict(frozen=True)`) composing the core plus the score: `chunk: Chunk`, `similarity: float`. Keep the name `ContentChunk` to limit downstream churn (Steps 8-10 reference it).
  - Construction still hydrates `theme` via `Theme.from_slug(row[...])` (retrieval already does this) and computes `similarity = 1 - distance`.
  - Decision to confirm at the check-in: nest (`result.chunk.text`) vs. flatten (re-expose `text`, `page_slug`, ... as passthrough properties so call sites keep `result.text`). Nesting is simpler but touches every call site and the citation access in Steps 8/9/10; flattening preserves the current surface at the cost of a little boilerplate. Recommend flattening the read wrapper's most-used fields so Step 8/9/10 code stays unchanged.
- **Check In:** Stop and confirm the retrieval shape and the nest-vs-flatten decision. Check in with the user before going forward.

**Part 4: Fold embedding-model identity into `content_hash` (for safer model migration).**

This addresses a silent failure mode that I discovered while implementing this step:
- `content_hash = sha256(text)` will change if the model dimensions are changed. But it does not encode which embedding model produced the stored vector.
- If in the future I swap the `EMBEDDING_MODEL` (to another model of the same dimension), the incremental aspect of the ingestion pipeline, which skips a chunk if the hash is unchanged, would keep stale vectors from the old embedding model.
- The result would be the retreival/query pipeline performaing similarity search of stale (old model) vectors against a query embedded with the new model. 
- No errors would surface but retrieval would degrade significantly because cosine distance across two model spaces is meaningless.

- **Test:** `tests/unit/scripts/test_ingest_corpus.py` (new; unittest because no DB and no network calls)
  - Chunks collected from identical corpus text under two different `EMBEDDING_MODEL` values produce different `content_hash` values (model identity participates in the hash). Drive this through the public `collect_pending_chunks` over a tiny corpus JSON written to `tmp_path`, toggling the model between the two runs.
  - The same text under the same model is deterministic/stable across runs (idempotency preserved: unchanged corpus + unchanged model must not re-embed).
  - To keep the model toggle testable, reference the constant via the module (`ingest_corpus`/`embedder`) rather than binding a bare local at import time. So a `monkeypatch.setattr` on the constant takes effect at call time.
- Extend `src/civica/scripts/ingest_corpus.py`:
  - Import `EMBEDDING_MODEL` from `embeddings/embedder` (alongside the existing `embed` import).
  - Change `_content_hash` to hash the model identity together with the NFC-normalized text, using an unambiguous separator that cannot occur in a model name (e.g. a NUL byte `"\x00"`, not a hyphen or colon). Keep the NFC normalization so accent drift still never duplicates rows.
  - No change to `collect_pending_chunks` / `ingest` control flow: the new hash flows through the existing dedup, skip, and prune logic unchanged.
- **Check In:** Stop and confirm the hash composition (separator choice) and the one-time full-re-embed operational effect. Check in with the user before going forward.

- **README:** Consider if anything should be added based on the changes in this section.

- **Update this plan:** 
  - Consider if anything in this plan (subsequent steps) should be updated based on the changes implemented.
  - When this step is completed, prefix the header with `✅` and add below notes on any diversions from the plan.
    -  The shared `make_chunk_row` test factory takes a `Theme` domain object as argument (instead of a slug string) to avoid callers using slugs to represent themes. Slugs should only be used at the SQL layer.
    -  Improved the implementation of the read wrapper `ContentChunk` (and underlying mechanisms) to make it mirror the write wrapper `EmbeddedChunk` more closely.
      - Nested the shared chunk properties within the read wrapper ContentChunk instead of flattening it because the write wrapper uses nesting. At this point the deserialized data was accessed by position index (`row[1]`, `row[6]`), which was extremely fragile. This implementation resulted in an invisible positional contract between the `SELECT` column order and the index literals. If the query was re-ordered, the wrong data would be asigned to the `ContentChunk` fields.
      - Interim refactor: Switched the retrieval cursor to a `dict_row` row factory so that the `_search_on_connection` method read result columns by name (`row["theme"]`, `row["distance"]`) instead of by position index (`row[1]`, `row[6]`). This increased safety compared to the previous implementation but relied on untyped data structures (e.g., dict[str, Any]).
      - Added a module-private `_SearchRow` pydantic model whose fields mirror the SELECT column list deserialized from the database. This improved the `_search_on_connection` method such that it opens the database cursor with `row_factory=psycopg.rows.class_row(_SearchRow)`, which means that `cursor.fetchall()` returns a typed list[_SearchRow] instead of raw tuples or a dict of untyped values. 
      - Now the read and write paths (relevant to chunks) are much more aligned and safe because they both move data through typed pydantic models with attribute access. Additionally, the `Theme` to slug conversion sits at the SQL boundary for both the read and write processes. 
      - The remaining inherent (and acceptable) asymmetry is directional: the read path needs a hydration step, whereas the write path's mirror-image step is just serializing `theme.slug` inline when building the SQL params.

---

## ✅ Step 6: Users + auth (username + local secret)

**Goal:** A `users` table plus register/verify flows. Every subsequent memory/quiz-log call is namespaced by `user_id`.

**Design decisions (from the pre-implementation review; see the rationale bullets inline):**
- `UserId` lives in the domain layer, not the auth service. Step 7B's `quiz_answers`, the Step 7 memory layer, and later the graph all import `UserId`; anchoring it in `domain/` (beside `Chunk` and `Theme`) keeps those data layers from importing the auth service and avoids an import cycle.
- The application owns the UUID. `register` generates `uuid4()` in Python and inserts it explicitly (no DB `DEFAULT`, no `RETURNING` roundtrip). This mirrors how `content_chunks` owns its own primary key (`content_hash` generated in Python) rather than delegating identity to the database.
- `UsernameTaken` comes from the DB `UNIQUE` constraint, not a pre-check. `register` attempts the `INSERT` and catches `psycopg.errors.UniqueViolation`, translating it to `UsernameTaken`. A `SELECT`-then-`INSERT` would carry a time-of-check/time-of-use race; relying on the constraint is both correct and simpler.
- Usernames are normalized at the service boundary (trim surrounding whitespace + `casefold()`) before hashing, insert, and lookup, so `Alex`, `alex`, and `alex ` resolve to one account. The stored column stays plain `TEXT UNIQUE` (no `CITEXT`, no functional index); normalization is a service-layer concern, keeping the schema simple.
- `register` validates secret length to sidestep bcrypt's silent 72-byte truncation (two long secrets sharing a 72-byte prefix would otherwise verify as equal, and newer `bcrypt` releases raise on over-length input). Reject an over-long secret with a clear error rather than pre-hashing.
- Service functions follow the established optional-connection convention (`conn: psycopg.Connection | None = None`; pool by default, explicit connection in tests), identical to `search`, `upsert_chunks`, and `apply_schema`.


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
  - `user_id` has no `DEFAULT`: the application supplies the UUID (see design decisions). `username` stores the already-normalized value; `secret_hash` stores the bcrypt hash with its embedded salt (no separate salt column needed).

- **Test:** `tests/unit/domain/test_user.py` (new unit test)
  - `UserId` wraps a `UUID` and round-trips unchanged (a plain construction/identity check, mirroring the lightweight `Theme`/`Chunk` domain tests). This exists so the shared type has a home and a test before the service imports it.
- Create `src/civica/domain/user.py`:
  - `UserId = NewType("UserId", UUID)`. Imports only `uuid.UUID`; no db/service imports, keeping the domain layer dependency-light (same discipline as `domain/chunk.py`).

- **Check In:** Stop and confirm with user that implementation is satisfactory.

- **Test:** `tests/integration/users/test_service.py` (integration: writes real `users` rows; use the `db_schema` fixture and pass its connection)
  - Register → verify: happy path returns a `UserId` (and `verify` with the same credentials returns the same `UserId`).
  - Register duplicate username → raises `UsernameTaken`.
  - Register a username that differs only in case or surrounding whitespace from an existing one → raises `UsernameTaken` (normalization collapses them to one account).
  - `verify` succeeds when the supplied username differs only in case/whitespace from the registered form (lookup normalizes identically).
  - Verify with wrong secret → returns `None`.
  - Verify with unknown username → returns `None`.
  - Register with an over-length secret (exceeds the bcrypt 72-byte limit) → raises a clear validation error and writes no row.
- Create `src/civica/users/__init__.py` and `src/civica/users/service.py`:
  - `register(username: str, secret: str, conn: psycopg.Connection | None = None) -> UserId` - normalizes the username (trim + `casefold`), validates the secret length (raises on over-length input, before any hashing or write), hashes the secret with bcrypt, generates `uuid4()`, inserts the row, and returns the `UserId`. Catches `psycopg.errors.UniqueViolation` on the `username` constraint and raises `UsernameTaken`.
  - `verify(username: str, secret: str, conn: psycopg.Connection | None = None) -> UserId | None` - normalizes the username identically, looks up the row, checks the secret with `bcrypt.checkpw`, and returns the stored `UserId` on success or `None` on either an unknown username or a bad secret (the two cases are indistinguishable to the caller by design).
  - `UsernameTaken` is defined here (service-layer error) and exported. `UserId` is imported from `domain/user.py`.
  - Optional-connection convention (pool by default, explicit connection in tests) matches `search`/`upsert_chunks`/`apply_schema`.
  
- **Check In:** Stop and confirm with user that implementation is satisfactory and that the auth surface is what the UI will consume.

- **README:**
  - Add to technology table: | `bcrypt`               | Local-secret hashing for username auth       |
  - No new dirs in the project structure diagram (`src/civica/users/` is a sub-package).

- **Update this plan:** 
  - Consider if anything in this plan (subsequent steps) should be updated based on the changes implemented.
  - When this step is completed, prefix the header with `✅` and add below notes on any diversions from the plan. 
    - Extracted `_BCRYPT_ROUNDS` to variable and override in tests so that production hash cycles remain near libary default (12) but tests can run on fewer cycles to maintain performance.
    - Extracted shared logic for managing database sessions to one place (`session.py`).
    - Manually confirmed creation of a user in database.

---

## Step 7: Memory layer (LangGraph checkpointer + store + records + memory_writer)

**Goal:** The LangGraph-backed learner memory: short-term thread state (checkpointer), a long-term namespaced store holding four record types, and an allowlist-guarded `memory_writer` helper over that store. The raw quiz-answer log lives outside LangGraph state and is built in Step 7B.

**Conventions to follow:**
- `UserId` is imported from `domain/user.py`, never redefined here. The memory writer, the quiz log, and the record types all type their user parameter as `UserId`, keeping the namespace type single-sourced in the domain layer and this layer free of any dependency on the auth service.
- Typing (mirror `domain/`): every record/row type is a **frozen pydantic `BaseModel`** (`model_config = ConfigDict(frozen=True)`, `Field(...)` constraints), matching `Chunk`/`Theme`. Each `BaseModel` class needs a `# type: ignore[explicit-any]` comment on its `class` line, because the project's `disallow_any_explicit = true` flags pydantic's base. No plain dataclasses in this layer.
- `Theme` persists as its slug, everywhere. The `Theme` pydantic model is never stored whole. In store records, add a field serializer so `theme` dumps to `theme.slug` (rehydrate with `Theme.from_slug`); in `quiz_answers` the `theme TEXT` column holds `theme.slug`. Slug is the single wire form across the store JSON and the SQL column. (Applied in `records.py` below; the same rule governs `quiz_log.py` in Step 7B.)

**Part 1**
- Add dependency: `langgraph`, `langgraph-checkpoint-postgres` (or the equivalent monorepo package that ships `PostgresSaver` + `PostgresStore`).
  - No `schema.sql` change in this step because the LangGraph tables are created by `setup()`. The `quiz_answers` table is added in a subsequent step.
- **Test:** `tests/integration/memory/test_checkpointer.py` (integration: exercises real `PostgresSaver` tables)
  - A checkpoint saved for thread `t1` can be restored; unrelated threads return `None`.
  - Note: LangGraph's `setup()` creates its own tables outside `schema.sql` so the `db_schema` fixture does not isolate them. These tests run against the test DB's public schema and must clean up the threads they create.
- Pool config (Do this first because the existing pool is not drop-in compatible.): 
  - LangGraph's `PostgresSaver`/`PostgresStore` require their connections to use `row_factory=dict_row` and `prepare_threshold=0` (in addition to `autocommit=True`, which the pool already sets). 
  - The current `db/pool.py` hands back `TupleRow` connections. 
  - Verify the exact required kwargs against the installed `langgraph-checkpoint-postgres` version, then extend `db/pool.py` to add `row_factory=dict_row` and `prepare_threshold=0` to the shared pool's `kwargs`, and re-check existing `TupleRow` consumers (they read rows positionally via `session.run_on_connection`.
  - Confirm none break or have them set their own per-connection row factory. 
  - Keep the single-pool rule from `CLAUDE.md`: Do not spin up a second pool.
- Create `src/civica/memory/__init__.py`, `src/civica/memory/checkpointer.py`, and `src/civica/memory/store.py`:
  - `checkpointer.get_saver() -> PostgresSaver` - singleton bound to the shared connection pool (constructed directly over the pool, not via the `from_conn_string` context manager, so it stays open for the process). Calls `setup()` once.
  - `store.get_store() -> PostgresStore` - singleton bound to the shared pool. Calls `setup()` once. Key-based only (no vector index config); nothing in the current plan needs semantic memory search.
- **Check In:** Stop and confirm with user that the implemenation is satisfactory.

**Part 2**
- **Test (unit):** `tests/unit/memory/test_records.py` (pure serialization, no DB, no network)
  - `SourceRef` round-trips through `model_dump(mode="json")` / `model_validate` as an object with `page_slug`/`section_id` intact (no tuple-vs-list drift).
  - A `TopicMastery` (a Theme-bearing record) dumps its `theme` as `theme.slug` (a string, not a nested object) and rehydrates via `Theme.from_slug` to an equal record, locking in the Theme-as-slug convention.
  - Each of the four record models (`LearnerProfile`, `TopicMastery`, `MistakeEpisode`, `SessionSummary`) round-trips to an equal instance and exposes the expected `kind`.
  - Note: This is the fast test for the pydantic/typing layer.
- Create `src/civica/domain/source_ref.py` and `src/civica/memory/records.py`:
  - `domain/source_ref.py` - `SourceRef(BaseModel, frozen)` with `page_slug: str = Field(min_length=1)` and `section_id: str = Field(min_length=1)`. Lives in `domain/` (not `memory/`) because Step 9's `Question` also carries `list[SourceRef]` forward into `MistakeEpisode`. Replaces bare `(page_slug, section_id)` tuples and mirrors `Chunk`'s identity fields.
  - `records.py`:
    - `MemoryKind(StrEnum)` with members `LEARNER_PROFILE = "learner_profile"`, `TOPIC_MASTERY = "topic_mastery"`, `MISTAKE_EPISODE = "mistake_episode"`, `SESSION_SUMMARY = "session_summary"`. Single source of truth for the namespaces.
    - Four frozen pydantic record models - `LearnerProfile`, `TopicMastery`, `MistakeEpisode`, `SessionSummary` - each declaring `kind: ClassVar[MemoryKind]` so the writer derives the namespace from the record (no separate stringly-typed `namespace_kind` param, no CamelCase-to-snake drift). Each has an obvious `Theme`-typed field where relevant (persisted as slug per the Theme rule above). No hidden state.
    - `MistakeEpisode` captures the source context of the missed question, not just the theme: `theme`, `question_id: str`, and `sources: list[SourceRef]` (a question may draw on several passages). This is what lets Step 10's mistake-driven review reconstruct a corpus-prefix-style retrieval query and re-teach the exact passages the learner missed.
- **Check In:** Stop and confirm with user that the implemenation is satisfactory.

**Part 3**
- **Test (integration):** `tests/integration/memory/test_writer.py` (integration: exercises the real `PostgresStore`; same public-schema caveat as the checkpointer tests - use unique per-test `user_id`s so runs do not collide)
  - `writer.put` with a typed record succeeds and is readable back via `writer.get` as the same typed record.
  - `writer.put_raw` with a kind **not** in the allowlist raises `MemoryNotAllowed` (this is the graph/LLM-boundary entrypoint - the guard is reachable here because the caller supplies the kind as a string).
  - Reads/writes are scoped per `user_id` - a value written for `alex` is not returned for `jordan`.
  - A `MistakeEpisode` written with `sources: list[SourceRef]` reads back as an equal `MistakeEpisode` with those `SourceRef`s intact (end-to-end through the real store, complementing the unit round-trip above).
- Create `src/civica/memory/writer.py`:
  - `MEMORY_WRITE_ALLOWLIST: frozenset[MemoryKind]` = all four `MemoryKind` members.
  - `put_raw(user_id: UserId, kind: str, key: str, value: Mapping[str, object]) -> None` - the boundary entrypoint (graph/LLM path, where the kind arrives as an untrusted string). Raises `MemoryNotAllowed` if `kind` is not a member of the allowlist; otherwise `store.put((kind, str(user_id)), key, value)`. This is where the allowlist is enforced.
  - `put(user_id: UserId, key: str, record: MemoryRecord) -> None` - typed convenience wrapper for in-process callers; delegates to `put_raw(user_id, record.kind.value, key, record.model_dump(mode="json"))`. (`MemoryRecord` is the union / shared base of the four record models; its `kind` is always allowed by construction, so this path never raises.)
  - `get(user_id: UserId, key: str, kind: type[R]) -> R | None` where `R` is bound to `MemoryRecord` - reads the raw mapping and returns `kind.model_validate(mapping)` (typed, not a bare `Mapping`), or `None`. No allowlist on reads.
  - **Key conventions** (state them, downstream steps depend on them): `LearnerProfile` / `SessionSummary` - one per user, a constant key (e.g. `"current"`); `TopicMastery` - keyed by `theme.slug`; `MistakeEpisode` - unique key per episode (e.g. `question_id` + timestamp).
- **Check In:** Stop and confirm with user that the implemenation is satisfactory and that the memory surface (record shapes, allowlist) is what the graph will consume.

- **README:** 
  - Add `src/civica/memory/` to the project structure (and note `domain/source_ref.py` if the diagram lists domain files).
  - Add to technology table: | `langgraph`            | Graph orchestration, checkpointer, store     |

- **Update this plan:** 
  - Consider if anything in this plan (subsequent steps) should be updated based on the changes implemented.
  - When this step is completed, prefix the header with `✅` and add below notes on any diversions from the plan.

---

## Step 7B: Quiz-answer log (raw append-only table outside LangGraph)

**Goal:** A plain-SQL append-only record of every quiz answer, separate from LangGraph state. It powers scoring and the mistake-driven review.

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
  CREATE INDEX IF NOT EXISTS quiz_answers_user_recent_idx
    ON quiz_answers(user_id, is_correct, answered_at DESC);
  ```
  - This index matches the only reader (`recent_mistakes`: filter `user_id` + `is_correct = false`, order `answered_at DESC`). No theme-scoped index yet. Add one only when a theme-scoped reader exists.
  - `quiz_answers` is the **append-only stats/scoring log** (every answer, right or wrong). It is complementary to `MistakeEpisode` store record, which captures the corpus **source context** of wrong answers for re-teaching. Do not fold one into the other.
  - No tests. Schema only.

- **Test:** `tests/integration/progress/test_quiz_log.py` (integration: writes real `quiz_answers` rows; use the `db_schema` fixture, which also provides the `users` table for the FK)
  - **First insert a `users` row** (call `users.register`, or insert directly) - `quiz_answers.user_id` has a FK to `users(user_id)`, so logging against a non-existent user raises a `ForeignKeyViolation`. Use that row's `UserId` for the rest of the test.
  - Log 5 answers, `recent_mistakes(user_id, limit=3)` returns the 3 most recent incorrect ones in reverse-chronological order.
  - Field fidelity: a returned `QuizAnswerRow` matches what was logged - `theme` rehydrated (via `Theme.from_slug`) to the same `Theme` that was written as a slug, and `question_id` / `chosen_index` / `correct_index` / `is_correct` preserved.
  - `recent_mistakes` for a fresh user returns an empty list.
- Create `src/civica/progress/__init__.py` and `src/civica/progress/quiz_log.py`:
  - `QuizAnswerRow` - frozen pydantic model over a `quiz_answers` row: `id: int`, `user_id: UserId`, `theme: Theme`, `question_id: str`, `chosen_index: int = Field(ge=0)`, `correct_index: int = Field(ge=0)`, `is_correct: bool`, `answered_at: datetime`. Reads rehydrate `theme` via `Theme.from_slug`.
  - `log_answer(user_id: UserId, theme: Theme, question_id: str, chosen_index: int, correct_index: int, is_correct: bool, conn: psycopg.Connection | None = None) -> None` - writes `theme.slug` into the `theme` column.
  - `recent_mistakes(user_id: UserId, limit: int = 20, conn: psycopg.Connection | None = None) -> list[QuizAnswerRow]` - filters `is_correct = false`, orders `answered_at DESC` (served by `quiz_answers_user_recent_idx`).
- Notes on conventions:
  - `UserId` from `domain/user.py`, `Theme` from `domain/themes.py`. Optional-connection convention (pool by default, explicit connection in tests) matches the rest of the codebase.
  - `Theme` persists as its slug: `log_answer` writes `theme.slug` into the `theme TEXT` column; reads rehydrate via `Theme.from_slug`.  

- **Check In:** Stop and confirm with user that implementation is satisfactory and that the quiz-log surface (row shape, schema) is what the graph will consume.

- **README:** 
  - Add `src/civica/progress/` to the project structure.

- **Update this plan:** 
  - Consider if anything in this plan (subsequent steps) should be updated based on the changes implemented.
  - When this step is completed, prefix the header with `✅` and add below notes on any diversions from the plan.

---

## Step 8: Explanation engine (Claude + retrieval)

**Goal:** Given a theme and a learner question, produce a concise English explanation (with French vocabulary) grounded in retrieved French corpus chunks.

- Add dependency: `langchain-anthropic`.
- **Test:** `tests/unit/explanation/test_engine.py`
  - Unit test (no DB, no network) with a fake retriever (returns fixed chunks) and a fake `ChatAnthropic` (records the prompt, returns a canned response). Assert:
    - The final prompt contains all retrieved passages verbatim.
    - The system message forbids drawing on outside knowledge (assert the exact substring is present).
    - Returned `Explanation.citations` matches the fake retriever's output.
- Create `src/civica/explanation/engine.py` (noun-named package per the Step 5 naming convention; the public function stays the verb `explain`):
  - `explain(user_question: str, theme: Theme) -> Explanation` - calls `retrieval.content.search`, packs top-k passages into a Claude prompt with a fixed system message that mandates using only the provided passages and translating French → English, calls `ChatAnthropic(model="claude-...")`, returns an `Explanation` dataclass with `text`, `citations: list[ContentChunk]`.
  - Query construction: pass the learner's question to `search` raw, scoped by the hard `theme` filter. Do not prepend theme context to the query here: a natural user question exists, the filter already scopes the theme deterministically, and prefix tokens would dilute the question's own signal. (Contrast with Step 9, where no natural question exists and the query is manufactured from context.)
  - Prompt lives in a `PROMPTS` dict at module top so it can be tested without invoking the LLM.
- **Test:** `tests/integration/explanation/test_engine_external.py` - one `@pytest.mark.external` test that calls Claude with a tiny stub context and asserts a non-empty response (external tests live under `integration/` with the `_external` suffix, matching `test_embedder_external.py`).

- **Check In:** Stop and confirm the implementation with the user. Eyeball the explanation quality on a real theme before moving on.

- **README:**
  - Add to technology table: | `langchain-anthropic`  | Claude LLM client                            |

- **Update this plan:**
  - Consider if anything in this plan (subsequent steps) should be updated based on the changes implemented.
  - When this step is completed, prefix the header with `✅` and add below notes on any diversions from the plan.

---

## Step 9: Assessment engine (quiz + mock exam assembly)

**Goal:** Assemble quiz batches and mock exams that mirror the official structure.

- **Test:** `tests/unit/assessment/test_engine.py`
  - Unit test (no DB, no network): fake retriever + fake Claude client returning a deterministic JSON payload of MCQ items. Assert:
    - `generate_quiz` returns exactly `n` well-typed questions from the requested theme.
    - `generate_mock_exam` returns exactly 40 questions with the required per-theme counts.
    - `MockExamResult.passed` reflects the 80% threshold at boundaries (31 = fail, 32 = pass, 40 = pass).
    - The fake retriever records the query it was called with; assert the query is constructed from the theme's `display_name_fr` (never an empty string), per the query-construction rule below.
- Create `src/civica/assessment/engine.py`:
  - `generate_quiz(user_id: UserId, theme: Theme, n: int = 5) -> list[Question]` - retrieves corpus passages for the theme, prompts Claude to produce `n` MCQ questions in French (4 options, exactly one correct), returns typed `Question` records. Each `Question` carries `sources: list[(page_slug, section_id)]` taken from the retrieved `ContentChunk`s it was generated from, so a wrong answer can be logged as a `MistakeEpisode` (Step 7) with enough context for targeted re-teaching (Step 10).
  - Query construction: no natural user question exists here, so manufacture the retrieval query from context in the same style as the corpus chunk prefix (`<page title> - <section heading>`, see Step 4 notes): at minimum `theme.display_name_fr`, or `<theme display_name_fr> - <page title>` when targeting a specific page. Combined with the hard `theme` filter, this pulls the query embedding toward that topic's chunk cluster without any corpus change. Vary the targeted page/section across the `n` questions so a quiz batch draws on more than one passage cluster.
  - `generate_mock_exam(user_id: UserId) -> MockExam` - produces exactly 40 questions in the fixed theme distribution (11, 11, 8, 6, 4), also mirroring 28 knowledge + 12 scenario if separable via prompt. `MockExam` exposes `questions`, `time_limit_seconds = 45 * 60`, and a `score(answers) -> MockExamResult` method. Retrieval per theme follows the same query-construction rule as `generate_quiz`.
  - `MockExamResult` exposes `total_correct`, `passed` (>= 32/40), and `per_theme_scores: dict[Theme, int]`.

- **Check In:** Stop and confirm with user that the question quality is appropriate on a small manual run of `generate_quiz`.

- **README:** Consider if anything should be added based on the changes in this section.

- **Update this plan:** 
  - Consider if anything in this plan (subsequent steps) should be updated based on the changes implemented.
  - When this step is completed, prefix the header with `✅` and add below notes on any diversions from the plan.

---

## Step 10: Router + LangGraph graph wiring

**Goal:** Deterministic priority router + a single LangGraph that terminates in `teach`, `quiz`, or `mock_exam` and always runs `memory_writer` at the end.

- **Test:** `tests/unit/graph/test_router.py` (unit: stub `memory.writer.get` to return synthetic mastery/mistake values; the router's logic is pure once reads are stubbed)
  - Given synthetic mastery values and no mistakes, assert the router picks the highest-priority theme.
  - With ties, assert the theme with an active mistake episode wins.
- Create `src/civica/graph/router.py`:
  - `pick_next_theme(user_id: UserId) -> Theme` - implements `priority(theme) = weight(theme) × (1 − mastery(theme))`. Reads `topic_mastery` for the user via `memory.writer.get`; missing mastery treated as 0. Ties broken by presence of any `mistake_episode` for the theme (present = higher priority). Weights come from the official theme distribution.

- **Check In:** Stop and confirm with user that the implementation is satisfactory.

- **Test:** `tests/integration/graph/test_graph.py`
  - Integration test using real `explanation`/`assessment` code but fake Claude + fake retriever (no network; writes real `quiz_answers` and store rows). Run the graph through the `quiz` mode end-to-end for a test user and assert:
    - A quiz answer written through the graph appears in the raw `quiz_answers` table.
    - `topic_mastery` for the answered theme is updated (via `memory_writer_node`, respecting the allowlist).
    - The mock exam mode produces 40 questions and reports pass/fail.
- Create `src/civica/graph/nodes.py`:
  - `retrieve_content_node`, `teach_node`, `evaluate_node`, `quiz_node`, `retrieve_question_node`, `mock_exam_node`, `update_mastery_node`, `save_session_summary_node`, `memory_writer_node`.
  - Each node is a plain function `(state) -> state_update`. Business logic lives in `explanation`, `assessment`, `memory`; nodes only orchestrate.
  - Mistake-driven review: when the router selected the theme because of an active `mistake_episode`, `retrieve_content_node` reconstructs the retrieval query from the source context recorded on the episode (theme at minimum; `page_slug`/`section_id` when recorded) in the corpus prefix style, so re-teaching pulls the exact passages the learner missed rather than a generic theme sample. `ContentChunk.chunk_index`/`content_hash` (Step 5) give those citations stable identity across sessions. The needed source context flows from Step 9 (`Question.sources`) into Step 7's `MistakeEpisode.sources`.
- Create `src/civica/graph/graph.py`:
  - `build_graph() -> CompiledGraph` - compiles the graph with the shared `PostgresSaver` as checkpointer and the shared `PostgresStore` as store.
  - Terminal modes: `teach`, `quiz`, `mock_exam`. `memory_writer` is a required edge before any terminal write.

- **Check In:** Stop and confirm with user that the implementation is satisfactory and that the graph runs end-to-end from the CLI before layering the UI.

- **README:** Add `src/civica/graph/` to the project structure diagram.

- **Update this plan:**
  - Consider if anything in this plan (subsequent steps) should be updated based on the changes implemented.
  - When this step is completed, prefix the header with `✅` and add below notes on any diversions from the plan.

---

## Step 11: Streamlit UI

**Goal:** A single Streamlit app that handles login, mode selection, and the chat/quiz/mock-exam interactions by driving the compiled LangGraph.

- **Test:** `tests/integration/ui/test_chat_ui.py` (integration: register/login run through the real `users.service` against the test DB, even though the compiled graph is faked)
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
  - Fill in the **Usage → Streamlit app** section with the exact `uv run streamlit run chat_ui.py` command and a screenshot placeholder. Note that first-time users register with a username + local secret, and that usernames are case-insensitive and ignore surrounding whitespace. 
  - Add `chat_ui.py` to the project structure diagram (with inline comment: `# Streamlit entrypoint`).
  - Add to **Technology** table: | `streamlit`            | Web UI                                       |
  - Add to project structure:    ├── chat_ui.py                # streamlit entrypoint
  - Add under **Usage** section:
  > ### Streamlit app
  > ```
  > uv run streamlit run chat_ui.py
  > ```
  > Sign in with a username and local secret (created on first use), then pick a mode: **Teach**, **Quiz**, or **Mock Exam**.
  

- **Update this plan:**
  - Consider if anything in this plan (subsequent steps) should be updated based on the changes implemented.
  - When this step is completed, prefix the header with `✅` and add below notes on any diversions from the plan.

---

## Implementation Order

1. **Step 1 - Scaffolding + Postgres:** everything else needs a package layout, `uv` env, and a working DB connection
2. **Step 2 - Capture:** corpus acquisition has zero DB dependency; ship it first so a real snapshot exists for later ingestion tests
3. **Step 3 - Normalize:** depends on Step 2 for input HTML; produces the deterministic corpus consumed by ingest
4. **Step 4 - Ingest:** depends on Step 3's JSON shape and Step 1's pgvector setup
5. **Step 5 - Retriever:** depends on Step 4's `content_chunks` table
6. **Step 6 - Users + auth:** required to namespace memory in Step 7; independent of retrieval
7. **Step 7 - Memory layer:** needs Step 6's `user_id` and Step 1's pool; blocks the graph
   - **Step 7B - Quiz-answer log:** needs Step 6's `user_id` and Step 1's pool; independent of Step 7 (no LangGraph coupling), can be built in parallel
8. **Step 8 - Explanation engine:** needs Step 5 (retrieval); independent of memory
9. **Step 9 - Assessment engine:** needs Step 5 (retrieval) and Step 7B (persist quiz log); independent of the graph
10. **Step 10 - Router + graph:** needs Steps 7, 7B, 8, 9; last piece of pure back end
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
