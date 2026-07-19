# Civica

Civica is a learning coach whose initial focus is helping the user study for the [civics exam required for French naturalization](https://www.immigration.interieur.gouv.fr/documentation/guides-textes-et-brochures/lexamen-civique-pour-demande-de-naturalisation-ou-de-reintegration-dans-nationalite-francaise.html).
Civica includes a bespoke ingestion pipeline to ground guidance in the study materials published by the French government instead of relying on freeform LLM knowledge.

My primary reason for building Civica is to work on **memory-aware agents** that adapt to human users and improve over time.

## MVP Features (anticipated)

- **Memory-aware coaching:** remembers weak themes, recurring misconceptions, preferred explanation depth, and study cadence across sessions
- **Guided study by theme:** the five official themes
- **Quiz mode:** LLM-generated questions using the official French thematic content as source with explanations grounded in the official study material
- **Mock exam mode:** mirrors the official format (40 Q, 45-min timer, 80% pass, per-theme weighting) and reports pass/fail plus per-theme scores

_MVP uses embeddings of the official study materials in conjunction with an LLM to develop questions and answers. Use of the official `liste des questions de connaissance` (list of knowledge questions) is deferred to post-MVP due to lack of an official answer key._

### Examples

| User types                             | Civica responds with                                                                                         |
|----------------------------------------|--------------------------------------------------------------------------------------------------------------|
| "Teach me about the Republic's values" | a short, corpus-grounded lesson on *Principes et valeurs de la République* followed by a check question      |
| "Give me a quiz"                       | an LLM-generated multiple-choice question drawn from the theme most in need of review                        |
| "Start a mock exam"                    | a 40-question, 45-minute mock mirroring the official theme weighting; reports pass/fail and per-theme scores |


## Architecture

### Pipelines (anticipated)
```
                                        ┌────────┐
                                        │   UI   │
                                        └───┬────┘                                  
                                            │
                                            ▼
                             ┌──────────────────────────────┐
                             │       LangGraph (one graph)  │
   ┌────────────┐            │                              │
   │  Corpus    │───────────►│  route ──► teach             │
   │ (pgvector) │  content   │        ──► quiz              │
   └────────────┘  retrieval │        ──► mock_exam         │
                             │        ──► memory_writer     │
   ┌────────────┐            │                              │
   │  Memory    │◄──────────►│                              │
   │ (PGStore + │    user    └──────────────┬───────────────┘
   │  PGSaver)  │   history                 │        context
   └────────────┘                           ▼
                                      ┌───────────┐
                                      │    LLM    │
                                      └───────────┘
```

### Technology

| Major Dependencies   | Purpose                                        |
|----------------------|------------------------------------------------|
| uv                   | Python package + venv manager                  |
| python (3.12)        | Runtime                                        |
| docker (compose v2)  | Container runtime for local databases          |
| pgvector             | Vector similarity search extension of Postgres |
| psycopg[binary,pool] | Postgres driver + connection pool              |
| pytest               | Test suite                                     |


## Setup

### 1. Install prerequisites

- Docker Desktop (or another OCI runtime such as [OrbStack](https://orbstack.dev)): Install once per machine. Verify with `docker --version`.
- `uv` package manager: Install via `brew install uv` or `curl -LsSf https://astral.sh/uv/install.sh | sh`.

### 2. Clone the repo

```
  git clone <repo-url>
  cd civica
```

### 3. Install dependencies

```
  uv sync
```
Reads `pyproject.toml` and `uv.lock`, creates `.venv/`, and installs runtime + dev dependencies pinned to the versions in the lockfile.

### 4. Set environment variables

Copy `.env.example` to `.env` and insert your API keys. 
- `ANTHROPIC_API_KEY` (required at runtime): Go to console.anthropic.com to obtain a new key.
- `OPENAI_API_KEY` (required for one-time embedding): Go to platform.openai.com to obtain a new key.

_DATABASE_URL and TEST_DATABASE_URL variables are already present and correct for the default docker setup._

### 5. Create container and databases

```
  docker compose up -d
```
This command starts the Postgres container by reading the recipe (`docker-compose.yml`). 
It launches pgvector/pgvector:pg16 (Postgres 16 with the vector DB extension pre-installed) as a background container named `civica-postgres`.
It exposes the container at `localhost:5432`.

On first boot (if the database files are empty), it also runs the init script, which creates two new databases: `civica` (development DB) and `civica_test` (test DB).

### 6. Apply the database schema

```
  uv run python -m civica.scripts.migrate
```
- This command enables the vector extension of Postgres and executes the SQL migrations (`src/civica/db/schema.sql`) against the development database.
- Every statement in `schema.sql` should be idempotent (`CREATE ... IF NOT EXISTS`) so that this command is safe to run at any time.
  As long as the schema statements are idempotent, running the migration command will make sure that the tables exist and will create them only if they do not exist. It will not drop columns or reset state. 
- This command does not migrate the test database, which is migrated as a side effect of running the test suite.

### 7. Capture the official study corpus
`data/` is git-ignored so every developer must ingest the corpus.

```
  uv run python -m civica.scripts.capture_thematic_sheets
```
- Scrapes the ministry's [Fiches par thématiques](https://formation-civique.interieur.gouv.fr/fiches-par-thematiques/) using the site's sitemap as source of truth (~257 pages).
- Executes a second phase that searches all encountered links to catch discrepancies in the actual structure vs the sitemap (in case the sitemap is stale).
- Rate-limited to ~1 request/second. Expect the full run to take 5-10 minutes. Logs print to terminal for progress monitoring.
- Persists each page verbatim to `data/raw/thematic_sheets/<theme>/<subtheme>/<sheet>/index.html`, mirroring the URL hierarchy. 
- Idempotent: Safe to rerun any time to refresh. Pages whose bytes have not changed on the server are not rewritten. _NB: The sessionID is often captured in the HTML, which will trigger a rewrite of the captured HTML even if the french content is unchanged._


## Usage
TBD


## Development

### Run the test suite

**Run standard test suite (fast, no network calls)**
```
  uv run pytest       # runs unit, integration, and type checking tests
```
_NB: Type checking runs as one pytest test (`tests/test_typecheck.py`), so `uv run pytest` fails if `mypy --strict` reports any errors._


**Run external tests (network calls confirm API contract)**
```
  uv run pytest -m external
```


**Run type checking only**
```
  uv run mypy src
```


**Automatically clean up unused variables and imports**
```
  uv run autoflake --in-place --recursive --remove-all-unused-imports --remove-unused-variables --ignore-init-module-imports src tests
```


### Database and Container Management
The local postgres databases (development and test) are hosted via a docker container and exposed at `localhost:5432`.

**How to check state at any point**
```
  docker --version                                                  # Which docker version?
  docker compose ps                                                 # Is postgres running?
  docker compose exec postgres psql -U civica -d civica -c '\dx'    # Is vector extension installed?
  docker compose exec postgres psql -U civica -d civica -c '\dt'    # Which tables exist?
```


**Restart the container** 
```
  docker compose up -d
```
_After the databases are established locally, use this command day-to-day to restart the container if the user previously closed Docker or stopped/deleted the container.
This command reads and executes the container recipe (`docker-compose.yml`), which describes which containers should exist and how they should be setup. 
The init script (`docker/postgres/init/01-create-test-db.sql`) will not be triggered because the database files (aka data volume) are already populated - unless the user also deleted those previously._


**Stop/Delete the container**
```
  docker compose down
```
_This command does not delete the database files because Postgres data, which is also managed by Docker, is kept in a separate storage location (aka the 'named volume'). 
So the data itself survives the `docker compose down` command and will still be there the next time the user runs `docker compose up -d`._


**Wipe the database**
```
  docker compose down -v                        # -v: volumes too; Deletes the container AND the named volume (database files)
```


**Full reset of the database**
```
  docker compose down -v                         # Stop container + delete volume (wipes DB files)
  docker compose up -d                           # Fresh boot; init script re-runs to create the DBs
  uv run python -m civica.scripts.migrate        # Re-apply schema to fresh DB
```


### Project structure

```
civica/
  ├── data/                     # git-ignored; raw + normalized corpus
  ├── docker/postgres/init/     # SQL migration scripts, run on Postgres during first boot
  ├── docker-compose.yml        # local Postgres+pgvector container recipe
  │
  ├── docs/                     # design docs and plans
  │
  ├── src/civica/               # application package
  │
  └── tests/                    
        ├── integration/        # integration tests*
        └── unit/               # dependency-free tests that are confined to a single class
```
_*Any tests that make a network call (e.g., to confirm service contracts) are tagged with `@pytest.mark.external` annotation and are excluded from the default run._


### Reference Material
- [docs/](docs) for planning documentation, domain knowledge
- [French Nationality Procedures](https://www.immigration.interieur.gouv.fr/devenir-francais/procedures-dacces-a-nationalite-francaise)
