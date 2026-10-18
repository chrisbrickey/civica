CREATE TABLE IF NOT EXISTS content_chunks (
  content_hash TEXT PRIMARY KEY,
  theme TEXT NOT NULL,
  page_slug TEXT NOT NULL,
  section_id TEXT NOT NULL,
  chunk_index INT NOT NULL,
  text TEXT NOT NULL,
  embedding VECTOR({embedding_dim}) NOT NULL
);
CREATE INDEX IF NOT EXISTS content_chunks_theme_idx ON content_chunks(theme);
-- hnsw on the vector type is capped at 2000 dimensions; index a halfvec cast instead.
-- Queries must compare via the same cast (embedding::halfvec({embedding_dim})) to use this index.
CREATE INDEX IF NOT EXISTS content_chunks_embedding_idx
  ON content_chunks USING hnsw ((embedding::halfvec({embedding_dim})) halfvec_cosine_ops);

CREATE TABLE IF NOT EXISTS users (
  user_id UUID PRIMARY KEY,
  username TEXT UNIQUE NOT NULL,
  secret_hash TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

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


CREATE TABLE IF NOT EXISTS generated_questions (
  question_id TEXT PRIMARY KEY,
  theme TEXT NOT NULL,
  kind TEXT NOT NULL,
  text TEXT NOT NULL,
  options JSONB NOT NULL,
  correct_index INT NOT NULL,
  sources JSONB NOT NULL,
  prompt_version TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS generated_questions_theme_idx ON generated_questions(theme);

CREATE TABLE IF NOT EXISTS rejected_questions (
  id BIGSERIAL PRIMARY KEY,
  prompt_version TEXT NOT NULL,
  theme TEXT NOT NULL,
  reason TEXT NOT NULL,
  attempt INT NOT NULL,
  rejected_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS rejected_questions_prompt_version_idx
  ON rejected_questions(prompt_version);

CREATE TABLE IF NOT EXISTS planner_trace (
  id BIGSERIAL PRIMARY KEY,
  user_id UUID NOT NULL REFERENCES users(user_id),
  thread_id TEXT NOT NULL,
  turn_id UUID NOT NULL,
  step_index INT NOT NULL,
  tool_name TEXT,
  tool_args JSONB,
  halt_reason TEXT,
  prompt_version TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  -- Each turn ends in one row with a halt_reason and no tool.
  CHECK ((tool_name IS NULL) <> (halt_reason IS NULL))
);
CREATE INDEX IF NOT EXISTS planner_trace_turn_idx
  ON planner_trace(user_id, thread_id, turn_id, step_index);

CREATE TABLE IF NOT EXISTS retrieval_gaps (
  id BIGSERIAL PRIMARY KEY,
  theme TEXT NOT NULL,
  query TEXT NOT NULL,
  occurred_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS retrieval_gaps_theme_idx ON retrieval_gaps(theme);
