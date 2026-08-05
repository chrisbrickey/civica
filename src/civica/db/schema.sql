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
