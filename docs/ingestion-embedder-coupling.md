# Ingestion: Embedder / Content-Hash Coupling

## Context

Corpus ingestion (`civica.scripts.ingest_corpus`) computes a `content_hash` for
every chunk before embedding it. The hash folds in the embedding model identity
so that swapping to a different model (even one with the same dimensions)
changes every hash, which forces re-embedding on the next run. The embedding
step skips unchanged hashes to avoid paying for vectors that already exist.

Two collaborators carry the notion of "which embedding model":

- `EMBEDDING_MODEL` (in `civica.embeddings.embedder`): the string folded into the
  `content_hash`.
- `embedder.embed`: the function that actually produces the vectors, injected
  into `ingest()` as `embed: EmbedBatchFn`.

## Concern: model and embedder can silently drift

The hash records the model named by `embedding_model`, but the vectors are
produced by the injected `embed` function. Nothing enforces that the two agree.
The `_content_hash` docstring claims the hash "encodes the embedding model that
produced the vector," yet a caller can inject an `embed` for one model while the
hash records another. If they diverge:

- The re-embed trigger becomes unreliable. Changing the actual embedder without
  changing `embedding_model` leaves every hash unchanged, so the ingestion skips
  re-embedding and the table keeps vectors from the old model.
- The corpus can end up with a mix of vectors from different models sharing one
  namespace, which corrupts nearest-neighbor retrieval (distances across models
  are not comparable).

The root cause is that a single conceptual thing (an embedding model: its
identity *and* its `embed` function) is represented as two independently
injected values.

## Proposed plan: bind identity and function into one embedder unit

Represent the embedder as one object that exposes both the model identity and
the batch-embed function, and inject that single unit:

```python
@dataclass(frozen=True)
class Embedder:
    model: str
    embed: EmbedBatchFn
```

- `embedder` exports a default instance (`model=EMBEDDING_MODEL`,
  `embed=embed`) so production wiring stays a one-liner.
- `ingest()` and `collect_pending_chunks()` take a single `embedder: Embedder`
  argument instead of the separate `embed` and `embedding_model`.
- `_content_hash` reads `embedder.model`; the batch loop calls `embedder.embed`.

This makes it structurally impossible to hash under one model while embedding
with another, so the re-embed trigger and the "one model per namespace"
invariant hold by construction rather than by convention.

### Justification / trade-offs

- **Correctness by construction.** Collapsing the two values removes the only
  way they can disagree; the invariant no longer depends on callers remembering
  to keep them in sync.
- **Consistent with the codebase direction.** Recent work has been moving
  toward explicit dependency injection for separation of concerns. This is the
  same idea applied one level up: inject the collaborator as a cohesive unit
  rather than as loose parts.
- **Cost.** It is a broader change than the parameter threading already done:
  it touches the `ingest()` signature, its call sites, and the embedder module's
  public surface. It is therefore proposed as its own story rather than folded
  into the test-cleanup refactor.

### Suggested acceptance criteria

- Ingestion accepts a single injected embedder carrying both model identity and the embed function.
- `content_hash` is computed from the same embedder object that produces the vectors; there is no separate path to supply the model string.
- Existing behavior is preserved: Default wiring uses `text-embedding-3-large`. Re-running ingestion with an unchanged corpus and embedder writes zero new rows.
