# Civica MVP Overview
Civica is a learning coach whose initial focus is helping the user study for the **written civic exam required for French naturalization** ([Ministry of the Interior - Devenir français](https://www.immigration.interieur.gouv.fr/devenir-francais/procedures-dacces-a-nationalite-francaise)).
Civica is an *official-content study coach*: it teaches from material published by the French government, not from freeform LLM knowledge.
The primary reason for building Civica is to learn how to design, build, and operate a **memory-aware agent** that adapts to the user and improves over time.
Implementation language: **Python**. Orchestration framework: **LangGraph**.

## Civics Exam Background

> Exam facts below are drawn from public sources as of July 2026. Confirm with an immigration lawyer before making personal decisions based on this document.

Written civic exam for naturalization applicants, mandatory from 2026-01-01.

| Attribute            | Value                                                   |
|----------------------|---------------------------------------------------------|
| Format               | Multiple-choice (MCQ), digital, at an accredited center |
| Length               | 45 minutes                                              |
| Question count       | 40 (28 knowledge + 12 scenario/simulation)              |
| Answer options       | 4 per question, exactly one correct                     |
| Pass threshold       | ≥ 32 / 40 correct (80%)                                 |
| Assumed French level | B2                                                      |

**Theme weightings (fixed by the ministry):**

| Theme                                 | # of questions |
|---------------------------------------|----------------|
| Principes et valeurs de la République | 11             |
| Droits et devoirs                     | 11             |
| Histoire, géographie et culture       | 8              |
| Système institutionnel et politique   | 6              |
| Vivre dans la société française       | 4              |

Civica's `Quiz` and `Mock Exam` modes should mirror this structure exactly.

### How this exam differs from CSP / CR
The civics exam for naturalization appears to differ from the exams labelled as CR (carte de résident) and CSP (carte de séjour pluriannuel).

Same architecture (40 Q, 45 min, 80% pass, five themes with identical weighting), but:
- B2 French required (vs. B1 for CR, A2 for CSP): harder vocabulary and more nuanced distinctions in question wording
- Separate official question list
- Separate attestation: passing CSP/CR does not count for naturalization
- Naturalization also requires an oral interview (entretien d'assimilation), which is out of scope for civica MVP.

## MVP Scope
- Multiple Users: every store operation is namespaced by `user_id` (validated when user creates username; light passphrase check)
- UI surface: single Streamlit web app (no CLI)
- Target exam: naturalization written civic exam (not CSP, not CR)
- Content acquisition: Manually downloaded PDFs plus a one-shot capture of the ministry's web-only thematic sheets, all landing under `data/raw/`. No runtime scraping at query time.
- Explanation language: English (about French-language exam content)


## MVP Features

- **Memory-aware coaching:** remembers weak themes, recurring misconceptions, preferred explanation depth, and study cadence across sessions
- **Guided study by theme:** the five official themes
- **Quiz mode:** LLM-generated questions using the official French thematic content as source with explanations grounded in the official material*
- **Mock exam mode:** mirrors the official format (40 Q, 45-min timer, 80% pass, per-theme weighting) and reports pass/fail plus per-theme scores*

_*Use of official knowledge question list is deferred to post-MVP due to lack of official answer key._


## Architecture

Functional modules will contain domain logic that is invoked from graph nodes. This preserves the LangGraph layer as a control plane - not a bucket for business logic.
- **Official content ingester** pulls, normalizes, chunks, and embeds thematic sheets and the official question list from `data/corpus/`.
- **Progress store** persists mastery, quiz history, and next-review timestamps.
- **Memory service** wraps LangGraph store namespaces: `("learner_profile", user_id)`, `("topic_mastery", user_id)`, `("mistake_episode", user_id)`, `("session_summary", user_id)`.
- **Assessment engine** assembles quizzes and mock exams to the official proportions and theme weighting.
- **Explanation engine** generates concise, corpus-grounded explanations for teach and evaluate nodes. One mode in MVP. Tone controls are out of scope for MVP.


One graph, three terminal modes (teach, quiz, mock exam). Subgraphs are out of scope for MVP.

```
onboarding
    │
    ▼
route  ──► retrieve_content ──► teach ──► evaluate ──► update_mastery ──► save_session_summary
    │                     ▲                                                      │
    │                     └──── quiz ◄──── retrieve_question ────┐               │
    │                                                            │               │
    └──► mock_exam ─────────────────────────────────────────────►┘               │
                                                                                 │
                                            memory_writer ◄───────────────────────
```

Notes:
- `retrieve_learner_memory` is NOT a node. It is a helper the router and teach/quiz nodes call.
- `memory_writer` is a dedicated node that enforces the write allowlist before any durable long-term write.
- `route` is deterministic, not LLM-only. See adaptive routing rule section.


## Data Stores

**PostgreSQL with the `pgvector` extension.** One database for all data types, one connection pool.

- Docker Compose locally (pgvector image) accessed via `DATABASE_URL`
- LangGraph `PostgresSaver` handles short-term thread state (checkpoints).
- LangGraph `PostgresStore` handles long-term user-namespaced memory.
- `pgvector` indexes embeddings for content retrieval and semantic memory retrieval in the same database.
- Raw quiz-answer log lives in its own append-only table, **outside LangGraph state**, so checkpoints stay small.
- Single connection pool (autocommit) shared by the app.
- TTL / decay is deferred to after MVP.

Having one database dependency reduces complexity, but it requires additional attention to ensure that data types are not blended.

> Maintain strict separation of data types. Official facts never come from memory. Learner context never comes from the corpus.

## Corpus (official content)
The MVP only uses the [ministry's official thematic study materials](formation-civique.interieur.gouv.fr/fiches-par-thematiques) as source data. 
The official question list and other official documents will be incorporated post-MVP.

One-time ingestion and processing:
1. One-shot capture of the ministry's thematic webpages: The content at `formation-civique.interieur.gouv.fr/fiches-par-thematiques/` is served as a highly nested hierarchy of webpages organized by topic. Write `scripts/capture_thematic_sheets.py` to manually crawl the site and persist each page as HTML under `data/raw/`. 
2. Transform the HTML to cleaned markdown or json under `data/corpus/`
3. Chunk and embed text content (from `data/corpus/`) → store in `pgvector` for content retrieval.

Notes:
- The raw documents will be processed upfront (one time) to produce clean, deterministic inputs for ingestion that are stored in `data/corpus/`. 
- All content should be persisted in French because the exam will be in French.
- This is *not* runtime scraping. It is a deliberate offline snapshot, versioned into the corpus.
- Re-ingestion should be idempotent, meaning that re-running does not duplicate chunks/rows.
- `data/` will be git-ignored so I will include instructions in `README.md` for other developers to capture the corpus.
- Embedding model of choice: OpenAI `text-embedding-3-large` (multilingual, one-time cost, high quality)
   
Proposed corpus layout: 
- Original downloads stored in `data/raw/` for traceability.
- Ingestion-ready documents stored in `data/corpus/`. 

```
data/
    raw/                    # exactly as downloaded or otherwise captured; never edited or transformed
        reference_texts/    # ministry flyer, procedures summary
        thematic_sheets/    # raw HTML from one-shot capture
        
    corpus/                 # ingestion-ready artifacts (manually reviewed, deterministic input for ingestion) 
        thematic_sheets/    # cleaned markdown or json, grouped by slug for each french theme (matches enum values throughout)
            principes-et-valeurs-de-la-republique/
            droits-et-devoirs/
            histoire-geographie-et-culture/
            systeme-institutionnel-et-politique/
            vivre-dans-la-societe-francaise/
```

### Post-MVP incorporation of Official Question List
The [official list of questions](https://www.immigration.interieur.gouv.fr/documentation/guides-textes-et-brochures/questions-de-connaissance-pour-lexamen-civique-nationalite-francaise.html) is a pdf document of roughly 200 questions that the user is expected to answer.
However, there is no official answer key, which is why this list is not used as part of the MVP.

By incorporating the official list of questions, two major features will get major upgrade. 
**Quiz mode** and **Mock Exam mode** will draw qeustions from the official naturalization question list.

Two one-shot scripts under `scripts/`:
- `transform_pdfs_to_markdown.py` to convert pdfs in `data/raw/reference_texts/` to prose docs in `data/corpus/reference_texts/*.md`; skips the question list PDF
- `parse_questions_pdf.py` to parse the questions PDF into `data/corpus/questions.json`

Ingestion pipeline:
1. Manually download PDFs: The ministry publishes several PDFs (knowledge questions list, Livret du citoyen, Charte des droits et devoirs, official flyer).
2. Parse the knowledge questions list PDF into a structured table (question, options, correct index, theme, source citation).
3. Determine how to develop a standard for answers and how to incorporate them into the data corpus.
    - human-generated answers
    - LLM-generated answers using previously ingested corpus of the ministry's thematic material
    - online sources such as 
      - culture-civique.fr: [Questions CR](https://culture-civique.fr/questions-cr.html); This is the most comprehensive resource found. It has 200+ official questions across all 5 themes, each paired with the correct answer/explanation. It appears to be compiled from the official thematic sheets.
      - franceaccueil.com: [40 questions corrigées](https://franceaccueil.com/examen-civique-2026-40-questions-corrigees/); A sample of 40 questions in full MCQ format (A/B/C/D) with correct answers marked.
      - natification.fr: [300 questions](https://blog.natification.fr/2026/02/examen-civique/); Broad Q&A bank, but not MCQ format.

```
data/
    raw/                    
        reference_texts/    # knowledge question list, livret du citoyen, charte des droits, etc
        
    corpus/                 
        questions.json      # structured records from the official naturalization knowledge questions
        reference_texts/    # cleaned markdown and json
```

## Memory Model

Split retrieval by purpose:
- **Content retrieval:** grounded facts from the corpus (`pgvector`). Used by `teach` and `evaluate`.
- **Personalization retrieval:** learner-specific context from the LangGraph store. Used by `route`, `teach`, and `quiz`.

### Memory Layers

| Layer                           | Where it lives                              | Contents                                                       |
|---------------------------------|---------------------------------------------|----------------------------------------------------------------|
| Working                         | Thread state via `PostgresSaver`            | Current messages, active lesson plan, tool outputs, scratchpad |
| Long-term (semantic + episodic) | `PostgresStore`, namespaced per `user_id`   | four record types (detailed below)                             |
| Raw quiz log                    | Own Postgres table (not in LangGraph state) | Append-only history of every answered question                 |

Four record types for long-term records:
- **`learner_profile`:** target permit (naturalization), study cadence, preferred explanation depth, preferred correction tone; stable and rarely changes
- **`topic_mastery`:** one record per theme (5 total): score estimate, confidence, last-updated timestamp
- **`mistake_episode`:** clustered misconceptions (not raw wrong answers) - each with evidence pointers back to the raw quiz log
- **`session_summary`:** compact recap of what improved and what to revisit next

### Adaptive routing rule
The router picks the highest-priority theme not covered in the last N sessions. 
Ties are broken by presence of active `mistake_episode` records for that theme.
This MVP rule is simple, testable, and directly links memory to behavior.

```
priority(theme) = weight(theme) × (1 − mastery(theme))
```

### Memory Policies

Memory policy = what earns a memory, how confidence is tracked, and when old memory should decay or be superseded.

| Dimension     | MVP Implementation                                                           | Post-MVP Implementation                                                |
|---------------|------------------------------------------------------------------------------|------------------------------------------------------------------------|
| Write policy  | Save explicit preferences only                                               | Save inferred preferences with confidence + evidence                   |
| Retrieval     | Keyword + semantic search                                                    | Hybrid retrieval with recency, confidence, and scope ranking           |
| Consolidation | Rolling summary per thread; misconception clustering after every N questions | Separate semantic facts, episodes, and summaries as distinct pipelines |
| Safety        | Manual allowlist of memory types, enforced by `memory_writer` node           | Memory-reviewer LLM node before durable writes                         |
| Debugging     | Log retrieved memories per turn                                              | Full "why this memory was used" trace per response                     |

**Concrete rules:**
- Write `learner_profile` memory only when user intent is explicit or highly stable.
- Write `mistake_episode` memory only after assessed interactions (quiz or mock), never after casual chat.
- Consolidate raw mistakes into misconception records after every N questions; **never delete raw error rows** - they stay in the append-only quiz log.
- Retrieve at most a handful of memories per turn to avoid prompt pollution.


## Python Version

I chose python `3.12` for multiple reasons.

- Ecosystem-ready: Dependencies work well with 3.12 (e.g., LangGraph, Streamlit, psycopg, pgvector, langchain-anthropic, langchain-openai, bcrypt all publish for 3.12)
- Long support runway: EOL is October 2028
- mypy experience: Reportedly 3.12 has the best mypy experience of the widely-supported versions (e.g., generics syntax, better error messages, and mature support for pyproject.toml flags included in the MVP plan)
- Heroku support: heroku-24 stack supports 3.12 as a first-class runtime so local and prod python versions can be identical


## MVP Out of Scope

- Incorporation of the official list of questions into quizzes and mock exams
- Explanations in French (English only in MVP)
- Entretien d'assimilation, an in-person, oral interview at the prefecture that is required for naturalization
- CSP and CR variants of the civic exam
- Speech input/output
- Native mobile app
- Runtime / on-demand scraping of the ministry website; The one-shot offline capture into `data/` is in scope. Ongoing or query-time scraping is not.
- Payment, subscriptions, billing
- Memory decay / TTL
- A separate memory-reviewer LLM node before writes
- Deployed to cloud; Database: Heroku Postgres in production (accessed via `DATABASE_URL`)

