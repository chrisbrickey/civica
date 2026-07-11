# Civica

Civica is a learning coach whose initial focus is helping the user study for the [civics exam required for French naturalization](https://www.immigration.interieur.gouv.fr/documentation/guides-textes-et-brochures/lexamen-civique-pour-demande-de-naturalisation-ou-de-reintegration-dans-nationalite-francaise.html).
My primary reason for building Civica is to work on **memory-aware agents** that adapt to human users and that improve over time.
Civica includes bespoke ingestion pipeline to ground guidance in the study materials published by the French government instead of relying on freeform LLM knowledge.

## MVP Features (anticipated)

- **Memory-aware coaching:** remembers weak themes, recurring misconceptions, preferred explanation depth, and study cadence across sessions
- **Guided study by theme:** the five official themes
- **Quiz mode:** LLM-generated questions using the official French thematic content as source with explanations grounded in the official study material
- **Mock exam mode:** mirrors the official format (40 Q, 45-min timer, 80% pass, per-theme weighting) and reports pass/fail plus per-theme scores

_MVP uses embeddings of the official study materials in conjunction with an LLM to develop questions and answers. Use of the official `liste des questions de connaissance` (list of knowledge question) is deferred to post-MVP due to lack of an official answer key for those questions._

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

## Setup
TBD


## Usage
TBD


## Development
TBD

### Reference Material
- [docs/](docs) for planning documentation, domain knowledge
- [French Nationality Procedures](https://www.immigration.interieur.gouv.fr/devenir-francais/procedures-dacces-a-nationalite-francaise)
