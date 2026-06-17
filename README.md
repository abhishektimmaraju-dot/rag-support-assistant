# MyTelecom — RAG Customer-Care Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that resolves common telecom
support queries (connectivity, data, roaming, SIM, billing, voice, device,
account) by grounding every answer in curated knowledge — never inventing
policy or pricing.

Built per [PRD.md](PRD.md).

## How it works

```
User question
     │
     ▼
Router (router.py) ── classify ──► pricing | telecom | other
     │
     ├─ other   → deterministic refusal (no LLM call)
     ├─ pricing → plan_facts pricing reference + plans-only context
     └─ telecom → hybrid retrieval (dense + BM25) → cross-encoder re-rank
     │
     ▼
ChatPromptTemplate (context-only persona)
     ▼
Qwen3-32B on Groq (temperature=0)
     ▼
Streamed answer → Streamlit / CLI
```

- **Embeddings:** `all-MiniLM-L6-v2`, run locally (no embedding API cost).
- **Vector store:** ChromaDB, persisted to `chroma_store/`; BM25 keyword index in memory.
- **Re-ranker:** `ms-marco-MiniLM-L-6-v2` cross-encoder (local).
- **LLM:** `qwen/qwen3-32b` via Groq.
- **Framework:** LangChain. **UI:** Streamlit + CLI.

See [CODEBASE_EXPLANATION.md](CODEBASE_EXPLANATION.md) for the full architecture.

## Setup

1. **Install dependencies** (Python 3.11 or 3.12):

   ```bash
   uv sync
   # or:  pip install -e .
   ```

2. **Add your Groq API key** (free tier at https://console.groq.com):

   ```bash
   cp .env.example .env
   # edit .env and set GROQ_API_KEY=...
   ```

3. **Build the vector store** (downloads the ~90 MB embedding model on first run):

   ```bash
   uv run ingest_all.py
   # or individually: ingest_faq.py / ingest_tickets.py / ingest_guides.py
   ```

   This persists to `chroma_store/`. The app does **not** re-ingest on start —
   re-run an ingest script only when the underlying data changes.

## Run

**Web UI:**

```bash
uv run streamlit run app.py
```

**CLI:**

```bash
uv run main.py        # type 'quit' to exit
```

## Updating knowledge

| To update… | Edit… | Then run… |
|---|---|---|
| FAQ answers | `data/faq.csv` | `ingest_faq.py` |
| Resolved tickets | `data/tickets.db` (`tickets` table) | `ingest_tickets.py` |
| Guide content | `data/telecom_guide.pdf` | `ingest_guides.py` |

Ingest scripts are idempotent — each resets and rebuilds its own collection, so
re-running never creates duplicates.

## Extending (new knowledge source)

1. Write `ingest_<name>.py` that builds a new Chroma collection (copy an
   existing ingest script).
2. Register `(LABEL, COLLECTION_NAME, TOP_K)` in `COLLECTIONS` in
   [retriever.py](retriever.py).

## Project layout

| File | Responsibility |
|---|---|
| `config.py` | Paths, model names, constants, `.env` loading |
| `embeddings.py` | Local HuggingFace embedding singleton |
| `ingest_faq.py` / `ingest_tickets.py` / `ingest_guides.py` / `ingest_plans.py` | Per-source ingestion |
| `ingest_all.py` | Build all collections in one pass |
| `router.py` | Classifies each question: pricing / telecom / out-of-scope |
| `retriever.py` | Per-collection (or subset) retrieval + context formatting |
| `hybrid.py` | Telecom path: dense + BM25 recall, then re-rank |
| `rerank.py` | Cross-encoder re-ranking for precision |
| `plan_facts.py` | Deterministic pricing reference computed from `plans.json` |
| `chain.py` | Routes the query, builds context, prompts Groq, streams |
| `app.py` | Streamlit chat UI |
| `main.py` | CLI REPL |
| `eval.py` | Golden-question regression tests (`uv run eval.py`) |

## Troubleshooting

Hit an error or an odd answer? See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for
common issues (API keys, rate limits, port conflicts, plan/pricing answer
quality) and their fixes.
