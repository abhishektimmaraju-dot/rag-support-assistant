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
Merged Retriever (parallel, top-3 each)
  ├── ChromaDB · faq       FAQ entries
  ├── ChromaDB · tickets   resolved ticket resolutions
  └── ChromaDB · guides    PDF guide chunks
     │  (9 source-labelled context docs)
     ▼
ChatPromptTemplate (context-only persona)
     ▼
Qwen3-32B on Groq (temperature=0)
     ▼
Streamed answer → Streamlit / CLI
```

- **Embeddings:** `all-MiniLM-L6-v2`, run locally (no embedding API cost).
- **Vector store:** ChromaDB, persisted to `chroma_store/`.
- **LLM:** `qwen/qwen3-32b` via Groq.
- **Framework:** LangChain (LCEL). **UI:** Streamlit + CLI.

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
2. Register `(LABEL, COLLECTION_NAME)` in `COLLECTIONS` in
   [retriever.py](retriever.py).

## Project layout

| File | Responsibility |
|---|---|
| `config.py` | Paths, model names, constants, `.env` loading |
| `embeddings.py` | Local HuggingFace embedding singleton |
| `ingest_faq.py` / `ingest_tickets.py` / `ingest_guides.py` / `ingest_plans.py` | Per-source ingestion |
| `ingest_all.py` | Build all collections in one pass |
| `retriever.py` | Merged per-collection retrieval + context formatting |
| `plan_facts.py` | Deterministic pricing reference computed from `plans.json` |
| `chain.py` | Prompt + Groq LLM + LCEL chain (+ streaming) |
| `app.py` | Streamlit chat UI |
| `main.py` | CLI REPL |

## Troubleshooting

Hit an error or an odd answer? See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for
common issues (API keys, rate limits, port conflicts, plan/pricing answer
quality) and their fixes.
