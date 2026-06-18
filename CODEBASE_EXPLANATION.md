# Codebase Explanation

## MyTelecom RAG Chatbot — Codebase Architecture & Orchestration

Welcome to the MyTelecom architecture guide! This document explains how the
files in your repository fit together, how the system operates under the hood,
and how data flows from the user interface all the way to the language model and
back.

---

## 1. High-Level Architecture Diagram

The system is a single Python application with two interchangeable front ends
(a Streamlit web UI and a CLI). Each question is **routed** before retrieval:
the router decides *how* to answer, then sends it down a focused path. The
diagram below shows the flow:

```
                 ┌──────────────┐   ┌──────────────┐
                 │  app.py      │   │  main.py     │   FRONT ENDS
                 │ (Streamlit)  │   │  (CLI REPL)  │
                 └──────┬───────┘   └──────┬───────┘
                        └────────┬─────────┘
                                 ▼
                        ┌──────────────────┐
                        │    router.py     │   classifies the question
                        │   classify()     │   (one small Groq call)
                        └───┬──────┬───────┬┘
            pricing ───────┘      │        └─────── other
                ▼          telecom ▼                  ▼
   ┌────────────────────┐  ┌────────────────────┐  ┌────────────────────┐
   │ plan_facts.py      │  │ hybrid.py          │  │ deterministic       │
   │ pricing reference  │  │ dense + BM25 recall │  │ refusal (no LLM)    │
   │ + plans-only ctx   │  │ → rerank.py (top 5) │  └────────────────────┘
   └─────────┬──────────┘  └─────────┬──────────┘
             └───────────┬───────────┘
                         ▼
                 ┌──────────────────┐        ┌────────────────────────┐
                 │     chain.py     │───────▶│   Groq LLM (external)  │
                 │  prompt + stream │        │   qwen/qwen3-32b · T=0 │
                 └──────────────────┘        └────────────────────────┘

   Retrieval sources (built once by the ingest scripts):
   ┌──────────────────────────────────────────────────────────────────┐
   │  ChromaDB (chroma_store/):  faq · tickets · guides · plans         │
   │  + BM25 keyword index over faq/tickets/guides (in memory)          │
   │  embeddings: all-MiniLM-L6-v2 (local) · rerank: ms-marco-MiniLM    │
   └──────────────────────────────────────────────────────────────────┘
```

**Embedding model:** `all-MiniLM-L6-v2` (local, HuggingFace — no API cost)
**Vector store:** ChromaDB, persisted to `chroma_store/` · **Keyword index:** BM25
**Re-ranker:** `ms-marco-MiniLM-L-6-v2` cross-encoder (local)
**LLM:** `qwen/qwen3-32b` via Groq API · **Framework:** LangChain · **UI:** Streamlit + CLI

---

## 2. File-by-File Breakdown

Here is an overview of the key files in your workspace and their
responsibilities.

### A. Data Ingestion & Storage

**1. `ingest_faq.py`, `ingest_tickets.py`, `ingest_guides.py`, `ingest_plans.py`**

- **Purpose:** Parse each knowledge source and index it into ChromaDB. Run once
  (or whenever you add/update source data) to build the search index.
- **How they work:**
  1. Each script reads its own source and produces LangChain `Document` objects
     tagged with a `source` label (`FAQ`, `TICKETS`, `GUIDES`, or `PLANS`) in
     metadata — this label is later shown to the user as a citation.
  2. `ingest_faq.py` reads `data/faq.csv` and turns **one row into one vector
     document** (`Q: … / A: …`), keeping `category` and `id` as metadata.
  3. `ingest_tickets.py` opens `data/tickets.db` (SQLite), selects **only
     resolved tickets**, and turns **one ticket into one vector document**
     (issue + category + description + resolution).
  4. `ingest_guides.py` reads `data/telecom_guide.pdf` with `PyPDFLoader` and
     splits it into **600-character chunks with 100-character overlap** using a
     `RecursiveCharacterTextSplitter`.
  5. `ingest_plans.py` reads `data/plans.json` (the official NovaCell plan
     catalog) and turns **one plan/add-on into one vector document** — a readable
     block of name, price, data, network, hotspot, contract, eligibility, intro
     offer, features, and "best for", with `id`/`name`/`category` as metadata.
  6. Every document is embedded locally with the `all-MiniLM-L6-v2`
     SentenceTransformer model (see `embeddings.py`) and stored, with metadata,
     in a directory-based ChromaDB at `chroma_store/`.
  7. **Idempotency:** each script calls `reset_collection()` before re-adding, so
     re-running never produces duplicates. Updating a CSV/DB/PDF/JSON and
     re-running the matching script refreshes the bot without a redeploy.

**2. `ingest_all.py`**

- **Purpose:** Convenience runner that executes all four ingest scripts in one
  pass and prints a document count summary.

**3. ChromaDB vector store (`chroma_store/`)**

- **Purpose:** Stores the dense vector embeddings and metadata for all knowledge.
- **Collections:**
  - `faq` — FAQ question/answer pairs (1 per CSV row).
  - `tickets` — resolved support-ticket resolutions (1 per ticket).
  - `guides` — chunked text from the telecom guide PDF.
  - `plans` — plan & pricing catalog entries (1 per plan/add-on).
- Persists to disk, so the app does **not** re-ingest on every start.

**4. Source data (`data/`)**

- `data/faq.csv` — columns: `id, question, answer, category`.
- `data/tickets.db` — SQLite with a `tickets` table (`ticket_id, category,
  issue_type, description, resolution, status`). Categories covered:
  `connectivity, data, roaming, sim, billing, voice, device, account`.
- `data/telecom_guide.pdf` — long-form technical guide content.
- `data/plans.json` — official plan catalog (`{currency, last_updated,
  operator, plans:[…]}`): prepaid, postpaid, family, data-only, specialty
  (student / 55+) plans, plus roaming and hotspot add-ons. This is the
  **authoritative source for all plan and pricing answers** — when it conflicts
  with an older FAQ/guide entry, the catalog wins (see `chain.py`).

### B. Core Backend / RAG Services

**1. `config.py`**

- **Purpose:** Single source of truth for paths, model names, and tuning
  constants; loads secrets from `.env`.
- **Key config:**
  - Paths: `DATA_DIR`, `CHROMA_DIR`, plus per-source file paths. (It tolerates
    either a `data/` or `Data/` folder.)
  - Collection names, `EMBEDDING_MODEL`, `TOP_K = 3` (default per collection),
    `PLANS_TOP_K = 20` (the whole plan catalog — see `retriever.py`),
    `CHUNK_SIZE = 600`, `CHUNK_OVERLAP = 100`.
  - LLM settings: `LLM_MODEL = "qwen/qwen3-32b"`, `LLM_TEMPERATURE = 0`.
  - `require_groq_key()` — fails fast with a clear message if `GROQ_API_KEY` is
    missing (no credentials are ever hard-coded).

**2. `embeddings.py`**

- **Purpose:** Provides the shared, local embedding function used by both
  ingestion and retrieval.
- **How it works:** Returns a cached singleton `HuggingFaceEmbeddings` instance
  for `all-MiniLM-L6-v2`, with normalized embeddings. The model is downloaded
  once (~90 MB) on first use and then runs entirely on-device — no embedding API
  calls or cost.

**3. `retriever.py`**

- **Purpose:** Performs unstructured document search across all knowledge
  collections.
- **Workflow:**
  1. Holds a registry of `(label, collection, top_k)` tuples:
     `COLLECTIONS = [("FAQ", …, 3), ("TICKETS", …, 3), ("GUIDES", …, 3),
     ("PLANS", …, 20)]`. To add a new knowledge source, you write a new
     `ingest_*.py` and append one line here.
  2. For each incoming query, it runs `similarity_search` against the requested
     collections **in parallel** (a thread pool), pulling each one's `top_k`.
     The default is 3; the small **`plans` catalog is retrieved in full (k=20)**.
     `retrieve(query, labels=…)` accepts an optional list of collection labels so
     a caller can search only a **subset** — the router uses this to keep each
     path focused (e.g. `["PLANS"]` for pricing, `["FAQ","TICKETS","GUIDES"]` for
     the document path).
  3. `format_docs()` stitches a document list into a single block where every
     document is prefixed with its source label, e.g. `[FAQ] …`, `[TICKETS] …`,
     `[GUIDES] …`, `[PLANS] …`. This is what gets injected into the prompt.

**4. `plan_facts.py`**

- **Purpose:** Produces a deterministic, pre-sorted pricing reference computed
  directly from `data/plans.json` in plain Python.
- **Why it exists:** Hosted LLMs are not byte-for-byte deterministic even at
  temperature 0, so asking the model to *rank* plans by price gave inconsistent
  "cheapest plan" answers (sometimes $45, sometimes $60, sometimes the
  eligibility-gated $30). This module does the ranking in code and hands the
  model the result, turning "cheapest" from a judgment the model makes into a
  fact it reads. `build_pricing_reference()` returns a price-sorted plan table
  plus a single recommended lead ("cheapest unlimited plan with no eligibility:
  Prepaid Unlimited, $45/month"). `chain.py` prepends it to the context.

**5. `router.py`**

- **Purpose:** Classifies each question **before** retrieval, so the system can
  send it down a focused path instead of asking one overloaded prompt to do
  everything. One small Groq call returns `pricing`, `telecom`, or `other`
  (defaults safely to `telecom` on any ambiguity).
- **Why it matters:** Scope enforcement is now a **routing decision**, not a
  prompt rule the model could ignore — off-topic questions never reach the LLM.

**6. `hybrid.py`**

- **Purpose:** Two-stage retrieval for the telecom document path.
  1. **Recall** — gather candidates from **both** dense (Chroma vector) and
     sparse (BM25 keyword) search, then union and de-duplicate them. Dense
     captures meaning; sparse catches exact terms (codes, numbers, literal
     phrases).
  2. **Precision** — re-rank that pool with the cross-encoder (`rerank.py`) and
     keep the top 5.

**7. `rerank.py`**

- **Purpose:** A cross-encoder (`ms-marco-MiniLM-L-6-v2`, local, ~80 MB) reads
  the query and each candidate **together** and scores relevance directly — far
  more accurate than the embedding model's separate scoring, but only run over
  the small candidate set, not the whole corpus.

**8. `chain.py`**

- **Purpose:** The routed RAG core — classifies, assembles route-specific
  context, then prompts the LLM and streams the answer.
- **Workflow:**
  1. `classify()` (router) picks the route.
  2. **`other`** → returns a deterministic refusal, **no LLM call at all**.
  3. **`pricing`** → context = the `plan_facts.py` pricing reference + plans-only
     retrieval (no FAQ/guide), so stale data cannot leak into pricing answers.
  4. **`telecom`** → context = `hybrid.py` (dense + BM25 → rerank).
  5. The context + question go through `ChatPromptTemplate → ChatGroq →
     StrOutputParser`. The LLM is `qwen/qwen3-32b` at **temperature 0** with
     `reasoning_format="parsed"`.
  6. Two entry points: `answer()` (full) and `stream_answer()` (token-by-token).
- **System-prompt guardrails (still enforced):** answer ONLY from context; never
  invent a plan/price; use the pre-sorted pricing reference directly for
  "cheapest/best"; compare roaming totals correctly. (Scope is now handled by
  the router rather than the prompt.)

**9. `eval.py`**

- **Purpose:** Golden-question evaluation harness. Encodes the scenario's sample
  Q&A as objective `must_include` / `must_exclude` checks, so an architecture
  change can be verified against a fixed yardstick instead of eyeballing. Run
  with `uv run eval.py`.

### C. Front-End Interfaces

**1. `app.py` (Streamlit web UI)**

- **Purpose:** The browser-based chat interface.
- **Key components:**
  - **Chat interface:** Free-text input; replies stream in token-by-token via
    `st.write_stream`.
  - **Sample Prompts Panel:** Sidebar buttons (e.g. "Why is my mobile internet
    so slow?") that send a question with one click.
  - **Session history:** Conversation is kept in `st.session_state` and replayed
    on each rerun.
  - **Clear conversation:** A button resets the session.
  - **Health hints:** Warns in the sidebar if `GROQ_API_KEY` or `chroma_store/`
    is missing.

**2. `main.py` (CLI REPL)**

- **Purpose:** A no-browser interactive loop for terminal use.
- **Key operations:** Validates the Groq key on start, then loops: read a
  question, stream the grounded answer, repeat. Typing `quit` or `exit` (or
  Ctrl-C) ends the session. Errors are caught per-turn so the REPL never crashes.

---

## 3. End-to-End Chat Routing Flow

What happens when a user types a message? Here is the exact path:

```
[ User inputs a question ]
        │
        ▼
Front end (app.py / main.py) calls stream_answer(question)
        │
        ▼
router.py — classify(question)  ──►  pricing | telecom | other
        │
        ├──────────────── other ────────────────┐
        │                                        ▼
        │                          Deterministic refusal — NO LLM call
        │                          "I can only help with telecom topics"
        │
        ├──────────────── pricing ───────────────┐
        │                                         ▼
        │              plan_facts.py pricing reference
        │              + plans-only retrieval (no FAQ/guide)
        │                                         │
        └──────────────── telecom ────────────┐  │
                                               ▼  │
                  hybrid.py: dense (Chroma) + BM25 │
                  → union → rerank.py (top 5)      │
                                               │   │
                                               ▼   ▼
                                ChatPromptTemplate (context + question)
                                               │
                                               ▼
                          Qwen3-32B on Groq  (temperature 0, reasoning stripped)
                                               │
                                               ▼
                          StrOutputParser → tokens streamed to the UI
```

Two design guarantees: **the model never answers from its own internal
knowledge** (it answers only from retrieved context, else redirects to 611), and
**off-topic questions never reach the model** — the router refuses them
deterministically.

---

## 4. Setting Up and Running the Orchestration

To run the application and see the orchestration in real time, follow these
steps in your terminal.

### Step 1: Environment Setup

Ensure you have a `.env` file in the project root with the following variable
(copy it from `.env.example`):

```env
GROQ_API_KEY=your-groq-api-key-here
```

A free key is available at [console.groq.com](https://console.groq.com). The key
must start with `gsk_` and have no surrounding quotes or spaces.

### Step 2: Install Dependencies

The project targets Python 3.11–3.12.

```bash
uv sync
# or:  pip install -e .
```

(This installs LangChain, ChromaDB, sentence-transformers, the Groq client,
Streamlit, pypdf, and `rank-bm25` for hybrid search. On first run the app also
downloads the cross-encoder re-ranker model `ms-marco-MiniLM-L-6-v2`, ~80 MB.)

### Step 3: Index the Documents

If the local vector store isn't built yet, run the ingestion to parse the
sources and populate ChromaDB:

```bash
uv run ingest_all.py
```

(This reads `data/faq.csv`, `data/tickets.db`, `data/telecom_guide.pdf`, and
`data/plans.json`, computes local embeddings, and initializes `chroma_store/`.
The embedding model downloads ~90 MB on the first run only. Expected output:
`25 FAQ + 19 tickets + 37 guide chunks + 14 plans = 95 documents indexed`.)

You can also re-run a single source after editing it:

```bash
uv run ingest_faq.py        # after editing data/faq.csv
uv run ingest_tickets.py    # after editing data/tickets.db
uv run ingest_guides.py     # after editing data/telecom_guide.pdf
uv run ingest_plans.py      # after editing data/plans.json
```

### Step 4: Run the Web UI

Launch the Streamlit application:

```bash
uv run streamlit run app.py
```

Open your browser to `http://localhost:8501` to start interacting with the chat
UI, click sample questions, and clear the conversation.

### Step 5: Run the CLI (optional)

For a no-browser experience:

```bash
uv run main.py
```

Type a question and press Enter; answers stream live. Type `quit` to exit.

---

## 5. Requirement Traceability (quick reference)

| Area | PRD IDs | Where it lives |
|---|---|---|
| Free-text + sample-button chat, history, clear, streaming | FR-01…05 | `app.py` |
| Query routing (pricing / telecom / out-of-scope) | — | `router.py` |
| Hybrid retrieval (dense + BM25) + cross-encoder re-rank | FR-06…08 | `hybrid.py`, `rerank.py`, `retriever.py` |
| Local embeddings (`all-MiniLM-L6-v2`) | FR-09, NFR-02 | `embeddings.py` |
| Context-only answers, refusal→611, temp 0, Qwen3-32B/Groq | FR-10…13 | `chain.py` |
| Ingestion (CSV / SQLite / PDF / JSON), idempotent | FR-14…17 | `ingest_*.py` |
| CLI REPL with `quit` | FR-18…19 | `main.py` |
| Golden-question regression tests | — | `eval.py` |
| No secrets in code, persisted store, extensibility | NFR-03/05/06 | `config.py`, `retriever.py` |

---

## 6. Change Log

### 6.1 Plans & Pricing knowledge source

The `plans` collection was added after launch, when users reported wrong/vague
answers to plan and pricing questions: the original three sources simply had no
current plan data. Integrating `data/plans.json` followed the existing
extensibility pattern (NFR-06) — **one new ingest script plus one line in the
retriever registry** — with two scenario-specific refinements:

1. **Full-catalog retrieval for plans** (`PLANS_TOP_K = 20`): "cheapest" and
   "which roaming pass" questions need the whole catalog in view, so the plans
   collection is retrieved in full rather than top-3.
2. **`[PLANS]` precedence + pricing guardrails in the prompt**: the catalog is
   authoritative over older FAQ/guide entries (e.g. a stale "EU Roaming Bundle"
   that no longer exists), and the model is constrained to never invent a
   plan/price and to compare roaming costs correctly.
3. **Deterministic pricing reference** (`plan_facts.py`): because the LLM is not
   reproducible at temperature 0, ranking was moved out of the model. Prices are
   sorted in Python and the recommended "cheapest" lead is computed and injected
   into the context, so the answer is stable across runs.
4. **Stale data reconciled**: an outdated `faq.csv` row ("EU Roaming Bundle
   $15/day") contradicted the catalog and occasionally leaked through. Because a
   prompt rule can't reliably override bad data in a non-deterministic model, the
   row was rewritten to match the current add-ons and the FAQ re-ingested.
5. **Scope-awareness**: off-topic requests (itineraries, weather, general
   knowledge) now get a brief "telecom topics only" redirect instead of being
   answered over-confidently. (`$` is also escaped in the UI so prices don't
   render as LaTeX math.)

Files touched: `ingest_plans.py` (new), `plan_facts.py` (new), `config.py`,
`retriever.py`, `chain.py`, `ingest_all.py`, `app.py`, `data/faq.csv`.

### 6.2 Routed architecture (query routing + hybrid search + re-ranking)

The fixes in 6.1 were pragmatic patches around one overloaded prompt — the LLM
was doing ranking, stale-data suppression, and scope judgment itself. This
change moves those jobs to the right layer:

1. **Query router** (`router.py`): classify each question first; pricing,
   telecom, and out-of-scope each get a focused path. Scope becomes a hard
   routing decision (no LLM for `other`), and pricing context no longer pulls
   FAQ/guide, so stale data cannot leak into pricing answers.
2. **Hybrid retrieval** (`hybrid.py`): the telecom path fuses dense (Chroma) and
   sparse (BM25) recall so exact terms are caught alongside semantic matches.
3. **Cross-encoder re-ranking** (`rerank.py`): the candidate pool is re-scored
   for precision; the top 5 go to the LLM.
4. **Eval harness** (`eval.py`): golden-question checks; held at **9/9** across
   the baseline and each step, so the refactor preserved correctness.

New files: `router.py`, `hybrid.py`, `rerank.py`, `eval.py`. Modified:
`chain.py`, `retriever.py`. New dependency: `rank-bm25`.
```
