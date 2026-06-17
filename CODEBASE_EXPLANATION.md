# Codebase Explanation

## MyTelecom RAG Chatbot — Codebase Architecture & Orchestration

Welcome to the MyTelecom architecture guide! This document explains how the
files in your repository fit together, how the system operates under the hood,
and how data flows from the user interface all the way to the language model and
back.

---

## 1. High-Level Architecture Diagram

The system is a single Python application with two interchangeable front ends
(a Streamlit web UI and a CLI), a shared retrieval-augmented generation (RAG)
core, a local vector database, and one external service (Groq) for generation.
The diagram below illustrates how the components interact:

```
                          ┌─────────────────────────────────────┐
                          │            FRONT ENDS                 │
                          │  ┌──────────────┐   ┌──────────────┐  │
                          │  │  app.py      │   │  main.py     │  │
                          │  │ (Streamlit)  │   │  (CLI REPL)  │  │
                          │  └──────┬───────┘   └──────┬───────┘  │
                          └─────────┼──────────────────┼──────────┘
                                    │                  │
                                    └────────┬─────────┘
                                             ▼
                              ┌──────────────────────────┐
                              │        chain.py          │   RAG core
                              │  prompt + LLM + LCEL      │
                              │  stream_answer()          │
                              └───────┬───────────┬───────┘
                                      │           │
                  ┌───────────────────┘           └───────────────────┐
                  ▼                                                    ▼
      ┌────────────────────────┐                          ┌────────────────────────┐
      │     retriever.py       │                          │   Groq LLM (external)  │
      │  MergedRetriever        │                          │   qwen/qwen3-32b       │
      │  per-collection top-k   │                          │   temperature = 0      │
      └───────────┬────────────┘                          └────────────────────────┘
                  │ similarity_search()
                  ▼
      ┌──────────────────────────────┐    ┌────────────────────────┐
      │           ChromaDB            │◄───│     embeddings.py      │
      │        chroma_store/          │    │  all-MiniLM-L6-v2      │
      │  ┌─────┬──────┬─────┬───────┐ │    │  (local, on-device)    │
      │  │ faq │tickets│guide│ plans │ │    └────────────────────────┘
      │  └─────┴──────┴─────┴───────┘ │
      └──────────────▲───────────────┘
                     │ built once by
                     │
      ┌──────────────┴─────────────────────────────────────────────────────┐
      │                INGESTION (run once / on data change)                 │
      │  ingest_faq.py  ingest_tickets.py  ingest_guides.py  ingest_plans.py │
      │       │               │                  │                 │         │
      │       ▼               ▼                  ▼                 ▼         │
      │  data/faq.csv   data/tickets.db   data/telecom_guide.pdf  data/plans.json
      └──────────────────────────────────────────────────────────────────────┘
```

**Embedding model:** `all-MiniLM-L6-v2` (local, HuggingFace — no API cost)
**Vector store:** ChromaDB, persisted to `chroma_store/`
**LLM:** `qwen/qwen3-32b` via Groq API
**Framework:** LangChain (LCEL chain) · **UI:** Streamlit + CLI

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
  2. For each incoming query, it runs `similarity_search` against **every
     collection in parallel** (a thread pool), pulling that collection's `top_k`.
     The default is 3; the small **`plans` catalog is retrieved in full (k=20)**
     so pricing/comparison questions ("cheapest unlimited?", "which roaming
     pass?") see *every* option, not just the 3 nearest matches.
  3. `format_context()` stitches the results into a single block where every
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

**5. `chain.py`**

- **Purpose:** The RAG core — assembles retrieval, the prompt, and the LLM into
  one LangChain LCEL chain, and exposes streaming.
- **Workflow:**
  1. A strict **system prompt** instructs the model to answer using **only** the
     retrieved context — no outside knowledge, no invented prices/codes/policy.
  2. If the context is insufficient, the model must say so plainly and direct the
     user to **call 611 or use the MyTelecom app**.
  3. The LCEL chain wires it together:
     `{context: retriever, question} → ChatPromptTemplate → ChatGroq → StrOutputParser`.
  4. The Groq LLM is `qwen/qwen3-32b` at **temperature 0** (deterministic), with
     `reasoning_format="parsed"` so the model's internal reasoning is stripped
     from the final answer.
  5. Two entry points: `answer()` (full response) and `stream_answer()`
     (token-by-token streaming for a responsive UI).
- **Scope-awareness (in the system prompt):** the prompt restricts the bot to
  NovaCell telecom topics. Off-topic requests (holiday itineraries, weather,
  news, general knowledge) get a one-line "I can only help with NovaCell telecom
  topics" redirect; if there's a relevant telecom angle (e.g. the user mentioned
  travelling abroad) it may then offer that help (roaming options) without
  inventing the off-topic content. This stops it from over-confidently answering
  a "3-day India holiday plan" as if it were a roaming request.
- **Plan & pricing rules (in the system prompt):** because plan/pricing is a
  distinct, high-stakes domain, the prompt adds explicit guardrails:
  - **`[PLANS]` is authoritative.** If an older `[FAQ]` or `[GUIDES]` entry
    conflicts with the catalog (e.g. a stale roaming bundle), follow `[PLANS]`
    and suppress the outdated one. Never surface a plan/price that isn't in
    `[PLANS]`.
  - **No invented plans or prices** — only names/prices that appear verbatim in
    a `[PLANS]` block.
  - **"Cheapest/best" questions** lead with the lowest-priced plan that has *no*
    eligibility restriction, then note any cheaper eligibility-gated plans
    (Student, 55+) — an eligibility-restricted plan is never presented as
    available to a customer who hasn't said they qualify.
  - **Roaming/travel** answers compute each option's total cost for the stated
    trip length and recommend the genuinely lower-cost option.

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
chain.py builds the LCEL input { question }
        │
        ▼
retriever.py — MergedRetriever.format_context(question)
        │
        ├────────────┬────────────┬────────────┬────────────┐
        ▼            ▼            ▼            ▼          (parallel)
   ChromaDB·faq ChromaDB·tickets ChromaDB·guides ChromaDB·plans
   top-3 docs    top-3 docs      top-3 docs   full catalog (k=20)
        └────────────┴────────────┴────────────┴────────────┘
                       │  source-labelled documents
                       ▼
        ChatPromptTemplate
          ├─ system: "answer ONLY from context; else say so + call 611"
          └─ human:  the user's question
                       │
                       ▼
        Qwen3-32B on Groq  (temperature = 0, reasoning stripped)
                       │
                       ▼
        StrOutputParser → tokens streamed back
                       │
        ┌──────────────┴───────────────┐
        ▼                               ▼
  Context sufficient?            Context insufficient?
  → Grounded, source-based       → "I don't have that info —
    step-by-step answer            call 611 or use the MyTelecom app"
        │                               │
        └───────────────┬───────────────┘
                        ▼
        Front end renders the streamed reply
        (chat bubble in Streamlit, or live text in the CLI)
```

The key design guarantee: **the model never answers from its own internal
knowledge.** Off-topic questions (e.g. "What is the capital of France?") are
declined and redirected — proven in testing.

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
Streamlit, and pypdf.)

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
| Parallel retrieval from 4 collections, source labels | FR-06…08 | `retriever.py` |
| Local embeddings (`all-MiniLM-L6-v2`) | FR-09, NFR-02 | `embeddings.py` |
| Context-only answers, refusal→611, temp 0, Qwen3-32B/Groq | FR-10…13 | `chain.py` |
| Ingestion (CSV / SQLite / PDF / JSON), idempotent | FR-14…17 | `ingest_*.py` |
| CLI REPL with `quit` | FR-18…19 | `main.py` |
| No secrets in code, persisted store, extensibility | NFR-03/05/06 | `config.py`, `retriever.py` |

---

## 6. Change Log — Plans & Pricing knowledge source

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
```
