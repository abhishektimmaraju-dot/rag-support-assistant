# MyTelecom Chatbot — Plain-English Guide for Project Managers

This explains *what the product does* and *how it works*, without assuming you
write code. If you can picture a new employee with a binder, you already
understand the core idea.

---

## The one big idea: RAG (Retrieval-Augmented Generation)

Think of the app as **a new employee with a binder.**

A plain AI chatbot answers from memory — and sometimes "remembers" things that
were never true (the industry word for this is **hallucination**). For a telecom
support bot, that's dangerous: it could invent a price, a promo code, or a
policy.

RAG fixes that by handing the AI **a binder of verified company documents** and
one strict rule:

> *"Only answer using what's in the binder. If it's not there, say so and tell
> the customer to call 611 or use the MyTelecom app."*

So **RAG = Retrieve** the right pages → **Augment** the question with those pages
→ **Generate** an answer grounded in them. That's the whole philosophy of this
product, and it's why the bot won't make up telecom pricing.

There are always **two phases**, and it helps to keep them separate in your head.

---

## Phase 1 — Building the binder (done once, or when documents change)

Before the bot can answer anything, we have to turn our knowledge into something
a computer can search by *meaning*.

| Concept | Plain meaning | Why a PM cares |
|---|---|---|
| **Data sources** | Our raw knowledge: a spreadsheet of FAQs, a database of solved support tickets, and a PDF guide. | This is the "source of truth." Quality of answers = quality of these documents. |
| **Chunking** | The PDF is too big to search as one blob, so it's sliced into ~600-character pieces. Each piece becomes one searchable "index card." | Lets the bot quote a *specific* paragraph instead of a whole document. |
| **Embeddings** | An AI model reads each piece of text and turns its *meaning* into a list of numbers. Texts with similar meaning get similar numbers — so *"my net is slow"* matches *"poor data speeds"* even with zero shared words. | This is what makes search feel smart instead of keyword-literal. Runs **locally — free, private, no internet.** |
| **Vector store** | A special database (**ChromaDB**) that holds all those number-versions and instantly finds the closest matches. Saved to disk so we don't rebuild it every time. | This is the "binder." Persisted = fast startup, cheap to run. |

In our build, Phase 1 produces **81 searchable cards**: 25 FAQs + 19 resolved
tickets + 37 guide chunks.

---

## Phase 2 — Answering a live question (every time someone asks)

1. **You type a question** in the web page → handled by the interface layer (Streamlit).
2. **Question → numbers:** the question is turned into the same kind of number-version, so it can be compared to the stored cards.
3. **Retrieval:** the bot searches **all three collections at once** and grabs the **top 3 closest matches from each = 9 documents**, each labelled by source (FAQ / TICKETS / GUIDES).
4. **Build the prompt:** those 9 documents + the question + the strict rules are assembled into one instruction for the AI.
5. **Generation:** the language model (the actual "brain," *Qwen3-32B*) writes the answer. It's set to **"no creativity, be factual"** so the same question gives a consistent answer. It runs on **Groq**, a cloud service that runs big AI models very fast.
6. **Streaming:** the answer appears word-by-word instead of all at once, so it feels responsive.

If steps 3–4 don't surface enough to answer, the bot **declines and redirects to
611** rather than guessing. That refusal behavior is a feature, not a bug.

---

## The pieces, and what each one *is*

| Piece | What it is | The layer it plays |
|---|---|---|
| **Streamlit** | Turns Python into a web app with almost no web code — that's why we got a chat page, sidebar, and sample buttons from one command. | Interface |
| **Embedding model** (`all-MiniLM-L6-v2`) | The "meaning-to-numbers" translator. From HuggingFace, runs on our own machine. | Retrieval |
| **ChromaDB** | The vector database — the binder you can search by similarity. | Retrieval |
| **LangChain** | The wiring. It glues retriever → prompt → LLM → output into one clean "chain." Without it, every handoff would be hand-coded. | Orchestration |
| **Groq + the LLM** (Qwen3-32B) | The cloud engine and the actual language model that composes the answer. | Generation |

---

## A few concepts worth naming (the "why," for stakeholder conversations)

- **Grounding & hallucination** — the entire design exists to keep answers
  *grounded* in real documents and stop the AI from inventing things. The
  out-of-scope test (asking "what's the capital of France?" and getting a polite
  redirect to 611) proves it works.
- **Local vs. cloud split** — embeddings run *on our machine* (private, free);
  only the final answer-writing goes to the cloud (Groq). A deliberate
  **cost + privacy trade-off** — customer questions aren't shipped to a paid
  embedding API.
- **Persistence** — the binder is saved to disk (`chroma_store/`), so startup is
  instant and we only re-index when documents actually change.
- **Idempotent ingestion** — re-running the index-builder safely rebuilds the
  binder **without creating duplicates**. This matters operationally: a support
  team can update the FAQ and re-run one script, no engineering involved.
- **Separation of concerns** — each file does one job (configure, embed, index,
  retrieve, chain, UI). That's why adding a *new* knowledge source is one new
  file plus one line of config — cheap to extend.

---

## What it can and can't do (scope, in PM terms)

**Can do today**
- Answer Tier-1 questions on connectivity, data, roaming, SIM, billing, voice,
  device, and account topics — grounded in our documents.
- Cite which source it used. Stream answers in a browser or a terminal.
- Refuse gracefully and redirect to a human channel when it doesn't know.

**Intentionally out of scope (v1)**
- No login or account-specific answers ("what's *my* balance?").
- No live billing/CRM integration — it doesn't see real-time customer data.
- No ticket creation, no multi-language, no conversation-aware retrieval.

These aren't gaps to fix before launch — they're the deliberate v1 boundary in
the PRD, and good candidates for a roadmap conversation.

---

## How "done" is measured (maps to the PRD goals)

| Goal | Target | Status |
|---|---|---|
| Deflect Tier-1 queries | >70% answered without escalation | Verified on sample questions |
| Ground every answer | 0 answers from the AI's own memory | Enforced by prompt + proven by the off-topic test |
| Speed | Under ~10s end-to-end | Local search + fast Groq generation |
| Accessible | No login, runs in a browser | Streamlit UI |

---

## If you want to go one level deeper

- **The technical companion doc:** [CODEBASE_EXPLANATION.md](CODEBASE_EXPLANATION.md) — file-by-file, with the full architecture and routing diagrams.
- **The requirements:** [PRD.md](PRD.md) — every feature mapped to an ID.
- **Run it yourself:** [README.md](README.md) — five terminal steps from zero to a live chat.
