# Architecture Comparison — Baseline vs Routed (for future reference)

This document compares the two architectures of the assistant and records the
**measured** trade-offs, so a future decision about retrieval design can be made
from evidence rather than intuition.

- **Baseline** — `main` branch: one flat RAG pipeline, dense retrieval only.
- **Routed** — `feature/routed-architecture` branch: a query router in front,
  hybrid retrieval + cross-encoder re-ranking on the document path.

Reproduce any number here with `uv run compare_retrieval.py` (retrieval quality
+ latency, local, no API) and `uv run eval.py` (end-to-end answer correctness).

---

## 1. The two architectures at a glance

| | Baseline (`main`) | Routed (`feature`) |
|---|---|---|
| Entry decision | none — every query takes the same path | **router** classifies: pricing / telecom / other |
| Scope control | a rule in the LLM prompt (can be ignored) | a **routing decision** — off-topic never reaches the LLM |
| Pricing answers | LLM ranks plans from dumped text | deterministic ranking in code (`plan_facts.py`) |
| Document retrieval | dense (vector) only, top-3 per collection | **hybrid** (dense + BM25) → **cross-encoder rerank** |
| LLM calls per query | 1 (answer) | 2 (router + answer); **0 answer calls** for off-topic |

```
Baseline                              Routed
--------                              ------
question                              question
   │                                     │
   ▼                                     ▼
dense retrieve (top-3 each)           router.classify ──► pricing | telecom | other
   │                                     │        │             │
   ▼                                     │   (code ranking) (hybrid+rerank)  (refuse, no LLM)
LLM does ranking / scope / answer        └──────────────┬──────────────┘
   │                                                    ▼
   ▼                                              LLM phrases answer
answer                                                  │
                                                        ▼
                                                     answer
```

---

## 2. Components — what each is, and why it's there

| Stage | Method / model | Type | Role |
|---|---|---|---|
| Dense retrieval | `all-MiniLM-L6-v2` | bi-encoder (local) | semantic match — meaning, paraphrase |
| Sparse retrieval | BM25 (`rank-bm25`) | lexical algorithm (no ML) | exact terms — codes, numbers, literal strings |
| Hybrid fusion | union + de-dup / RRF | plain code | combine the two candidate sets for recall |
| Re-ranking | `ms-marco-MiniLM-L-6-v2` | cross-encoder (local) | precision — score query+doc together, reorder |
| Router | `qwen/qwen3-32b` (1 word) | LLM (Groq) | decide *how* to answer before retrieval |
| Generation | `qwen/qwen3-32b` | LLM (Groq) | write the grounded answer |

Key distinction: the **bi-encoder** (dense) encodes query and doc *separately*
— fast, approximate, good for searching everything. The **cross-encoder**
(rerank) reads query and doc *together* — accurate, but only affordable on the
small candidate pool. BM25 is **not** a model; it's classic TF-IDF-style keyword
scoring, which is exactly why it catches what embeddings blur.

---

## 3. Measured results (retrieval quality + latency)

From `compare_retrieval.py` — 12 labelled questions (keyword-exact, paraphrase,
and near-duplicate-competition cases), measuring the rank of the correct document.

| Metric | dense-only | + hybrid (BM25) | + hybrid + rerank |
|---|---|---|---|
| **MRR** (rank of right doc; 1.0 = always first) | 0.88 | 0.86 | **0.96** |
| **Hit@1** (right doc is #1) | 10/12 | 10/12 | **11/12** |
| **Hit@3** (right doc in top 3) | 11/12 | 10/12 | **12/12** |
| **Latency** (local, per query) | ~25 ms | ~15 ms | ~80 ms |

(Quality numbers are identical on re-runs — retrieval is **deterministic**,
unlike the LLM's final wording.)

---

## 4. What the numbers actually mean

1. **The clearest win is a query dense-only missed entirely.** *"What does
   dialing \*123# do?"* — a literal token (`*123#`) has no semantic meaning, so
   dense returned a **miss**. BM25 found it (rank 7); the cross-encoder pulled it
   to **rank 1**. This is the textbook hybrid+rerank benefit.

2. **Hybrid alone did not help here — it slightly hurt (0.88 → 0.86).** Adding
   BM25 by itself injected noise (one query dropped from rank 1 → 6). On a small
   corpus, raw keyword fusion demotes some good semantic hits. The value is not
   "hybrid"; it's **"hybrid + re-ranking"** — the cross-encoder re-sorts the
   noisy union into a clean order.

3. **The re-ranker is the real quality lever:** MRR 0.88 → 0.96, and *every*
   correct doc lands in the top 3.

4. **On this small, easy corpus dense-only was already strong** (which is why
   both branches pass the end-to-end eval 9/9). The aggregate gain is real but
   **modest here, and grows with corpus size and messier real-world queries**.

---

## 5. Latency — where the cost actually is

The retrieval upgrades are cheap and local. The dominant added latency is the
**router**, which is a network LLM call — but it also *removes* latency on
off-topic questions by skipping the answer call entirely.

| Added component | Cost | Buys |
|---|---|---|
| BM25 | ~0 ms (local) | exact-term recall |
| Cross-encoder rerank | ~60 ms (local) | MRR 0.88 → 0.96, perfect Hit@3 |
| **Router** | **~0.3–1 s (network LLM call)** | scope guarantee; saves the answer call on off-topic |

---

## 6. Verdict & recommendations

- **Cross-encoder re-ranking: worth it.** Clear quality gain, ~60 ms, local,
  deterministic. Keep it.
- **Hybrid (BM25): keep it *with* re-ranking, not alone.** Its job is recall
  (don't lose the exact-term doc); the re-ranker cleans up the ordering.
- **Router: a product decision, not a retrieval one.** It costs ~0.5 s per
  in-scope query but guarantees the bot stays in its lane and never answers
  off-topic. If latency matters more than that guarantee, a **rules-based
  router** (keyword/regex, ~0 ms, no API) is a cheaper alternative worth
  benchmarking.
- **Determinism principle (the throughline):** retrieval and ranking are
  reproducible; the LLM is not. Keep sorting, scope, and pricing math in code;
  reserve the LLM for phrasing. This is why the routed design is more reliable.

### When the routed design pays off most
- Large corpora (thousands+ of docs) where dense recall alone misses.
- Domain queries full of exact terms (codes, SKUs, prices, identifiers).
- Strict scope requirements (the bot must refuse out-of-domain).
- A need for consistent, auditable answers (pricing, compliance, policy).

### How to reproduce
```bash
uv run compare_retrieval.py   # retrieval quality + latency (local, no API)
uv run eval.py                # end-to-end answer correctness (uses Groq)
```
