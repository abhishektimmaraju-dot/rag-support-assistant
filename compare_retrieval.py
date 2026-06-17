"""Compare retrieval strategies — quality and latency, locally (no Groq calls).

Isolates the contribution of each upgrade on the telecom document path:
  1. dense-only            (baseline / main)
  2. dense + BM25 (RRF)    (hybrid, no rerank)
  3. dense + BM25 + rerank (feature: cross-encoder reorders the pool)

For each labelled question we record the RANK of the correct document under each
strategy, then report:
  - MRR   (mean reciprocal rank — rewards putting the right doc *first*)
  - Hit@1 / Hit@3  (was the right doc top-1 / in the top-3)
  - average latency per query (ms)

All three retrievers run on-device, so this is fast and free to re-run.

Usage:
    uv run compare_retrieval.py
"""

from __future__ import annotations

import time

from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from config import CHROMA_DIR, FAQ_COLLECTION, GUIDES_COLLECTION, TICKETS_COLLECTION
from embeddings import get_embeddings
from hybrid import _telecom_corpus
from rerank import _model

TELECOM = [FAQ_COLLECTION, TICKETS_COLLECTION, GUIDES_COLLECTION]
DEPTH = 15          # how deep we look for the gold doc
POOL = 10           # candidates pulled from each retriever before fusion/rerank
RRF_K = 60

# (question, substring that uniquely identifies the correct document)
# Mix of: exact-keyword (favors BM25), paraphrase (favors dense), and
# near-duplicate competition (favors the cross-encoder).
CASES = [
    ("How do I turn on VoLTE on my phone?", "What is VoLTE and how do I enable it"),
    ("My eSIM won't activate", "eSIM activation failing"),
    ("There's an echo during my calls", "Echo on every call"),
    ("Reset the APN to fix no mobile internet", "internet.telecom.example"),
    ("My mobile internet crawls in the evenings", "Why is my mobile internet so slow"),
    ("I think I paid for my plan twice this month", "Double charged for monthly plan"),
    ("Phone says no SIM after I rebooted it", "SIM not detected after a restart"),
    ("Can I keep my phone number when I switch carriers?", "keep my number when switching"),
    ("All my calls go straight to voicemail", "Calls going straight to voicemail"),
    ("The whole street has no service", "Network outage affecting whole street"),
    ("My number transfer is stuck and taking forever", "Number port taking too long"),
    ("What does dialing *123# do?", "*123#"),
]

_emb = get_embeddings()
_stores = {
    c: Chroma(collection_name=c, embedding_function=_emb, persist_directory=str(CHROMA_DIR))
    for c in TELECOM
}
_corpus = list(_telecom_corpus())
_bm25 = BM25Retriever.from_documents(_corpus)
_bm25.k = DEPTH


def _key(doc: Document) -> tuple[str, str]:
    return (doc.metadata.get("source", ""), doc.page_content)


def dense_only(query: str) -> list[Document]:
    scored: list[tuple[float, Document]] = []
    for c in TELECOM:
        for doc, dist in _stores[c].similarity_search_with_score(query, k=POOL):
            scored.append((dist, doc))
    scored.sort(key=lambda pair: pair[0])  # lower cosine distance = closer
    return [doc for _, doc in scored[:DEPTH]]


def hybrid_rrf(query: str) -> list[Document]:
    dense = dense_only(query)
    sparse = _bm25.invoke(query)
    scores: dict[tuple[str, str], float] = {}
    by_key: dict[tuple[str, str], Document] = {}
    for ranked in (dense, sparse):
        for rank, doc in enumerate(ranked):
            k = _key(doc)
            scores[k] = scores.get(k, 0.0) + 1.0 / (RRF_K + rank + 1)
            by_key.setdefault(k, doc)
    order = sorted(scores, key=lambda k: scores[k], reverse=True)
    return [by_key[k] for k in order[:DEPTH]]


def hybrid_rerank(query: str) -> list[Document]:
    seen: set[tuple[str, str]] = set()
    candidates: list[Document] = []
    for doc in [*dense_only(query)[:POOL], *_bm25.invoke(query)[:POOL]]:
        k = _key(doc)
        if k not in seen:
            seen.add(k)
            candidates.append(doc)
    scores = _model().predict([(query, d.page_content) for d in candidates])
    ranked = sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)
    return [doc for doc, _ in ranked[:DEPTH]]


STRATEGIES = [
    ("dense-only", dense_only),
    ("+ hybrid (BM25)", hybrid_rrf),
    ("+ hybrid + rerank", hybrid_rerank),
]


def _gold_rank(docs: list[Document], gold: str) -> int | None:
    g = gold.lower()
    for i, doc in enumerate(docs):
        if g in doc.page_content.lower():
            return i + 1
    return None


def run() -> None:
    # Sanity-check labels and warm up the models (so timing excludes model load).
    for q, gold in CASES:
        hits = sum(1 for d in _corpus if gold.lower() in d.page_content.lower())
        if hits != 1:
            print(f"  ! label warning: '{gold}' matches {hits} docs (expected 1)")
    for _, fn in STRATEGIES:
        fn("warmup query")

    results: dict[str, dict] = {n: {"rr": [], "h1": 0, "h3": 0, "t": 0.0} for n, _ in STRATEGIES}

    header = f"{'question':<46}" + "".join(f"{n:>20}" for n, _ in STRATEGIES)
    print("\n" + header)
    print("-" * len(header))

    for q, gold in CASES:
        cells = []
        for name, fn in STRATEGIES:
            t0 = time.perf_counter()
            docs = fn(q)
            results[name]["t"] += (time.perf_counter() - t0) * 1000
            rank = _gold_rank(docs, gold)
            rr = 1.0 / rank if rank else 0.0
            results[name]["rr"].append(rr)
            results[name]["h1"] += 1 if rank == 1 else 0
            results[name]["h3"] += 1 if rank and rank <= 3 else 0
            cells.append(f"rank {rank}" if rank else "miss")
        print(f"{q[:44]:<46}" + "".join(f"{c:>20}" for c in cells))

    n = len(CASES)
    print("-" * len(header))
    for metric, fmt in (("MRR", "{:.2f}"), ("Hit@1", "{}/%d" % n), ("Hit@3", "{}/%d" % n),
                        ("avg ms/query", "{:.0f}")):
        row = f"{metric:<46}"
        for name, _ in STRATEGIES:
            r = results[name]
            if metric == "MRR":
                val = fmt.format(sum(r["rr"]) / n)
            elif metric == "Hit@1":
                val = fmt.format(r["h1"])
            elif metric == "Hit@3":
                val = fmt.format(r["h3"])
            else:
                val = fmt.format(r["t"] / n)
            row += f"{val:>20}"
        print(row)


if __name__ == "__main__":
    run()
