"""Hybrid retrieval for the telecom document path: dense + sparse, fused.

Dense (vector) search captures meaning but can miss exact terms — a plan code,
a number, a literal phrase. Sparse (BM25) keyword search catches those but
misses paraphrases. We run both and merge their rankings with Reciprocal Rank
Fusion (RRF), which rewards documents that rank highly in *either* list without
needing the two scores to be on the same scale.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from ingest_faq import load_faq_documents
from ingest_guides import load_guide_documents
from ingest_tickets import load_ticket_documents
from retriever import get_retriever

TELECOM_LABELS = ["FAQ", "TICKETS", "GUIDES"]
BM25_K = 6          # candidates from the sparse retriever
FINAL_K = 6         # documents kept after fusion
RRF_K = 60          # RRF damping constant (standard default)


@lru_cache(maxsize=1)
def _telecom_corpus() -> tuple[Document, ...]:
    """All telecom documents in memory, for the BM25 index."""
    docs = load_faq_documents() + load_ticket_documents() + load_guide_documents()
    return tuple(docs)


@lru_cache(maxsize=1)
def _bm25() -> BM25Retriever:
    retriever = BM25Retriever.from_documents(list(_telecom_corpus()))
    retriever.k = BM25_K
    return retriever


def _key(doc: Document) -> tuple[str, str]:
    return (doc.metadata.get("source", ""), doc.page_content)


def retrieve_telecom(query: str) -> list[Document]:
    """Return telecom docs fused from dense (Chroma) and sparse (BM25) rankings."""
    dense = get_retriever().retrieve(query, labels=TELECOM_LABELS)
    sparse = _bm25().invoke(query)

    # Reciprocal Rank Fusion: score = sum of 1 / (RRF_K + rank) across both lists.
    scores: dict[tuple[str, str], float] = {}
    docs_by_key: dict[tuple[str, str], Document] = {}
    for ranked in (dense, sparse):
        for rank, doc in enumerate(ranked):
            k = _key(doc)
            scores[k] = scores.get(k, 0.0) + 1.0 / (RRF_K + rank + 1)
            docs_by_key.setdefault(k, doc)

    ordered = sorted(scores, key=lambda k: scores[k], reverse=True)
    return [docs_by_key[k] for k in ordered[:FINAL_K]]
