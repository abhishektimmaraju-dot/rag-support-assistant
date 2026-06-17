"""Hybrid retrieval + re-ranking for the telecom document path.

Two-stage retrieval:
1. Recall — gather candidates from BOTH dense (Chroma vector) and sparse (BM25
   keyword) search. Dense captures meaning; sparse catches exact terms (codes,
   numbers, literal phrases). The union maximises the chance the right document
   is in the candidate pool.
2. Precision — re-rank that small pool with a cross-encoder (rerank.py), which
   scores each candidate against the query directly, and keep the best few.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from ingest_faq import load_faq_documents
from ingest_guides import load_guide_documents
from ingest_tickets import load_ticket_documents
from rerank import rerank
from retriever import get_retriever

TELECOM_LABELS = ["FAQ", "TICKETS", "GUIDES"]
BM25_K = 6          # candidates from the sparse retriever
FINAL_K = 5         # documents kept after re-ranking


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
    """Return the top telecom docs: dense + sparse recall, cross-encoder rerank."""
    dense = get_retriever().retrieve(query, labels=TELECOM_LABELS)
    sparse = _bm25().invoke(query)

    # Stage 1 — union the candidate pool, de-duplicated.
    seen: set[tuple[str, str]] = set()
    candidates: list[Document] = []
    for doc in [*dense, *sparse]:
        k = _key(doc)
        if k not in seen:
            seen.add(k)
            candidates.append(doc)

    # Stage 2 — cross-encoder re-ranks the pool; keep the best.
    return rerank(query, candidates, FINAL_K)
