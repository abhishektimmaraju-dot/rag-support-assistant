"""Cross-encoder re-ranking.

Bi-encoders (the embedding model) score query and document *separately*, which
is fast but approximate. A cross-encoder reads the query and a candidate
document *together* and outputs a direct relevance score — much more accurate,
but too slow to run over the whole corpus. So the standard pattern is:
retrieve broadly (hybrid dense + sparse), then re-rank the small candidate set
with the cross-encoder and keep the best.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@lru_cache(maxsize=1)
def _model() -> CrossEncoder:
    # Downloaded once (~80 MB) on first use, then cached; runs locally.
    return CrossEncoder(RERANK_MODEL)


def rerank(query: str, docs: list[Document], top_k: int) -> list[Document]:
    """Return the top_k documents most relevant to the query, cross-encoded."""
    if not docs:
        return []
    pairs = [(query, doc.page_content) for doc in docs]
    scores = _model().predict(pairs)
    ranked = sorted(zip(docs, scores), key=lambda pair: pair[1], reverse=True)
    return [doc for doc, _ in ranked[:top_k]]
