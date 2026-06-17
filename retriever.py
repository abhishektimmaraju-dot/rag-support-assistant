"""Merged retriever over the knowledge collections (FR-06, FR-07, FR-08).

For each query it fetches the top-k documents from every registered collection
in parallel and formats them into a single source-labelled context block for
prompt injection. Most collections use the default top-3; the small `plans`
catalog is retrieved in full so pricing/comparison questions see every option.

Extensibility (NFR-06): add a new collection by appending to COLLECTIONS.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from langchain_chroma import Chroma
from langchain_core.documents import Document

from config import (
    CHROMA_DIR,
    FAQ_COLLECTION,
    GUIDES_COLLECTION,
    PLANS_COLLECTION,
    PLANS_TOP_K,
    TICKETS_COLLECTION,
    TOP_K,
)
from embeddings import get_embeddings

# Registry of collections to retrieve from: (label, collection_name, top_k).
COLLECTIONS: list[tuple[str, str, int]] = [
    ("FAQ", FAQ_COLLECTION, TOP_K),
    ("TICKETS", TICKETS_COLLECTION, TOP_K),
    ("GUIDES", GUIDES_COLLECTION, TOP_K),
    ("PLANS", PLANS_COLLECTION, PLANS_TOP_K),
]


class MergedRetriever:
    """Retrieves and labels documents from every registered collection."""

    def __init__(self) -> None:
        embeddings = get_embeddings()
        self._stores: dict[str, Chroma] = {
            collection: Chroma(
                collection_name=collection,
                embedding_function=embeddings,
                persist_directory=str(CHROMA_DIR),
            )
            for _, collection, _ in COLLECTIONS
        }

    def _search(self, label: str, collection: str, k: int, query: str) -> list[Document]:
        docs = self._stores[collection].similarity_search(query, k=k)
        for doc in docs:
            doc.metadata.setdefault("source", label)
        return docs

    def retrieve(self, query: str) -> list[Document]:
        """Return the merged, source-labelled documents for the query."""
        with ThreadPoolExecutor(max_workers=len(COLLECTIONS)) as pool:
            futures = [
                pool.submit(self._search, label, collection, k, query)
                for label, collection, k in COLLECTIONS
            ]
            results: list[Document] = []
            for future in futures:
                results.extend(future.result())
        return results

    def format_context(self, query: str) -> str:
        """Retrieve documents and render them as a source-labelled context block."""
        docs = self.retrieve(query)
        if not docs:
            return ""
        blocks = []
        for doc in docs:
            source = doc.metadata.get("source", "UNKNOWN")
            blocks.append(f"[{source}]\n{doc.page_content}")
        return "\n\n---\n\n".join(blocks)


# Process-wide singleton so the Streamlit app and CLI reuse loaded stores.
_retriever: MergedRetriever | None = None


def get_retriever() -> MergedRetriever:
    global _retriever
    if _retriever is None:
        _retriever = MergedRetriever()
    return _retriever
