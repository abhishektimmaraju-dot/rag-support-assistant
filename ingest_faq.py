"""Ingest FAQ entries into the `faq` Chroma collection (FR-14).

One CSV row -> one vector document. Re-runs are idempotent (FR-17): the
collection is dropped and rebuilt each run, so updating faq.csv and re-running
this script refreshes the bot without duplicates (US-06).
"""

from __future__ import annotations

import csv

from langchain_chroma import Chroma
from langchain_core.documents import Document

from config import CHROMA_DIR, FAQ_COLLECTION, FAQ_CSV
from embeddings import get_embeddings


def load_faq_documents() -> list[Document]:
    docs: list[Document] = []
    with open(FAQ_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            question = (row.get("question") or "").strip()
            answer = (row.get("answer") or "").strip()
            if not question and not answer:
                continue
            content = f"Q: {question}\nA: {answer}"
            docs.append(
                Document(
                    page_content=content,
                    metadata={
                        "source": "FAQ",
                        "id": row.get("id", ""),
                        "category": (row.get("category") or "").strip(),
                        "question": question,
                    },
                )
            )
    return docs


def ingest() -> int:
    docs = load_faq_documents()

    # Idempotent: reset the collection before re-ingesting.
    store = Chroma(
        collection_name=FAQ_COLLECTION,
        embedding_function=get_embeddings(),
        persist_directory=str(CHROMA_DIR),
    )
    store.reset_collection()
    store.add_documents(docs)

    print(f"[faq] Ingested {len(docs)} FAQ entries into '{FAQ_COLLECTION}'.")
    return len(docs)


if __name__ == "__main__":
    ingest()
