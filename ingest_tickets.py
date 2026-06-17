"""Ingest resolved support tickets into the `tickets` collection (FR-15).

One ticket row -> one vector document. Only resolved tickets are indexed so
retrieval surfaces proven resolution patterns (US-07). Re-runs are idempotent
(FR-17).
"""

from __future__ import annotations

import sqlite3

from langchain_chroma import Chroma
from langchain_core.documents import Document

from config import CHROMA_DIR, TICKETS_COLLECTION, TICKETS_DB
from embeddings import get_embeddings


def load_ticket_documents() -> list[Document]:
    docs: list[Document] = []
    conn = sqlite3.connect(TICKETS_DB)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT ticket_id, category, issue_type, description, resolution, status
            FROM tickets
            WHERE LOWER(status) = 'resolved'
            ORDER BY id
            """
        ).fetchall()
    finally:
        conn.close()

    for row in rows:
        content = (
            f"Issue: {row['issue_type']}\n"
            f"Category: {row['category']}\n"
            f"Description: {row['description']}\n"
            f"Resolution: {row['resolution']}"
        )
        docs.append(
            Document(
                page_content=content,
                metadata={
                    "source": "TICKETS",
                    "ticket_id": row["ticket_id"],
                    "category": row["category"],
                    "issue_type": row["issue_type"],
                },
            )
        )
    return docs


def ingest() -> int:
    docs = load_ticket_documents()

    store = Chroma(
        collection_name=TICKETS_COLLECTION,
        embedding_function=get_embeddings(),
        persist_directory=str(CHROMA_DIR),
    )
    store.reset_collection()
    store.add_documents(docs)

    print(f"[tickets] Ingested {len(docs)} resolved tickets into '{TICKETS_COLLECTION}'.")
    return len(docs)


if __name__ == "__main__":
    ingest()
