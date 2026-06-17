"""Ingest the telecom guide PDF into the `guides` collection (FR-16).

The PDF is chunked at 600 characters with 100-character overlap before
embedding. Re-runs are idempotent (FR-17).
"""

from __future__ import annotations

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHROMA_DIR, CHUNK_OVERLAP, CHUNK_SIZE, GUIDE_PDF, GUIDES_COLLECTION
from embeddings import get_embeddings


def load_guide_documents() -> list[Document]:
    pages = PyPDFLoader(str(GUIDE_PDF)).load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(pages)
    for chunk in chunks:
        chunk.metadata["source"] = "GUIDES"
    return chunks


def ingest() -> int:
    docs = load_guide_documents()

    store = Chroma(
        collection_name=GUIDES_COLLECTION,
        embedding_function=get_embeddings(),
        persist_directory=str(CHROMA_DIR),
    )
    store.reset_collection()
    store.add_documents(docs)

    print(f"[guides] Ingested {len(docs)} guide chunks into '{GUIDES_COLLECTION}'.")
    return len(docs)


if __name__ == "__main__":
    ingest()
