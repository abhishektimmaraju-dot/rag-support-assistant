"""Shared local embedding function (FR-09, NFR-02).

Uses HuggingFace `all-MiniLM-L6-v2` running on-device — no external embedding
API calls. The model is downloaded once (~90 MB) on first use and cached.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

from config import EMBEDDING_MODEL


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """Return a process-wide singleton embedding function."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )
