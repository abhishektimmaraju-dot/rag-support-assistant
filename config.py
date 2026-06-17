"""Central configuration for the Telecom RAG chatbot.

All paths are resolved relative to this file so scripts run from any CWD.
Secrets are loaded from .env (never hard-coded) per NFR-03.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env once, on import.
load_dotenv()

# --- Paths -------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent


def _resolve_data_dir() -> Path:
    """Return the data directory, tolerating either `data/` or `Data/`."""
    for name in ("data", "Data"):
        candidate = ROOT_DIR / name
        if candidate.is_dir():
            return candidate
    # Default to lowercase if neither exists yet.
    return ROOT_DIR / "data"


DATA_DIR = _resolve_data_dir()
CHROMA_DIR = ROOT_DIR / "chroma_store"

FAQ_CSV = DATA_DIR / "faq.csv"
TICKETS_DB = DATA_DIR / "tickets.db"
GUIDE_PDF = DATA_DIR / "telecom_guide.pdf"
PLANS_JSON = DATA_DIR / "plans.json"

# --- Collections (FR-06) -----------------------------------------------------
FAQ_COLLECTION = "faq"
TICKETS_COLLECTION = "tickets"
GUIDES_COLLECTION = "guides"
PLANS_COLLECTION = "plans"

# --- Embeddings (FR-09, NFR-02) ---------------------------------------------
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# --- Retrieval (FR-07) -------------------------------------------------------
TOP_K = 3  # default documents fetched per collection

# Pricing/comparison questions ("cheapest unlimited?", "which roaming pass?")
# need the whole catalog in context, not just the top-3 semantic matches.
# The plan catalog is small, so we retrieve all of it.
PLANS_TOP_K = 20

# --- Chunking (FR-16) --------------------------------------------------------
CHUNK_SIZE = 600
CHUNK_OVERLAP = 100

# --- LLM (FR-12, FR-13) ------------------------------------------------------
LLM_MODEL = "qwen/qwen3-32b"
LLM_TEMPERATURE = 0

GROQ_API_KEY = os.getenv("GROQ_API_KEY")


def require_groq_key() -> str:
    """Return the Groq API key or raise a clear error if missing."""
    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Copy .env.example to .env and add your key "
            "(free tier at https://console.groq.com)."
        )
    return GROQ_API_KEY
