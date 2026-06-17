"""Ingest the plan & pricing catalog into the `plans` collection.

Source: data/plans.json (NovaCell official plan catalog). One plan/add-on ->
one vector document. Re-runs are idempotent: the collection is reset and
rebuilt each run, so updating plans.json and re-running refreshes pricing
answers without duplicates.
"""

from __future__ import annotations

import json

from langchain_chroma import Chroma
from langchain_core.documents import Document

from config import CHROMA_DIR, PLANS_COLLECTION, PLANS_JSON
from embeddings import get_embeddings


def _fmt_price(plan: dict) -> str:
    """Render a human-readable price line from whichever price fields exist."""
    if plan.get("monthly_price") is not None:
        line = f"${plan['monthly_price']:.0f}/month"
        if plan.get("price_per_line") is not None:
            line += f" (about ${plan['price_per_line']:.0f} per line)"
        if plan.get("lines_included"):
            line += f", includes {plan['lines_included']} lines"
        return line
    if plan.get("price") is not None:
        unit = plan.get("price_unit", "")
        return f"${plan['price']:.0f} {unit}".strip()
    return "see plan details"


def _fmt_data(plan: dict) -> str:
    if plan.get("data_unlimited"):
        return "Unlimited data"
    data = plan.get("data_gb")
    if isinstance(data, (int, float)):
        return f"{data:g} GB data"
    if isinstance(data, str):
        return data  # e.g. "uses your plan's high-speed data"
    return ""


def _plan_to_document(plan: dict, currency: str, last_updated: str) -> Document:
    name = plan.get("name", plan.get("id", "Unnamed plan"))
    lines: list[str] = [f"Plan: {name}"]

    if plan.get("type"):
        lines.append(f"Type: {plan['type']}")
    if plan.get("category"):
        lines.append(f"Category: {plan['category']}")

    lines.append(f"Price: {_fmt_price(plan)} ({currency})")

    data_str = _fmt_data(plan)
    if data_str:
        lines.append(f"Data: {data_str}")
    if plan.get("network"):
        lines.append(f"Network: {plan['network']}")
    if plan.get("hotspot_gb") is not None:
        lines.append(f"Hotspot: {plan['hotspot_gb']} GB")
    if plan.get("talk_minutes") is not None:
        lines.append(f"Talk: {plan['talk_minutes']}")
    if plan.get("texts") is not None:
        lines.append(f"Texts: {plan['texts']}")
    if plan.get("coverage"):
        lines.append(f"Coverage: {plan['coverage']}")
    if plan.get("contract"):
        lines.append(f"Contract: {plan['contract']}")
    if plan.get("eligibility"):
        lines.append(f"Eligibility: {plan['eligibility']}")
    if plan.get("intro_offer"):
        lines.append(f"Intro offer: {plan['intro_offer']}")
    if plan.get("features"):
        feats = "; ".join(plan["features"])
        lines.append(f"Features: {feats}")
    if plan.get("best_for"):
        lines.append(f"Best for: {plan['best_for']}")

    content = "\n".join(lines)

    metadata = {
        "source": "PLANS",
        "id": plan.get("id", ""),
        "name": name,
        "category": plan.get("category", plan.get("type", "")),
        "last_updated": last_updated,
    }
    return Document(page_content=content, metadata=metadata)


def load_plan_documents() -> list[Document]:
    with open(PLANS_JSON, encoding="utf-8") as f:
        catalog = json.load(f)

    currency = catalog.get("currency", "USD")
    last_updated = catalog.get("last_updated", "")
    plans = catalog.get("plans", [])
    return [_plan_to_document(p, currency, last_updated) for p in plans]


def ingest() -> int:
    docs = load_plan_documents()

    store = Chroma(
        collection_name=PLANS_COLLECTION,
        embedding_function=get_embeddings(),
        persist_directory=str(CHROMA_DIR),
    )
    store.reset_collection()
    store.add_documents(docs)

    print(f"[plans] Ingested {len(docs)} plans into '{PLANS_COLLECTION}'.")
    return len(docs)


if __name__ == "__main__":
    ingest()
