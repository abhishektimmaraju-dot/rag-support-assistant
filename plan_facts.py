"""Deterministic pricing facts computed directly from data/plans.json.

Hosted LLMs are not byte-for-byte deterministic even at temperature 0, so
asking the model to rank plans by price gives inconsistent "cheapest" answers.
This module does the ranking in plain Python and hands the model a pre-sorted,
authoritative reference, turning "cheapest" from a judgment into a fact lookup.
"""

from __future__ import annotations

import json
from functools import lru_cache

from config import PLANS_JSON


def _data_desc(plan: dict) -> str:
    if plan.get("data_unlimited"):
        return "unlimited data"
    gb = plan.get("data_gb")
    return f"{gb:g} GB" if isinstance(gb, (int, float)) else str(gb or "")


@lru_cache(maxsize=1)
def build_pricing_reference() -> str:
    """Return a deterministic, sorted pricing summary for monthly plans.

    Add-ons (Day Pass, hotspot boosts, etc.) use per-day/per-month add-on
    pricing and are intentionally excluded from the monthly-plan ranking.
    """
    with open(PLANS_JSON, encoding="utf-8") as f:
        catalog = json.load(f)

    plans = catalog.get("plans", [])
    monthly = [p for p in plans if p.get("monthly_price") is not None]
    monthly.sort(key=lambda p: p["monthly_price"])

    lines = [
        "PLAN PRICING REFERENCE (authoritative, pre-sorted from the catalog — "
        "use these rankings directly for cheapest/best/compare questions):",
        "Monthly plans, lowest price first:",
    ]
    for p in monthly:
        elig = p.get("eligibility")
        tag = (
            f"requires eligibility ({elig})"
            if elig
            else "no eligibility requirement"
        )
        lines_inc = f", {p['lines_included']} lines" if p.get("lines_included") else ""
        lines.append(
            f"- {p['name']}: ${p['monthly_price']:.0f}/month{lines_inc}, "
            f"{_data_desc(p)} — {tag}"
        )

    # Derived answer for the common "cheapest unlimited" question. We give ONE
    # recommended lead (cheapest unlimited with no eligibility) plus the cheaper
    # eligibility-gated options to mention afterwards — so the model does not
    # have to decide which "cheapest" to present.
    unlimited = [p for p in monthly if p.get("data_unlimited")]
    open_unl = [p for p in unlimited if not p.get("eligibility")]
    gated = [p for p in unlimited if p.get("eligibility")]
    if open_unl:
        best = min(open_unl, key=lambda p: p["monthly_price"])
        lines.append("")
        lines.append(
            'RECOMMENDED ANSWER to "cheapest unlimited plan": LEAD with '
            f"{best['name']} at ${best['monthly_price']:.0f}/month — it is the "
            "cheapest unlimited plan available to every customer (no eligibility "
            "requirement)."
        )
        if gated:
            cheaper = [g for g in gated if g["monthly_price"] < best["monthly_price"]]
            if cheaper:
                noted = ", ".join(
                    f"{g['name']} (${g['monthly_price']:.0f}/month, "
                    f"requires {g['eligibility'].split(';')[0].strip().lower()})"
                    for g in sorted(cheaper, key=lambda p: p["monthly_price"])
                )
                lines.append(
                    f"Then note these are cheaper but require eligibility: {noted}."
                )
    return "\n".join(lines)


if __name__ == "__main__":
    print(build_pricing_reference())
