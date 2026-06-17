"""Query router — classifies each question before retrieval.

Routing the query first is the key architectural improvement: instead of one
overloaded prompt asking the LLM to rank prices, suppress stale data, and judge
scope all at once, we decide *how* to answer up front and send the question
down a focused path.

Routes:
- "pricing"  -> plan / price / cost / cheapest / roaming-pass / discount questions
- "telecom"  -> other NovaCell support (connectivity, SIM, billing, setup, etc.)
- "other"    -> not about NovaCell telecom (handled with a deterministic refusal)
"""

from __future__ import annotations

from functools import lru_cache

from langchain_groq import ChatGroq

from config import LLM_MODEL, require_groq_key

ROUTES = ("pricing", "telecom", "other")

ROUTER_SYSTEM = """You classify a customer's message for a telecom assistant. \
Reply with EXACTLY ONE word — one of: pricing, telecom, other. No explanation.

- pricing: anything about plans, prices, cost, "cheapest/best plan", comparing \
plans, roaming passes or charges, add-ons, data/hotspot allowances by plan, \
student/family/senior discounts.
- telecom: other NovaCell support that is not primarily about price — slow data, \
no signal, SIM issues, billing problems, how to activate roaming, voicemail, \
device setup, account/login help.
- other: anything not about NovaCell telecom at all — travel itineraries, \
holiday plans, weather, sports, general knowledge, coding, etc.

Answer with one word only."""


@lru_cache(maxsize=1)
def _router_llm() -> ChatGroq:
    require_groq_key()
    # Low temperature for stable routing; parsed reasoning so .content is the label.
    return ChatGroq(model=LLM_MODEL, temperature=0, reasoning_format="parsed")


def classify(query: str) -> str:
    """Return one of ROUTES. Defaults to 'telecom' on any ambiguity."""
    try:
        raw = _router_llm().invoke(
            [("system", ROUTER_SYSTEM), ("human", query)]
        ).content.lower()
    except Exception:
        # If the router call fails, fall back to the safe in-domain path.
        return "telecom"

    # Pick the first route keyword that appears in the reply.
    for route in ("pricing", "other", "telecom"):
        if route in raw:
            return route
    return "telecom"
