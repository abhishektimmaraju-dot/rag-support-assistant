"""Routed RAG chain: classify -> focused retrieve -> prompt -> Qwen3-32B -> stream.

The router (router.py) decides how to answer before retrieval:
- "pricing": deterministic pricing reference + plans-only context
- "telecom": document retrieval over FAQ / tickets / guide
- "other":   a deterministic refusal — no LLM call at all

The system prompt still hard-constrains the model to answer ONLY from the
retrieved context (FR-10..13); scope enforcement now lives in the router rather
than relying on the model to police itself.
"""

from __future__ import annotations

from collections.abc import Iterator

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from config import LLM_MODEL, LLM_TEMPERATURE, require_groq_key
from hybrid import retrieve_telecom
from plan_facts import build_pricing_reference
from retriever import format_docs, get_retriever
from router import classify

SYSTEM_PROMPT = """You are NovaCell's customer-care assistant. You help \
subscribers resolve Tier-1 support issues (connectivity, mobile data, roaming, \
SIM, billing, voice, device, account) and answer questions about our published \
plans and pricing.

PLAN & PRICING RULES (apply when the CONTEXT contains plan/pricing information):
- [PLANS] is the authoritative source for all plan, add-on, bundle, and pricing \
information. Only mention a plan, add-on, or price that appears VERBATIM in the \
CONTEXT — never invent a plan/bundle name or a price.
- A PLAN PRICING REFERENCE block (pre-sorted, authoritative) may be provided. \
For any "cheapest", "best", or price-comparison question, use its rankings and \
its stated "cheapest …" facts DIRECTLY — do not re-rank or recompute prices. \
Treat a plan as "unlimited" whenever it is listed with unlimited data; a \
fair-use/congestion note does not disqualify it.
- For roaming/travel questions, the valid travel options are the priced roaming \
add-ons in [PLANS] (e.g. the Day Pass and Monthly Pass). When the trip length \
is given, compute each option's total cost and recommend the lower-cost one. \
Never describe a higher-cost option as "cheaper" than a lower-cost one.

STRICT RULES:
1. Answer using ONLY the information in the CONTEXT below. Do not use outside or \
prior knowledge about telecom services, pricing, or policy.
2. If the CONTEXT does not contain enough information to answer confidently, say \
so plainly and tell the user to call 611 or use the MyTelecom app. Do not guess \
or invent steps, prices, codes, or policies.
3. Be concise and practical. Present steps as a short numbered list.
4. Do not mention the words "context", "documents", "FAQ", or "tickets" — just \
help the customer directly.

CONTEXT:
{context}"""

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ]
)

OUT_OF_SCOPE_MESSAGE = (
    "I can only help with NovaCell telecom topics — questions about your plans "
    "and pricing, connectivity, mobile data, roaming, SIM, billing, voice, "
    "device, or account. For anything else, I'm not the right resource. If you "
    "have a telecom question, I'm happy to help."
)

# Collections used by the pricing route (telecom route uses hybrid.py).
PLANS_LABELS = ["PLANS"]


def _build_llm() -> ChatGroq:
    require_groq_key()  # fail fast with a clear message if the key is missing
    return ChatGroq(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        reasoning_format="parsed",  # keep <think> reasoning out of the answer
    )


_llm_chain = None


def _get_llm_chain():
    """Lazily-built `prompt | llm | parser` chain (takes {context, question})."""
    global _llm_chain
    if _llm_chain is None:
        _llm_chain = PROMPT | _build_llm() | StrOutputParser()
    return _llm_chain


def _build_context(question: str, route: str) -> str:
    """Assemble route-specific context — focused, not everything-at-once."""
    if route == "pricing":
        # Deterministic ranking first, then plans-only docs (no FAQ/guide noise).
        plans_ctx = get_retriever().format_context(question, labels=PLANS_LABELS)
        return f"[PLANS]\n{build_pricing_reference()}\n\n---\n\n{plans_ctx}"
    # telecom: hybrid (dense + BM25) retrieval over the document collections
    return format_docs(retrieve_telecom(question))


def answer(question: str) -> str:
    """Return a full grounded answer (non-streaming)."""
    route = classify(question)
    if route == "other":
        return OUT_OF_SCOPE_MESSAGE
    context = _build_context(question, route)
    return _get_llm_chain().invoke({"context": context, "question": question})


def stream_answer(question: str) -> Iterator[str]:
    """Yield the grounded answer token-by-token (FR-05)."""
    route = classify(question)
    if route == "other":
        # Deterministic refusal — no LLM call. Streamed in word chunks for the UI.
        for word in OUT_OF_SCOPE_MESSAGE.split(" "):
            yield word + " "
        return
    context = _build_context(question, route)
    yield from _get_llm_chain().stream({"context": context, "question": question})
