"""Golden-question evaluation harness.

Runs a fixed set of questions through the chain and checks each answer for
required and forbidden content. This turns "ask it a few times and eyeball it"
into an objective, repeatable pass/fail — so we can compare the current design
against the routed re-architecture on the same yardstick.

Checks are deliberately loose (case-insensitive substrings): the LLM's wording
varies, so we assert on facts that must (or must not) appear, not exact text.

Usage:
    uv run eval.py
"""

from __future__ import annotations

import time

from chain import answer
from config import require_groq_key

# Each case: question + facts that must appear (all of) + facts that must not.
CASES = [
    {
        "id": "cheapest-unlimited",
        "q": "What's your cheapest unlimited plan?",
        "must_include": ["Prepaid Unlimited", "$45"],
        "must_exclude": [],
        "note": "Lead with the $45 no-eligibility plan",
    },
    {
        "id": "family-plan",
        "q": "Do you have a family plan?",
        "must_include": ["Family Unlimited", "$140"],
        "must_exclude": [],
        "note": "4 lines for $140",
    },
    {
        "id": "roaming-europe",
        "q": "I'm going to Europe for about a week. What are my roaming options?",
        "must_include": ["Day Pass", "Monthly Pass", "$50"],
        "must_exclude": ["EU Roaming Bundle"],
        "note": "Day Pass + Monthly Pass; no stale EU bundle",
    },
    {
        "id": "student-discount",
        "q": "Is there a discount for students?",
        "must_include": ["Student", "$30"],
        "must_exclude": [],
        "note": "Student Unlimited $30 with eligibility",
    },
    {
        "id": "calling-pack-vs-roaming",
        "q": "What's the difference between the International Calling Pack and a roaming pass?",
        "must_include": ["Calling Pack", "$15"],
        "must_exclude": [],
        "note": "Distinguish from-US calls vs usage abroad",
    },
    {
        "id": "regression-slow-data",
        "q": "My mobile data is really slow. How do I fix it?",
        "must_include": ["airplane"],
        "must_exclude": [],
        "note": "Original troubleshooting still works",
    },
    {
        "id": "refusal-bill",
        "q": "What is my current bill and balance?",
        "must_include": ["611"],
        "must_exclude": [],
        "note": "Redirect to account channels, no invented balance",
    },
    {
        "id": "scope-india-holiday",
        "q": "I am visiting India, can you please build a 3 day holiday plan",
        "must_include": ["telecom"],
        "must_exclude": [],
        "note": "Off-topic: scope redirect",
    },
    {
        "id": "scope-weather",
        "q": "What is the weather in Mumbai?",
        "must_include": ["telecom"],
        "must_exclude": [],
        "note": "Off-topic: scope redirect",
    },
]

SLEEP_BETWEEN = 18  # seconds, to respect the Groq free-tier rate limit


def _ask_with_retry(question: str, retries: int = 2) -> str:
    for attempt in range(retries + 1):
        try:
            return answer(question)
        except Exception as exc:  # most often a 429 rate-limit
            if attempt == retries:
                raise
            wait = 30
            print(f"    (retry after error: {type(exc).__name__}; waiting {wait}s)")
            time.sleep(wait)
    return ""


def run() -> int:
    require_groq_key()
    passed = 0
    failed_ids = []

    for i, case in enumerate(CASES):
        text = _ask_with_retry(case["q"])
        low = text.lower()

        missing = [s for s in case["must_include"] if s.lower() not in low]
        present = [s for s in case["must_exclude"] if s.lower() in low]
        ok = not missing and not present

        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {case['id']} — {case['note']}")
        if missing:
            print(f"        missing: {missing}")
        if present:
            print(f"        should not appear: {present}")

        if ok:
            passed += 1
        else:
            failed_ids.append(case["id"])

        if i < len(CASES) - 1:
            time.sleep(SLEEP_BETWEEN)

    print("\n" + "=" * 50)
    print(f"RESULT: {passed}/{len(CASES)} passed")
    if failed_ids:
        print(f"Failed: {', '.join(failed_ids)}")
    return passed


if __name__ == "__main__":
    run()
