"""PROTOTYPE — structured (text-to-SQL) query path for plans, at scale.

The shipped pricing route dumps the whole plan catalog into the prompt. That is
fine for 14 plans and impossible for 5,000. This prototype shows the scalable
alternative: keep plans in a structured store, let the LLM translate the
question into a read-only SQL query, execute it deterministically, and feed only
the matching rows back.

Crucially, the context the LLM sees is the SCHEMA (tiny, fixed) — never the
catalog. So it costs the same whether there are 14 plans or 50,000.

This is a standalone demo (not wired into the app). Run:
    uv run prototype_plan_query.py
"""

from __future__ import annotations

import json
import random
import re
import sqlite3

from langchain_groq import ChatGroq

from config import LLM_MODEL, PLANS_JSON, require_groq_key

# Columns we expose to the model. Keep this list and the prompt schema in sync.
COLUMNS = [
    "name", "type", "category", "monthly_price", "data_unlimited",
    "data_gb", "hotspot_gb", "network", "contract", "eligibility",
    "intro_offer", "lines_included", "best_for",
]

SCHEMA_FOR_LLM = """Table: plans
Columns:
  name            TEXT
  type            TEXT    -- prepaid | postpaid | data-only | add-on
  category        TEXT    -- prepaid | postpaid | family | specialty | roaming | add-on
  monthly_price   REAL    -- USD per month; NULL for add-ons priced per-day/per-use
  data_unlimited  INTEGER -- 1 if unlimited data, else 0
  data_gb         REAL    -- high-speed data allowance in GB; NULL if unlimited
  hotspot_gb      REAL
  network         TEXT
  contract        TEXT
  eligibility     TEXT    -- NULL means available to everyone; otherwise a requirement
  intro_offer     TEXT
  lines_included  INTEGER
  best_for        TEXT"""


def _row_from_plan(plan: dict) -> tuple:
    return tuple(
        1 if plan.get("data_unlimited") else 0 if c == "data_unlimited" else plan.get(c)
        for c in COLUMNS
    )


def build_db(n_synthetic: int = 0) -> sqlite3.Connection:
    """In-memory SQLite of the real catalog, optionally padded with synthetic plans."""
    conn = sqlite3.connect(":memory:")
    cols = ", ".join(f"{c} {'INTEGER' if c in ('data_unlimited','lines_included') else 'REAL' if c in ('monthly_price','data_gb','hotspot_gb') else 'TEXT'}" for c in COLUMNS)
    conn.execute(f"CREATE TABLE plans ({cols})")

    with open(PLANS_JSON, encoding="utf-8") as f:
        plans = json.load(f)["plans"]
    rows = [_row_from_plan(p) for p in plans]

    # Synthetic plans to demonstrate scale (deterministic via fixed seed).
    rng = random.Random(42)
    types = ["prepaid", "postpaid", "data-only"]
    for i in range(n_synthetic):
        unlimited = rng.random() < 0.4
        rows.append((
            f"Synthetic Plan {i}", rng.choice(types), rng.choice(["prepaid", "postpaid", "specialty"]),
            round(rng.uniform(10, 120), 0), 1 if unlimited else 0,
            None if unlimited else rng.choice([5, 10, 20, 30, 50]),
            rng.choice([0, 5, 10, 25, 50]), rng.choice(["4G LTE", "5G"]),
            "monthly, cancel anytime", rng.choice([None, None, None, "Students only"]),
            None, rng.choice([1, 1, 1, 2, 4]), "synthetic",
        ))

    conn.executemany(
        f"INSERT INTO plans ({', '.join(COLUMNS)}) VALUES ({', '.join(['?'] * len(COLUMNS))})",
        rows,
    )
    conn.commit()
    return conn


_llm = None


def _get_llm() -> ChatGroq:
    global _llm
    if _llm is None:
        require_groq_key()
        _llm = ChatGroq(model=LLM_MODEL, temperature=0, reasoning_format="parsed")
    return _llm


def text_to_sql(question: str) -> str:
    """Ask the LLM to translate the question into a single SQLite SELECT."""
    prompt = (
        "You translate a question into ONE read-only SQLite query over this schema.\n\n"
        f"{SCHEMA_FOR_LLM}\n\n"
        "Rules: output ONLY the SQL, no prose, no code fences. Use a single "
        "SELECT (or WITH). Remember eligibility IS NULL means 'available to "
        "everyone'. Limit results sensibly.\n\n"
        f"Question: {question}\nSQL:"
    )
    raw = _get_llm().invoke(prompt).content.strip()
    # Strip accidental ```sql fences and a leading "SQL:" label.
    cleaned = re.sub(r"^```(?:sql)?|```$", "", raw, flags=re.MULTILINE).strip()
    return re.sub(r"(?is)^\s*sql\s*:\s*", "", cleaned).strip()


_FORBIDDEN = re.compile(r"\b(insert|update|delete|drop|alter|create|replace|attach|pragma)\b", re.I)


def is_safe(sql: str) -> bool:
    """Allow a single read-only SELECT/WITH statement only."""
    s = sql.strip().rstrip(";")
    if ";" in s:  # reject multiple statements
        return False
    if not re.match(r"(?is)^\s*(select|with)\b", s):
        return False
    return _FORBIDDEN.search(s) is None


def run(question: str, conn: sqlite3.Connection) -> None:
    print("\n" + "=" * 72)
    print(f"Q: {question}")
    sql = text_to_sql(question)
    print(f"\nGenerated SQL:\n  {sql}")
    if not is_safe(sql):
        print("\n  REJECTED — not a safe read-only query.")
        return
    cur = conn.execute(sql)
    headers = [d[0] for d in cur.description]
    rows = cur.fetchall()
    print(f"\nRows returned: {len(rows)} (of {conn.execute('SELECT COUNT(*) FROM plans').fetchone()[0]} plans in the table)")
    for r in rows[:5]:
        print("  " + " | ".join(f"{h}={v}" for h, v in zip(headers, r) if v is not None))


def main() -> None:
    n = 5000
    conn = build_db(n_synthetic=n)
    total = conn.execute("SELECT COUNT(*) FROM plans").fetchone()[0]
    print(f"Built a plans table with {total} rows (14 real + {n} synthetic).")
    print("Context the LLM sees per question = the schema only (~120 tokens),")
    print("regardless of catalog size. 'Dump everything' would send all "
          f"{total} rows.")

    for q in [
        "What is the cheapest unlimited plan with no eligibility requirement?",
        "List postpaid plans under $50 with unlimited data, cheapest first.",
        "How many plans include more than 20 GB of hotspot?",
    ]:
        run(q, conn)


if __name__ == "__main__":
    main()
