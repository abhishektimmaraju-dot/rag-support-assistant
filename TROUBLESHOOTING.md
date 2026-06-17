# Troubleshooting

Issues encountered while building and extending the NovaCell RAG chatbot, and
how each was resolved. Useful if you hit the same symptoms.

## Setup & runtime

- **Groq returns `401 Invalid API Key`**
  The key is missing or malformed in `.env`. Ensure the line is exactly
  `GROQ_API_KEY=gsk_...` — no quotes, no spaces, full key. Get/rotate one at
  [console.groq.com/keys](https://console.groq.com/keys).

- **Groq returns `429 rate_limit_exceeded`**
  Free-tier cap (6000 tokens/min), not a bug — usually from rapid repeated
  calls. Ask questions one at a time, or upgrade the Groq tier. Plan questions
  are token-heavy because the full catalog is sent as context.

- **Streamlit: `Port 8501 is not available`**
  A previous app instance still holds the port. Free it and relaunch:
  ```bash
  lsof -ti tcp:8501 | xargs kill -9
  uv run streamlit run app.py
  ```

## Knowledge & data

- **`data/plans.json` not found during ingest**
  The catalog must be in the `data/` folder before running `ingest_plans.py`.
  Confirm with `ls data/plans.json`, then `uv run ingest_plans.py`.

## Answer quality

- **"Cheapest plan" answers were wrong or incomplete**
  Cause: only the top-3 plans were retrieved, so the model never saw the full
  lineup. Fix: the `plans` collection is retrieved in full (`PLANS_TOP_K` in
  `config.py`) so comparison questions see every option.

- **Roaming answer invented a "$15/day EU Bundle"**
  Cause: it wasn't invented — it was a *stale FAQ entry* conflicting with the
  new catalog. Fix: a prompt rule in `chain.py` makes `[PLANS]` authoritative
  over older FAQ/guide content. (Long-term: clean the stale FAQ row and re-run
  `ingest_faq.py`.)

- **Roaming cost comparison was backwards** (called the $70 option "cheaper"
  than the $50 one)
  Fix: prompt rule to compute each option's total for the trip length and
  recommend the genuinely lower-cost one.

- **Same question gave different answers across runs ($30 / $45 / $60)**
  Cause: hosted LLMs are not reproducible even at `temperature=0`, and the model
  was ranking prices itself. Fix: `plan_facts.py` computes the price ranking
  deterministically in Python and injects a fixed "cheapest" recommendation into
  the context — the model reads the answer instead of deriving it. Now stable.

- **Bot answered an off-topic request (e.g. a holiday itinerary) as if it were
  telecom**
  Cause: it's a telecom-only bot with no tourism data, and "3-day plan for India"
  is semantically close to its roaming content, so it assumed telecom intent.
  Fix: a SCOPE rule in the prompt makes it open with "I can only help with
  NovaCell telecom topics" for off-topic requests, then optionally offer a
  relevant telecom angle (e.g. roaming) without inventing the off-topic content.

- **Prices rendered as italic math (e.g. `$70 ... $100`)**
  Cause: Streamlit renders `$...$` as a LaTeX equation. Fix: `app.py` escapes
  `$` as `\$` in the streamed output.

## General principle

When an answer depends on **sorting, math, or comparison**, do it in code and
hand the model the result. Reserve the LLM for phrasing and reasoning over text,
not for deterministic computation.
