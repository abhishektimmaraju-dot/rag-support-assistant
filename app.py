"""Streamlit chat UI for the Telecom RAG chatbot.

Implements FR-01..FR-05: free-text questions, clickable sample questions,
session-scoped history, clear-conversation, and token-by-token streaming.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

from chain import stream_answer
from config import CHROMA_DIR, GROQ_API_KEY

SAMPLE_QUESTIONS = [
    "What's your cheapest unlimited plan?",
    "Do you have a family plan?",
    "I'm going to Europe for a week — what are my roaming options?",
    "Is there a discount for students?",
    "Why is my mobile internet so slow?",
    "My SIM is not recognised, what should I do?",
]

st.set_page_config(page_title="MyTelecom Support", page_icon="📶", layout="centered")


def _ensure_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []  # list[dict(role, content)]


def _handle_question(question: str) -> None:
    """Append the user message and stream the assistant's grounded reply."""
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            # Escape '$' so Streamlit doesn't render "$70 ... $100" as LaTeX math.
            escaped = (tok.replace("$", "\\$") for tok in stream_answer(question))
            response = st.write_stream(escaped)
        except Exception as exc:
            response = (
                "Sorry, something went wrong reaching the assistant. "
                f"Please try again, or call 611.\n\n_Error: {exc}_"
            )
            st.error(response)
    st.session_state.messages.append({"role": "assistant", "content": response})


def main() -> None:
    _ensure_state()

    # --- Sidebar -------------------------------------------------------------
    with st.sidebar:
        st.title("📶 MyTelecom")
        st.caption("Self-serve customer-care assistant")

        if not GROQ_API_KEY:
            st.error("GROQ_API_KEY missing. Add it to your .env file.")
        if not CHROMA_DIR.exists():
            st.warning(
                "Vector store not found. Run `python ingest_all.py` first."
            )

        st.subheader("Try a question")
        clicked = None
        for q in SAMPLE_QUESTIONS:
            if st.button(q, use_container_width=True, key=f"sample_{q}"):
                clicked = q

        st.divider()
        if st.button("🗑️ Clear conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    # --- Main pane -----------------------------------------------------------
    st.title("How can I help?")
    st.caption(
        "I answer using MyTelecom's published support knowledge only. "
        "For account-specific help, call 611 or use the MyTelecom app."
    )

    # Replay history (FR-03).
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Inputs: sample-question click (FR-02) or free text (FR-01).
    typed = st.chat_input("Ask about data, billing, roaming, SIM…")
    question = clicked or typed
    if question:
        _handle_question(question)


if __name__ == "__main__":
    main()
