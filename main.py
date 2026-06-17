"""Interactive CLI REPL for non-browser use (FR-18, FR-19).

Usage:
    python main.py        (or: uv run main.py)

Type a question and press Enter. Answers stream token-by-token. Type `quit`
(or `exit`) to leave.
"""

from __future__ import annotations

import sys

from chain import stream_answer
from config import require_groq_key

BANNER = """\
==================================================
 MyTelecom Customer-Care Assistant  (CLI)
 Ask about data, connectivity, roaming, SIM,
 billing, voice, device, or account issues.
 Type 'quit' to exit.
==================================================
"""

EXIT_WORDS = {"quit", "exit"}


def main() -> None:
    try:
        require_groq_key()
    except RuntimeError as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    print(BANNER)
    while True:
        try:
            question = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not question:
            continue
        if question.lower() in EXIT_WORDS:
            print("Goodbye.")
            break

        print("Bot> ", end="", flush=True)
        try:
            for token in stream_answer(question):
                print(token, end="", flush=True)
        except Exception as exc:  # surface API/runtime errors without crashing the REPL
            print(f"\n[error] {exc}")
        print("\n")


if __name__ == "__main__":
    main()
