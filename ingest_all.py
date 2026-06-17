"""Run all ingest scripts in one pass.

Convenience entry point so a support operator can rebuild the entire vector
store with a single command.
"""

from __future__ import annotations

import ingest_faq
import ingest_guides
import ingest_plans
import ingest_tickets


def main() -> None:
    print("Building vector store in chroma_store/ ...")
    faq = ingest_faq.ingest()
    tickets = ingest_tickets.ingest()
    guides = ingest_guides.ingest()
    plans = ingest_plans.ingest()
    total = faq + tickets + guides + plans
    print(
        f"\nDone. {faq} FAQ + {tickets} tickets + {guides} guide chunks "
        f"+ {plans} plans = {total} documents indexed."
    )


if __name__ == "__main__":
    main()
