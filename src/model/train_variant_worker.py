"""Isolated process entry point for one SciBERT variant training run."""

from __future__ import annotations

import argparse
import logging

from .experiments import fine_tune_variant
from .variants import MODEL_VARIANTS


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and index one SciBERT study variant")
    parser.add_argument("--variant", choices=list(MODEL_VARIANTS), required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--no-qlora", action="store_true")
    parser.add_argument("--steps-per-epoch", type=int, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    path = fine_tune_variant(
        variant_key=args.variant,
        epochs=max(1, args.epochs),
        batch_size=max(1, args.batch_size),
        use_qlora=not args.no_qlora,
        # Let SentenceTransformer and its Trainer agree on MPS vs CPU. The
        # worker is isolated from Streamlit, so using MPS here no longer piles
        # training allocations on top of the app's inference models.
        device=None,
        steps_per_epoch=args.steps_per_epoch,
    )
    print(f"TRAINING_COMPLETE={path}")


if __name__ == "__main__":
    main()
