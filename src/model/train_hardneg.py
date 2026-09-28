"""Fine-tune SciBERT pooling variants on hard-negative citation triplets.

Thin wrapper over ``train.create_model`` / ``train.train`` that points at the
hard-negative triplet file and skips FAISS index building. ``eval_retrieval``
scores by dense matrix product, so the on-disk index is not needed for the
study and building it per variant would only add time.
"""

from __future__ import annotations

import argparse
import gc
import os
from typing import List

import pandas as pd
from sentence_transformers import InputExample

from .train import create_model, train
from .variants import MODEL_VARIANTS, get_variant

TRIPLET_COLUMNS = ("anchor_text", "positive_text", "negative_text")


def load_triplet_examples(
    triplets_path: str,
    anchor_column: str = "anchor_id",
    limit: int = 0,
) -> List[InputExample]:
    """Load triplets, splitting by anchor id so no query spans train and test.

    ``anchor_id`` is the paper id recorded by the miner. Splitting on text
    instead would break whenever two distinct papers share an abstract.
    """
    frame = pd.read_csv(triplets_path)
    missing = [column for column in TRIPLET_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{triplets_path} is missing columns: {missing}")

    if anchor_column in frame.columns:
        groups = frame[anchor_column].dropna().unique().tolist()
    else:
        groups = frame["anchor_text"].drop_duplicates().tolist()

    if limit and len(groups) > limit:
        groups = groups[:limit]
        frame = frame[frame[anchor_column].isin(set(groups))] if anchor_column in frame.columns else frame

    examples = [
        InputExample(texts=[row.anchor_text, row.positive_text, row.negative_text])
        for row in frame.itertuples(index=False)
    ]
    return examples


def fine_tune(
    variant_key: str,
    triplets_path: str = "data/hard_negative_triplets.csv",
    epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    use_qlora: bool = True,
    base_model: str = "allenai/scibert_scivocab_uncased",
    max_seq_length: int = 256,
    device: str | None = None,
    limit: int = 0,
) -> str:
    variant = get_variant(variant_key)
    examples = load_triplet_examples(triplets_path, limit=limit)
    if not examples:
        raise ValueError(f"No triplets loaded from {triplets_path}")

    model = create_model(
        model_name=base_model,
        max_seq_length=max_seq_length,
        use_qlora=use_qlora,
        use_lora=True,
        pooling=variant.pooling,
        device=device,
    )
    train(
        model,
        examples,
        output_path=variant.model_path,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )
    del model
    gc.collect()
    return variant.model_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune a variant on hard negatives")
    parser.add_argument("--variant", choices=list(MODEL_VARIANTS), required=True)
    parser.add_argument("--triplets", default="data/hard_negative_triplets.csv")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--no-qlora", action="store_true")
    parser.add_argument("--base-model", default="allenai/scibert_scivocab_uncased")
    parser.add_argument("--max-seq-length", type=int, default=256)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    path = fine_tune(
        variant_key=args.variant,
        triplets_path=args.triplets,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        use_qlora=not args.no_qlora,
        base_model=args.base_model,
        max_seq_length=args.max_seq_length,
        limit=args.limit,
    )
    print(f"TRAINING_COMPLETE={path}")


if __name__ == "__main__":
    main()
