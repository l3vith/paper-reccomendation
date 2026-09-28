"""LoRA fine-tune SPECTER2 on citation hard negatives (SciNUP push)."""

from __future__ import annotations

import argparse
import os

import pandas as pd
from sentence_transformers import InputExample

from .train import create_model, train
from .train_hardneg import load_triplet_examples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--triplets", default="data/hard_negative_triplets.csv")
    parser.add_argument("--output", default="models/specter2-finetuned-citations")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-seq-length", type=int, default=128)
    parser.add_argument("--limit", type=int, default=15000)
    args = parser.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

    import pandas as pd

    frame = pd.read_csv(args.triplets)
    if args.limit and len(frame) > args.limit:
        frame = frame.sample(n=args.limit, random_state=42).reset_index(drop=True)
    from sentence_transformers import InputExample
    examples = [
        InputExample(texts=[r.anchor_text, r.positive_text, r.negative_text])
        for r in frame.itertuples()
    ]
    print(f"examples={len(examples)}", flush=True)
    model = create_model(
        model_name="allenai/specter2_base",
        max_seq_length=args.max_seq_length,
        use_qlora=False,
        use_lora=True,
        pooling="mean",
    )
    train(
        model, examples, output_path=args.output,
        epochs=args.epochs, batch_size=args.batch_size,
        learning_rate=args.lr,
    )
    print(f"TRAINING_COMPLETE={args.output}", flush=True)


if __name__ == "__main__":
    main()
