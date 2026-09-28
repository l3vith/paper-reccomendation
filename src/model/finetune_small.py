"""Light fine-tune of a small open encoder on citation hard negatives.

No architecture surgery: load a stock SentenceTransformer, set max seq
length, train with MultipleNegativesRankingLoss, save. Full fine-tune is
cheap here (~22M params for MiniLM) so no LoRA adapters needed.
"""

from __future__ import annotations

import argparse
import os

import pandas as pd
from sentence_transformers import InputExample, SentenceTransformer, losses
from torch.utils.data import DataLoader


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--triplets", default="data/hard_negative_triplets.csv")
    parser.add_argument("--output", default="models/minilm-finetuned-citations")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-seq-length", type=int, default=192)
    parser.add_argument("--limit", type=int, default=15000)
    args = parser.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

    df = pd.read_csv(args.triplets)
    if args.limit and len(df) > args.limit:
        df = df.sample(n=args.limit, random_state=42).reset_index(drop=True)
    print(f"triplets={len(df)}", flush=True)
    examples = [
        InputExample(texts=[r.anchor_text, r.positive_text, r.negative_text])
        for r in df.itertuples()
    ]

    model = SentenceTransformer(args.base)
    model.max_seq_length = args.max_seq_length
    loader = DataLoader(examples, batch_size=args.batch_size, shuffle=True)
    loss = losses.MultipleNegativesRankingLoss(model=model, scale=20.0)
    model.fit(
        train_objectives=[(loader, loss)],
        epochs=args.epochs,
        warmup_steps=100,
        optimizer_params={"lr": args.lr},
        show_progress_bar=True,
    )
    model.save(args.output)
    print(f"TRAINING_COMPLETE={args.output}", flush=True)


if __name__ == "__main__":
    main()
