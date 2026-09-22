"""Evaluate a paper encoder on the untouched SciDocs recommendation split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from src.model.scidocs_data import paper_text


def evaluate(model_path: str, scidocs_dir: str, split: str) -> dict[str, float | int]:
    root = Path(scidocs_dir)
    with (root / "recomm" / "paper_metadata.json").open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    events = pd.read_csv(root / "recomm" / f"{split}.csv")

    paper_ids: set[str] = set(events["pid"].astype(str))
    for event in events.itertuples(index=False):
        paper_ids.update(str(event.similar_papers_avail_at_time).split(","))
    ordered_ids = sorted(pid for pid in paper_ids if paper_text(metadata.get(pid, {})))

    model = SentenceTransformer(model_path)
    embeddings = model.encode(
        [paper_text(metadata[pid]) for pid in ordered_ids],
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    vectors = dict(zip(ordered_ids, embeddings, strict=True))

    reciprocal_ranks: list[float] = []
    hits_at_1 = 0
    hits_at_5 = 0
    for event in events.itertuples(index=False):
        query_id = str(event.pid)
        positive_id = str(event.clicked_pid)
        candidate_ids = [pid for pid in str(event.similar_papers_avail_at_time).split(",") if pid in vectors]
        if query_id not in vectors or positive_id not in candidate_ids:
            continue
        scores = np.asarray([float(vectors[query_id] @ vectors[pid]) for pid in candidate_ids])
        ranked_ids = [candidate_ids[index] for index in np.argsort(-scores)]
        rank = ranked_ids.index(positive_id) + 1
        reciprocal_ranks.append(1.0 / rank)
        hits_at_1 += rank <= 1
        hits_at_5 += rank <= 5

    count = len(reciprocal_ranks)
    return {
        "events": count,
        "mrr": float(np.mean(reciprocal_ranks)),
        "hit@1": hits_at_1 / count,
        "hit@5": hits_at_5 / count,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/scibert-finetuned-papers")
    parser.add_argument("--scidocs-dir", default="data/scidocs")
    parser.add_argument("--split", default="test", choices=("val", "test"))
    parser.add_argument("--output", default="results/scidocs_after_training.json")
    args = parser.parse_args()
    metrics = evaluate(args.model, args.scidocs_dir, args.split)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))
