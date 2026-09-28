"""Mine embedding-based hard negatives for triplet training.

Replaces the random same-category negatives in ``build_triplets`` with genuine
nearest neighbours. The intuition matches the citation-reranking literature:
random negatives are already separated by the pretrained space, so they carry
almost no gradient, while a top-k neighbour is the example the model currently
gets wrong.

Training/query disjointness is enforced here, not left to chance. Papers that
serve as evaluation queries are dropped from training entirely, and each
anchor's own references are excluded from its negative pool. Without that
separation the retrieval numbers in ``eval_retrieval`` are meaningless.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Set

import numpy as np
import pandas as pd

from .eval_retrieval import (
    build_citation_queries,
    build_corpus,
    encode_corpus,
    load_metadata,
)
from .scidocs_data import paper_text


def mine_hard_negatives(
    embeddings: np.ndarray,
    corpus_ids: Sequence[str],
    pool_size: int = 100,
    exclude: Dict[str, Set[str]] | None = None,
    block: int = 512,
) -> Dict[str, List[str]]:
    """Return top-`pool_size` neighbours per paper, minus self and `exclude` hits.

    ``exclude`` maps a paper id to ids that must never appear as its negatives
    (its references, and any evaluation query).
    """
    count = len(corpus_ids)
    position = {pid: i for i, pid in enumerate(corpus_ids)}
    hard: Dict[str, List[str]] = {}
    # Rank in blocks to keep the score matrix at block x count instead of
    # count x count, which would be ~5 GB at this corpus size.
    for start in range(0, count, block):
        stop = min(start + block, count)
        scores = embeddings[start:stop] @ embeddings.T
        for offset, row_index in enumerate(range(start, stop)):
            pid = corpus_ids[row_index]
            scores[offset, row_index] = -np.inf
            blocked = exclude.get(pid) if exclude else None
            if blocked:
                for other in blocked:
                    other_index = position.get(other)
                    if other_index is not None:
                        scores[offset, other_index] = -np.inf
            top = np.argpartition(-scores[offset], pool_size)[:pool_size]
            top = top[np.argsort(-scores[offset, top], kind="stable")]
            hard[pid] = [corpus_ids[i] for i in top if i != row_index]
    return hard


def build_training_triplets(
    scidocs_dir: str = "data/scidocs",
    base_model: str = "allenai/scibert_scivocab_uncased",
    output_path: str = "data/hard_negative_triplets.csv",
    negatives_path: str = "data/hard_negatives.json",
    max_positives: int = 5,
    pool_size: int = 100,
    easy_negatives: int = 1,
    hard_negatives_per_anchor: int = 2,
    eval_queries: int = 300,
    seed: int = 42,
    max_seq_length: int = 256,
) -> pd.DataFrame:
    metadata = load_metadata(scidocs_dir)
    corpus_ids, texts = build_corpus(metadata)
    print(f"corpus={len(corpus_ids)}")

    # Reserve evaluation queries first, then ban them from training.
    held_out = build_citation_queries(metadata, corpus_ids, limit=eval_queries, seed=seed)
    banned: Set[str] = {q.paper_id for q in held_out}
    print(f"held_out_queries={len(held_out)}")

    embeddings = encode_corpus(
        base_model, texts, tag=f"base_{max_seq_length}", max_seq_length=max_seq_length
    )
    print("mined neighbour pools")

    in_corpus = set(corpus_ids)

    # A paper's negatives must exclude its own references, plus the whole
    # held-out query set (its neighbours would otherwise leak eval answers).
    reference_sets = {
        pid: {ref for ref in (metadata[pid].get("references") or []) if ref in in_corpus}
        for pid in corpus_ids
    }
    exclude = {pid: set(refs) | banned for pid, refs in reference_sets.items()}

    hard = mine_hard_negatives(embeddings, corpus_ids, pool_size=pool_size, exclude=exclude)

    rng = np.random.default_rng(seed)
    rows: List[Dict[str, str]] = []
    for pid in corpus_ids:
        if pid in banned:
            continue
        positives = [ref for ref in (metadata[pid].get("references") or []) if ref in in_corpus]
        if not positives:
            continue
        rng.shuffle(positives)
        for positive_id in positives[:max_positives]:
            if not paper_text(metadata[positive_id]):
                continue
            pool = [n for n in hard.get(pid, []) if n != positive_id]
            rng.shuffle(pool)
            negatives = pool[:hard_negatives_per_anchor]
            for _ in range(easy_negatives):
                negatives.append(corpus_ids[int(rng.integers(len(corpus_ids)))])
            for negative_id in negatives:
                if negative_id in reference_sets[pid] or negative_id in banned:
                    continue
                if not paper_text(metadata[negative_id]):
                    continue
                rows.append(
                    {
                        "anchor_id": pid,
                        "positive_id": positive_id,
                        "negative_id": negative_id,
                        "anchor_text": paper_text(metadata[pid]),
                        "positive_text": paper_text(metadata[positive_id]),
                        "negative_text": paper_text(metadata[negative_id]),
                    }
                )

    frame = pd.DataFrame(rows)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    Path(negatives_path).write_text(json.dumps(hard), encoding="utf-8")
    print(f"WROTE_TRIPLETS={len(frame)} path={output_path}")
    print(f"anchors={frame['anchor_id'].nunique() if len(frame) else 0}")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Mine hard negatives for triplet training")
    parser.add_argument("--scidocs-dir", default="data/scidocs")
    parser.add_argument("--base-model", default="allenai/scibert_scivocab_uncased")
    parser.add_argument("--output", default="data/hard_negative_triplets.csv")
    parser.add_argument("--negatives-cache", default="data/hard_negatives.json")
    parser.add_argument("--max-positives", type=int, default=5)
    parser.add_argument("--pool-size", type=int, default=100)
    parser.add_argument("--easy-negatives", type=int, default=1)
    parser.add_argument("--hard-per-anchor", type=int, default=2)
    parser.add_argument("--eval-queries", type=int, default=300)
    parser.add_argument("--max-seq-length", type=int, default=256)
    args = parser.parse_args()

    build_training_triplets(
        scidocs_dir=args.scidocs_dir,
        base_model=args.base_model,
        output_path=args.output,
        negatives_path=args.negatives_cache,
        max_positives=args.max_positives,
        pool_size=args.pool_size,
        easy_negatives=args.easy_negatives,
        hard_negatives_per_anchor=args.hard_per_anchor,
        eval_queries=args.eval_queries,
        max_seq_length=args.max_seq_length,
    )


if __name__ == "__main__":
    main()
