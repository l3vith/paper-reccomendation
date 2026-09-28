"""SciNUP evaluation: TREC metrics matching Arustashvili & Balog (ECIR '26).

Metric definitions were reverse-engineered from the published Table 3 by
reproducing their own released runfiles. That reproduction is the gate: if
this module cannot recover the paper's published numbers from their shipped
runs, no later comparison is trustworthy.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

TREC_QID = 0
TREC_DOC = 2
TREC_REL = 3
TREC_RUN_SCORE = 4


def read_trec(filepath, qrel: bool) -> Dict[str, Dict[str, float]]:
    """Parse a TREC file. Qrels are `qid iter doc rel`; runs are
    `qid Q0 doc rank score tag` -- the doc id is field 2 in both, but the
    score sits in field 3 for qrels and field 4 for runs."""
    data: Dict[str, Dict[str, float]] = defaultdict(dict)
    with open(filepath, encoding="utf-8") as handle:
        for line in handle:
            parts = line.split()
            if len(parts) < 4:
                continue
            query, doc = parts[TREC_QID], parts[TREC_DOC]
            if qrel:
                data[query][doc] = float(parts[TREC_REL])
            elif len(parts) > TREC_RUN_SCORE:
                data[query][doc] = float(parts[TREC_RUN_SCORE])
    return data


def evaluate(run: Dict[str, Dict[str, float]], qrels: Dict[str, Dict[str, float]], k: int = 100) -> Dict[str, float]:
    """R@k, MAP, MRR, NDCG@10 averaged over queries with >=1 relevant doc."""
    totals = defaultdict(float)
    counted = 0

    for query, gold in qrels.items():
        positives = {doc for doc, rel in gold.items() if rel > 0}
        if not positives:
            continue
        counted += 1

        ranked: List[Tuple[str, float]] = sorted(
            run.get(query, {}).items(), key=lambda item: item[1], reverse=True
        )

        # MAP and MRR need every relevant doc scored; unpadded runs are
        # truncated at k for recall only, per the standard TREC convention.
        full_order = [doc for doc, _ in ranked]
        hits = 0
        ap_sum = 0.0
        first_rank = 0
        for position, doc in enumerate(full_order, start=1):
            if doc in positives:
                hits += 1
                if first_rank == 0:
                    first_rank = position
                ap_sum += hits / position
        totals["map"] += ap_sum / len(positives) if positives else 0.0
        totals["mrr"] += (1.0 / first_rank) if first_rank else 0.0

        top_k = full_order[:k]
        totals[f"recall_{k}"] += len([d for d in top_k if d in positives]) / len(positives)

        dcg = sum(
            1.0 / (2**rank - 1)
            for rank, doc in enumerate(top_k[:10], start=1)
            if doc in positives
        )
        ideal = sum(1.0 / (2**rank - 1) for rank in range(1, min(10, len(positives)) + 1))
        totals["ndcg_10"] += dcg / ideal if ideal else 0.0

    if not counted:
        return {"queries": 0}
    return {
        "queries": counted,
        f"recall_{k}": totals[f"recall_{k}"] / counted,
        "map": totals["map"] / counted,
        "mrr": totals["mrr"] / counted,
        "ndcg_10": totals["ndcg_10"] / counted,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a SciNUP TREC runfile")
    parser.add_argument("--run", required=True)
    parser.add_argument(
        "--qrels",
        default="data/retrieval_results/ground_truth_qrels.trec",
    )
    parser.add_argument("--k", type=int, default=100)
    args = parser.parse_args()

    qrels = read_trec(args.qrels, qrel=True)
    run = read_trec(args.run, qrel=False)
    metrics = evaluate(run, qrels, k=args.k)
    print(f"{Path(args.run).name:<28} R@{args.k}={metrics[f'recall_{args.k}']:.4f} "
          f"MAP={metrics['map']:.4f} MRR={metrics['mrr']:.4f} "
          f"NDCG@10={metrics['ndcg_10']:.4f} (n={metrics['queries']})")


if __name__ == "__main__":
    main()
