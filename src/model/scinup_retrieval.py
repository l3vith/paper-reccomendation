"""SciNUP retrieval: BM25, dense, and hybrid arms over NL user profiles.

Mirrors Arustashvili & Balog (ECIR '26) exactly where it matters:
  * candidate document text is ``Title: {title} Abstract: {abstract}``
  * the query is the raw NL profile string
  * output is a TREC runfile scored by ``eval_scinup``

Each author carries its own 1,000-paper candidate pool, so scoring is done
per author and never against a shared global index.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np


def doc_text(article: dict) -> str:
    """Match the SciNUP index document format exactly."""
    return f"Title: {article['title'].strip()} Abstract: {article['abstract'].strip()}\n\n"


def load_dataset(path: str, split: int | None = None) -> List[dict]:
    records = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if split is None or record.get("split") == split:
                records.append(record)
    return records


def write_trec(path: str, runs: Dict[str, Sequence[tuple]], run_name: str, top_k: int) -> int:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(path, "w", encoding="utf-8") as out:
        for author_id, scored in runs.items():
            seen = set()
            rank = 1
            for docid, score in sorted(scored, key=lambda x: x[1], reverse=True)[:top_k]:
                if docid in seen:
                    continue
                seen.add(docid)
                out.write(f"{author_id} Q0 {docid} {rank} {score:.4f} {run_name}\n")
                rank += 1
                written += 1
    return written


# --------------------------------------------------------------------------- #
# Sparse arm
# --------------------------------------------------------------------------- #
def run_bm25(records: Sequence[dict], top_k: int = 1000) -> Dict[str, List[tuple]]:
    from rank_bm25 import BM25Okapi

    runs: Dict[str, List[tuple]] = {}
    for i, record in enumerate(records):
        ids = [c["article_id"] for c in record["candidate_items"]]
        corpus = [doc_text(c).lower().split() for c in record["candidate_items"]]
        bm25 = BM25Okapi(corpus)
        scores = bm25.get_scores(record["nl_profile"].lower().split())
        runs[record["author_id"]] = list(zip(ids, (float(s) for s in scores)))
        if (i + 1) % 100 == 0:
            print(f"  bm25 {i + 1}/{len(records)}", flush=True)
    return runs


# --------------------------------------------------------------------------- #
# Dense arm
# --------------------------------------------------------------------------- #
def run_dense(
    records: Sequence[dict],
    model_path: str,
    top_k: int = 1000,
    batch_size: int = 16,
    max_seq_length: int = 128,
    pooling: str = "mean",
) -> Dict[str, List[tuple]]:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_path)
    model.max_seq_length = max_seq_length

    runs: Dict[str, List[tuple]] = {}
    for i, record in enumerate(records):
        candidates = record["candidate_items"]
        ids = [c["article_id"] for c in candidates]
        vectors = model.encode(
            [doc_text(c) for c in candidates],
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        query = model.encode(
            [record["nl_profile"]],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )[0]
        scores = vectors @ query
        runs[record["author_id"]] = list(zip(ids, (float(s) for s in scores)))
        if (i + 1) % 50 == 0:
            print(f"  dense {i + 1}/{len(records)}", flush=True)
    return runs


# --------------------------------------------------------------------------- #
# Fusion
# --------------------------------------------------------------------------- #
def rrf(runs_list: Sequence[Dict[str, List[tuple]] | Dict[str, Dict[str, float]]], k: int = 60, top_k: int = 1000) -> Dict[str, List[tuple]]:
    """Reciprocal rank fusion, the recipe behind the published ensemble row."""
    fused: Dict[str, List[tuple]] = {}
    authors = set(runs_list[0])
    for author_id in authors:
        scores: Dict[str, float] = defaultdict(float)
        for runs in runs_list:
            items = runs.get(author_id, [])
            if isinstance(items, dict):
                items = sorted(items.items(), key=lambda x: x[1], reverse=True)
            for rank, (docid, _) in enumerate(
                sorted(items, key=lambda x: x[1], reverse=True), start=1
            ):
                scores[docid] += 1.0 / (k + rank)
        fused[author_id] = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return fused


def main() -> None:
    parser = argparse.ArgumentParser(description="SciNUP retrieval arms")
    parser.add_argument("--dataset", default="data/SciNUP/dataset.jsonl")
    parser.add_argument("--out-dir", default="results/scinup")
    parser.add_argument("--arm", choices=["bm25", "dense", "fuse"], required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--fuse-with", nargs="*", default=[], help="TREC runfiles to fuse")
    parser.add_argument("--top-k", type=int, default=1000)
    parser.add_argument("--max-seq-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--split", type=int, default=None)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    records = load_dataset(args.dataset, split=args.split)
    if args.limit:
        records = records[: args.limit]
    print(f"records={len(records)} arm={args.arm}")

    if args.arm == "bm25":
        runs = run_bm25(records)
    elif args.arm == "dense":
        if not args.model:
            raise SystemExit("--model required for dense arm")
        runs = run_dense(records, args.model, max_seq_length=args.max_seq_length, batch_size=args.batch_size)
    else:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from eval_scinup import read_trec
        runs_list = [read_trec(path, qrel=False) for path in args.fuse_with]
        runs = rrf(runs_list)

    out = Path(args.out_dir) / f"{args.run_name}.trec"
    count = write_trec(str(out), runs, args.run_name, args.top_k)
    print(f"WROTE {count} lines -> {out}")


if __name__ == "__main__":
    main()
