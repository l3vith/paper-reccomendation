"""Full-corpus scientific paper retrieval evaluation harness.

Protocol notes
--------------
The assignment target (arXiv:2504.02377) reports Precision@5, Recall@5, MAP and
MRR on DBLP, where the relevant set for a query paper is its reference list.

This harness reproduces that *shape* of evaluation on the SciDocs ``recomm``
corpus already vendored under ``data/scidocs`` (36,261 papers, 310k reference
edges) because the DBLP snapshot is not available offline:

* ``citation`` protocol - query paper -> the references that exist inside the
  corpus, with the entire corpus as the candidate pool. This is the closest
  analogue to the DBLP protocol and is the one to quote.
* ``click`` protocol - the 1,000 held-out ``recomm/test.csv`` clickthrough
  events, scored only over the candidates that were actually displayed.

Absolute numbers are therefore not directly comparable to the published DBLP
table. The harness is self-consistent though: every model is scored with
identical queries, candidates and code path, so the relative gap between rows
is trustworthy. Report it that way.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from .scidocs_data import paper_text

CACHE_DIR = Path("data/emb_cache")
DEFAULT_TOP_KS = (5, 10, 20)


# --------------------------------------------------------------------------- #
# Corpus
# --------------------------------------------------------------------------- #
def load_metadata(scidocs_dir: str) -> Dict[str, dict]:
    path = Path(scidocs_dir) / "recomm" / "paper_metadata.json"
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def build_corpus(metadata: Dict[str, dict]) -> Tuple[List[str], List[str]]:
    """Return (paper_ids, texts) for every corpus paper that has usable text."""
    ids: List[str] = []
    texts: List[str] = []
    for pid, record in metadata.items():
        text = paper_text(record)
        if text:
            ids.append(pid)
            texts.append(text)
    return ids, texts


# --------------------------------------------------------------------------- #
# Queries
# --------------------------------------------------------------------------- #
@dataclass
class Query:
    paper_id: str
    relevant: set[str]
    candidates: List[str] | None = None  # None => full corpus


def build_citation_queries(
    metadata: Dict[str, dict],
    corpus_ids: Sequence[str],
    limit: int = 0,
    seed: int = 42,
) -> List[Query]:
    """Query paper -> its in-corpus references, ranked against the full corpus."""
    in_corpus = set(corpus_ids)
    queries: List[Query] = []
    for pid in corpus_ids:
        relevant = {
            ref
            for ref in (metadata[pid].get("references") or [])
            if ref in in_corpus and ref != pid
        }
        if relevant:
            queries.append(Query(paper_id=pid, relevant=relevant))
    queries.sort(key=lambda q: q.paper_id)
    if limit and len(queries) > limit:
        rng = np.random.default_rng(seed)
        picked = rng.choice(len(queries), size=limit, replace=False)
        queries = [queries[i] for i in sorted(picked)]
    return queries


def build_click_queries(
    scidocs_dir: str,
    metadata: Dict[str, dict],
    corpus_ids: Sequence[str],
    split: str = "test",
) -> List[Query]:
    """Held-out clickthrough events, scored only over the displayed candidates."""
    in_corpus = set(corpus_ids)
    events = pd.read_csv(Path(scidocs_dir) / "recomm" / f"{split}.csv")
    queries: List[Query] = []
    for event in events.itertuples(index=False):
        query_id = str(event.pid)
        clicked = str(event.clicked_pid)
        candidates = [
            item
            for item in str(event.similar_papers_avail_at_time).split(",")
            if item in in_corpus and item != query_id
        ]
        if query_id not in in_corpus or clicked not in candidates:
            continue
        queries.append(Query(paper_id=query_id, relevant={clicked}, candidates=candidates))
    return queries


# --------------------------------------------------------------------------- #
# Scoring backends
# --------------------------------------------------------------------------- #
def _cache_path(tag: str) -> Path:
    safe = "".join(ch if ch.isalnum() or ch in "-._" else "_" for ch in tag)
    return CACHE_DIR / f"{safe}.npy"


def encode_corpus(
    model_path: str,
    texts: Sequence[str],
    tag: str,
    batch_size: int = 32,
    max_seq_length: int = 256,
) -> np.ndarray:
    """Encode the corpus with a SentenceTransformer, caching to disk."""
    cache = _cache_path(tag)
    if cache.exists():
        return np.load(cache)

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_path)
    model.max_seq_length = max_seq_length
    embeddings = model.encode(
        list(texts),
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(cache, embeddings)
    return embeddings


def load_precomputed_embeddings(
    jsonl_path: str,
    corpus_ids: Sequence[str],
    tag: str,
) -> np.ndarray:
    """Load an existing embedding dump (e.g. the SPECTER release) for our corpus."""
    cache = _cache_path(tag)
    if cache.exists():
        return np.load(cache)

    wanted = set(corpus_ids)
    vectors: Dict[str, np.ndarray] = {}
    with open(jsonl_path, encoding="utf-8") as handle:
        for line in handle:
            if '"embedding"' not in line:
                continue
            record = json.loads(line)
            pid = record.get("paper_id")
            if pid in wanted:
                vectors[pid] = np.asarray(record["embedding"], dtype=np.float32)

    missing = wanted - set(vectors)
    if missing:
        raise ValueError(
            f"{len(missing)} of {len(wanted)} corpus papers absent from {jsonl_path}"
        )

    matrix = np.stack([vectors[pid] for pid in corpus_ids])
    matrix /= np.clip(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12, None)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(cache, matrix.astype(np.float32))
    return matrix.astype(np.float32)


class Bm25Backend:
    """Lexical scorer used for the hybrid fusion arm."""

    def __init__(self, texts: Sequence[str]) -> None:
        from rank_bm25 import BM25Okapi

        self._tokenize = lambda text: text.lower().split()
        self._bm25 = BM25Okapi([self._tokenize(text) for text in texts])

    def scores(self, query_text: str, candidates: Sequence[str]) -> np.ndarray:
        raw = np.asarray(self._bm25.get_scores(self._tokenize(query_text)), dtype=np.float32)
        return raw[list(candidates)]


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def rank_metrics(
    ranked_ids: Sequence[Sequence[str]],
    relevant: Sequence[set[str]],
    top_ks: Sequence[int] = DEFAULT_TOP_KS,
) -> Dict[str, float]:
    """MAP / MRR / P@k / R@k / F1@k / Hit@k from already-sorted candidate ids."""
    if not ranked_ids:
        return {"n_queries": 0}

    max_k = max(top_ks)
    reciprocal: List[float] = []
    average_precision: List[float] = []
    precision = {k: 0.0 for k in top_ks}
    recall = {k: 0.0 for k in top_ks}
    f1 = {k: 0.0 for k in top_ks}
    hit = {k: 0.0 for k in top_ks}

    for order, rel in zip(ranked_ids, relevant):
        seen = 0
        first_rank = 0
        ap_sum = 0.0
        for position, pid in enumerate(order[:max_k], start=1):
            if pid in rel:
                seen += 1
                if first_rank == 0:
                    first_rank = position
                ap_sum += seen / position
        average_precision.append(ap_sum / len(rel) if rel else 0.0)
        reciprocal.append(1.0 / first_rank if first_rank else 0.0)

        for k in top_ks:
            found = sum(1 for pid in order[:k] if pid in rel)
            p = found / k
            r = found / len(rel) if rel else 0.0
            precision[k] += p
            recall[k] += r
            f1[k] += (2 * p * r / (p + r)) if (p + r) else 0.0
            hit[k] += 1.0 if found else 0.0

    n = len(ranked_ids)
    metrics: Dict[str, float] = {
        "n_queries": n,
        "mrr": float(np.mean(reciprocal)),
        "map": float(np.mean(average_precision)),
    }
    for k in top_ks:
        metrics[f"precision@{k}"] = precision[k] / n
        metrics[f"recall@{k}"] = recall[k] / n
        metrics[f"f1@{k}"] = f1[k] / n
        metrics[f"hit@{k}"] = hit[k] / n
    return metrics


def _minmax(values: np.ndarray) -> np.ndarray:
    span = np.ptp(values, axis=-1, keepdims=True)
    return (values - values.min(axis=-1, keepdims=True)) / (span + 1e-9)


def evaluate(
    corpus_ids: Sequence[str],
    texts: Sequence[str],
    queries: Sequence[Query],
    dense: np.ndarray,
    bm25: Bm25Backend | None = None,
    dense_weights: Sequence[float] = (1.0,),
    batch_size: int = 64,
    top_ks: Sequence[int] = DEFAULT_TOP_KS,
) -> Dict[str, Dict[str, float]]:
    """Score every dense weight arm. dense=1.0 is pure dense, 0.0 is pure BM25."""
    position = {pid: i for i, pid in enumerate(corpus_ids)}
    all_ids = np.asarray(corpus_ids)
    arms: Dict[str, List[List[str]]] = {f"w{weight:.2f}": [] for weight in dense_weights}
    relevant: List[set[str]] = []

    for start in range(0, len(queries), batch_size):
        chunk = queries[start : start + batch_size]
        query_vectors = np.stack([dense[position[q.paper_id]] for q in chunk])
        full_scores = query_vectors @ dense.T  # cosine, both sides normalized

        for row, query in enumerate(chunk):
            if query.candidates is None:
                candidates = all_ids
                scores = full_scores[row]
            else:
                idx = np.asarray([position[c] for c in query.candidates])
                candidates = np.asarray(query.candidates)
                scores = full_scores[row][idx]
                if bm25 is not None:
                    lexical = bm25.scores(texts[position[query.paper_id]], query.candidates)
                else:
                    lexical = None

            self_at = np.flatnonzero(candidates == query.paper_id)
            for weight in dense_weights:
                if weight >= 1.0:
                    fused = scores
                elif weight <= 0.0:
                    if lexical is None:
                        fused = scores
                    else:
                        fused = _minmax(lexical[None, :])[0]
                else:
                    dense_norm = (np.clip(scores, -1.0, 1.0) + 1.0) / 2.0
                    if lexical is None:
                        fused = dense_norm
                    else:
                        fused = weight * dense_norm + (1.0 - weight) * _minmax(lexical[None, :])[0]

                order = np.argsort(-fused, kind="stable")
                if self_at.size:
                    order = order[order != self_at[0]]
                arms[f"w{weight:.2f}"].append(candidates[order].tolist())
            relevant.append(query.relevant)

    return {
        arm: rank_metrics(arms[arm], relevant, top_ks)
        for arm in arms
    }


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def run(
    models: Sequence[Tuple[str, str]],
    protocol: str = "citation",
    scidocs_dir: str = "data/scidocs",
    limit: int = 1000,
    dense_weights: Sequence[float] = (1.0,),
    max_seq_length: int = 256,
    top_ks: Sequence[int] = DEFAULT_TOP_KS,
    precomputed: Tuple[str, str] | None = None,
    output: str = "results/retrieval_metrics.json",
) -> Dict[str, Any]:
    metadata = load_metadata(scidocs_dir)
    corpus_ids, texts = build_corpus(metadata)

    if protocol == "citation":
        queries = build_citation_queries(metadata, corpus_ids, limit=limit)
    elif protocol == "click":
        queries = build_click_queries(scidocs_dir, metadata, corpus_ids, split="test")
    else:
        raise ValueError(f"Unknown protocol: {protocol}")

    needs_bm25 = any(weight < 1.0 for weight in dense_weights)
    bm25 = Bm25Backend(texts) if needs_bm25 else None

    report: Dict[str, Any] = {
        "protocol": protocol,
        "corpus_papers": len(corpus_ids),
        "queries": len(queries),
        "top_ks": list(top_ks),
        "models": {},
    }

    if precomputed is not None:
        label, jsonl = precomputed
        matrix = load_precomputed_embeddings(jsonl, corpus_ids, tag=f"pre_{label}")
        report["models"][label] = {
            "source": jsonl,
            "embedding_dim": int(matrix.shape[1]),
            "arms": evaluate(
                corpus_ids, texts, queries, matrix, bm25, dense_weights, top_ks=top_ks
            ),
        }

    for label, model_path in models:
        if not os.path.exists(model_path) and "/" not in model_path:
            print(f"SKIP {label}: {model_path} not found")
            continue
        matrix = encode_corpus(
            model_path,
            texts,
            tag=f"{label}_{max_seq_length}",
            max_seq_length=max_seq_length,
        )
        report["models"][label] = {
            "source": model_path,
            "embedding_dim": int(matrix.shape[1]),
            "arms": evaluate(
                corpus_ids, texts, queries, matrix, bm25, dense_weights, top_ks=top_ks
            ),
        }

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Full-corpus paper retrieval evaluation")
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="Repeatable. e.g. --model base=allenai/scibert_scivocab_uncased",
    )
    parser.add_argument("--protocol", choices=("citation", "click"), default="citation")
    parser.add_argument("--scidocs-dir", default="data/scidocs")
    parser.add_argument("--limit", type=int, default=1000, help="0 = all queries")
    parser.add_argument(
        "--dense-weight",
        action="append",
        type=float,
        default=[],
        help="Repeatable. 1.0 = pure dense, 0.0 = pure BM25.",
    )
    parser.add_argument("--max-seq-length", type=int, default=256)
    parser.add_argument(
        "--precomputed",
        metavar="LABEL=PATH",
        help="e.g. --precomputed specter=data/scidocs/specter-embeddings/recomm.jsonl",
    )
    parser.add_argument("--output", default="results/retrieval_metrics.json")
    args = parser.parse_args()

    models: List[Tuple[str, str]] = []
    for item in args.model:
        label, _, path = item.partition("=")
        if not path:
            raise SystemExit(f"--model expects LABEL=PATH, got {item!r}")
        models.append((label, path))

    precomputed = None
    if args.precomputed:
        label, _, path = args.precomputed.partition("=")
        if not path:
            raise SystemExit(f"--precomputed expects LABEL=PATH, got {args.precomputed!r}")
        precomputed = (label, path)

    report = run(
        models=models,
        protocol=args.protocol,
        scidocs_dir=args.scidocs_dir,
        limit=args.limit,
        dense_weights=args.dense_weight or (1.0,),
        max_seq_length=args.max_seq_length,
        precomputed=precomputed,
        output=args.output,
    )

    print(f"\nprotocol={report['protocol']} corpus={report['corpus_papers']} "
          f"queries={report['queries']}")
    header = f"{'model':<22}{'arm':<8}{'P@5':>8}{'R@5':>8}{'F1@5':>8}{'MRR':>8}{'MAP':>8}"
    print(header)
    print("-" * len(header))
    for label, payload in report["models"].items():
        for arm, metrics in payload["arms"].items():
            print(
                f"{label:<22}{arm:<8}"
                f"{metrics['precision@5']:>8.4f}{metrics['recall@5']:>8.4f}"
                f"{metrics['f1@5']:>8.4f}{metrics['mrr']:>8.4f}{metrics['map']:>8.4f}"
            )


if __name__ == "__main__":
    main()
