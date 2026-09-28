"""ACL-200 prefetch eval: local citation context -> rank all papers.

Query follows HAtten/DualEnh: local context + citing title/abstract.
Candidates: all papers (title + abstract). Metrics: MRR, R@10.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load(path: str):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run(model_path: str, acl_dir: str = "/tmp/acl200", limit: int = 0,
        batch_size: int = 64, max_seq_length: int = 192) -> dict:
    from sentence_transformers import SentenceTransformer

    papers = load(f"{acl_dir}/papers.json")
    contexts = load(f"{acl_dir}/contexts.json")
    test = load(f"{acl_dir}/test.json")
    if limit:
        test = test[:limit]

    model = SentenceTransformer(model_path)
    model.max_seq_length = max_seq_length

    pids = list(papers.keys())
    ptexts = [
        f"{(papers[p].get('title') or '').strip()} [SEP] {(papers[p].get('abstract') or '').strip()}"
        for p in pids
    ]
    print(f"papers={len(pids)} queries={len(test)}", flush=True)
    P = model.encode(ptexts, batch_size=batch_size, normalize_embeddings=True,
                     convert_to_numpy=True, show_progress_bar=True).astype(np.float32)

    queries, gold = [], []
    for item in test:
        ctx = contexts[item["context_id"]]
        citing = papers.get(ctx["citing_id"], {})
        q = (f"{ctx['masked_text'].replace('TARGETCIT', 'CIT')} [SEP] "
             f"{(citing.get('title') or '').strip()} [SEP] {(citing.get('abstract') or '').strip()}")
        queries.append(q)
        gold.append(set(item["positive_ids"]))
    Q = model.encode(queries, batch_size=batch_size, normalize_embeddings=True,
                     convert_to_numpy=True, show_progress_bar=True).astype(np.float32)

    rr, r10, n = 0.0, 0.0, 0
    pos_index = {p: j for j, p in enumerate(pids)}
    for i in range(len(queries)):
        scores = Q[i] @ P.T
        ranked_pos = np.argsort(-scores, kind="stable")
        rank_of = np.empty(len(pids), dtype=np.int64)
        rank_of[ranked_pos] = np.arange(1, len(pids) + 1)
        rel = [g for g in gold[i] if g in pos_index]
        if not rel:
            continue
        first = min(int(rank_of[pos_index[g]]) for g in rel)
        rr += 1.0 / first
        r10 += 1.0 if first <= 10 else 0.0
        n += 1
    return {"n": n, "mrr": rr / n, "r@10": r10 / n}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--acl-dir", default="/tmp/acl200")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    m = run(args.model, args.acl_dir, args.limit)
    print(f"MRR={m['mrr']:.4f} R@10={m['r@10']:.4f} (n={m['n']})")


if __name__ == "__main__":
    main()
