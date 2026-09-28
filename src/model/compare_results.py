"""Assemble the assignment comparison table from retrieval result files.

Every model is scored by the same harness on the same queries, so the
comparison is internally valid. The published DBLP row is carried along for
context only and is explicitly marked as a different corpus.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

# Published figures from arXiv:2504.02377 (Table II / Fig. 4), DBLP corpus.
# Different dataset and different candidate construction, so these are NOT
# directly comparable to our SciDocs citation-protocol rows.
PUBLISHED = {
    "specter_published": {
        "label": "SPECTER (published, DBLP)",
        "precision@5": 0.446,
        "recall@5": 0.788,
        "map": 0.7829,
        "mrr": 0.8693,
    },
    "sections_published": {
        "label": "Section-weighted SPECTER (published, DBLP)",
        "precision@5": 0.4591,
        "recall@5": 0.8125,
        "map": 0.8081,
        "mrr": 0.8860,
    },
}

ORDER = ["precision@5", "recall@5", "f1@5", "mrr", "map"]


def load(path: str) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def best_arm(metrics_by_arm: Dict[str, Dict[str, float]], key: str = "map") -> tuple[str, Dict[str, float]]:
    arm = max(metrics_by_arm, key=lambda name: metrics_by_arm[name].get(key, 0.0))
    return arm, metrics_by_arm[arm]


def collect(reports: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for report in reports:
        for model_label, payload in report.get("models", {}).items():
            if model_label in seen:
                continue
            seen.add(model_label)
            arm, metrics = best_arm(payload["arms"])
            rows.append(
                {
                    "model": model_label,
                    "arm": arm,
                    "source": payload.get("source"),
                    "embedding_dim": payload.get("embedding_dim"),
                    **{metric: metrics.get(metric) for metric in ORDER},
                    "n_queries": metrics.get("n_queries"),
                }
            )
    return rows


def render(rows: List[Dict[str, Any]]) -> str:
    header = f"{'model':<34}{'arm':<8}" + "".join(f"{m.replace('precision','P').replace('recall','R'):>10}" for m in ORDER)
    lines = [header, "-" * len(header)]
    for row in rows:
        cells = "".join(
            f"{row[m]:>10.4f}" if isinstance(row.get(m), (int, float)) else f"{'-':>10}" for m in ORDER
        )
        lines.append(f"{row['model']:<34}{row['arm']:<8}{cells}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the comparison table")
    parser.add_argument("reports", nargs="+", help="retrieval_metrics json files")
    parser.add_argument("--output", default="results/comparison_table.json")
    args = parser.parse_args()

    reports = [load(path) for path in args.reports]
    rows = collect(reports)
    ours = sorted(rows, key=lambda r: r["map"] or 0.0, reverse=True)
    published = [PUBLISHED["specter_published"], PUBLISHED["sections_published"]]

    print("\nOURS - SciDocs citation protocol, full corpus (36,261 papers)")
    print(render(ours))
    print("\nPUBLISHED - DBLP, different corpus (context only, not directly comparable)")
    print(render([{**p, "arm": "-", "embedding_dim": None} for p in published]))

    specter = next((r for r in ours if r["model"] == "specter"), None)
    best = ours[0] if ours else None
    verdict: Dict[str, Any] = {"best_model": best["model"] if best else None}
    if specter and best:
        verdict["delta_vs_specter"] = {
            metric: round((best[metric] or 0.0) - (specter[metric] or 0.0), 4) for metric in ORDER
        }
        verdict["beats_specter_on"] = [m for m in ORDER if (best[m] or 0) > (specter[m] or 0)]
        print("\nVERDICT")
        print(f"  best = {best['model']} (arm {best['arm']})")
        print(f"  beats our SPECTER anchor on: {verdict['beats_specter_on'] or 'nothing'}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(
        json.dumps({"ours": ours, "published_dblp": published, "verdict": verdict}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
