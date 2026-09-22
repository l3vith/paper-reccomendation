"""Build citation/click-grounded training triplets from the SciDocs release."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd


def paper_text(paper: dict) -> str:
    title = str(paper.get("title") or "").strip()
    abstract = str(paper.get("abstract") or "").strip()
    return " [SEP] ".join(part for part in (title, abstract) if part)


def build_scidocs_triplets(
    scidocs_dir: str = "data/scidocs",
    output_path: str = "data/scidocs_triplets.csv",
    negatives_per_click: int = 2,
    seed: int = 42,
) -> pd.DataFrame:
    """Convert real recommendation clicks and displayed alternatives to triplets."""
    root = Path(scidocs_dir)
    with (root / "recomm" / "paper_metadata.json").open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    clicks = pd.read_csv(root / "recomm" / "train.csv")
    rng = random.Random(seed)
    rows: list[tuple[str, str, str]] = []

    for event in clicks.itertuples(index=False):
        anchor = paper_text(metadata.get(str(event.pid), {}))
        positive = paper_text(metadata.get(str(event.clicked_pid), {}))
        candidate_ids = [item for item in str(event.similar_papers_avail_at_time).split(",") if item]
        negative_ids = [item for item in candidate_ids if item != str(event.clicked_pid) and paper_text(metadata.get(item, {}))]
        if not anchor or not positive or not negative_ids:
            continue
        rng.shuffle(negative_ids)
        for negative_id in negative_ids[: max(1, negatives_per_click)]:
            rows.append((anchor, positive, paper_text(metadata[negative_id])))

    frame = pd.DataFrame(rows, columns=["anchor_text", "positive_text", "negative_text"]).drop_duplicates()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return frame


def build_enhanced_triplets(
    local_path: str = "data/triplets.csv",
    output_path: str = "data/triplets_enhanced.csv",
    **kwargs,
) -> pd.DataFrame:
    """Combine project triplets with SciDocs click-grounded triplets."""
    scidocs = build_scidocs_triplets(**kwargs)
    local = pd.read_csv(local_path) if Path(local_path).exists() else pd.DataFrame(columns=scidocs.columns)
    combined = pd.concat([scidocs, local], ignore_index=True).drop_duplicates()
    combined.to_csv(output_path, index=False)
    return combined


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build SciDocs recommendation triplets")
    parser.add_argument("--scidocs-dir", default="data/scidocs")
    parser.add_argument("--local", default="data/triplets.csv")
    parser.add_argument("--output", default="data/triplets_enhanced.csv")
    parser.add_argument("--negatives-per-click", type=int, default=2)
    args = parser.parse_args()
    result = build_enhanced_triplets(
        local_path=args.local,
        output_path=args.output,
        scidocs_dir=args.scidocs_dir,
        negatives_per_click=args.negatives_per_click,
    )
    print(f"WROTE_TRIPLETS={len(result)} path={args.output}")
