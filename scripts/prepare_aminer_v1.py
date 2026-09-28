"""Fetch and inspect the AMiner Citation-network V1 source used by Xu et al.

Run this on a temporary GPU/CPU machine before training.  It persists only
the parsed records and a manifest; it does not use validation/test labels.
"""

from __future__ import annotations

import json
import pickle
import urllib.request
import zipfile
import argparse
from pathlib import Path


URL = "https://lfs.aminer.cn/lab-datasets/citation/citation-network1.zip"


def parse(path: Path) -> dict[str, dict]:
    papers: dict[str, dict] = {}
    current: dict[str, object] = {}

    def save() -> None:
        paper_id = str(current.get("id") or "")
        if paper_id:
            papers[paper_id] = {
                "title": str(current.get("title") or "").strip(),
                "abstract": str(current.get("abstract") or "").strip(),
                "references": list(current.get("references") or []),
            }

    with path.open(encoding="utf-8", errors="replace") as source:
        for raw in source:
            line = raw.rstrip("\n")
            if line.startswith("#*"):
                save()
                current = {"title": line[2:], "references": []}
            elif line.startswith("#index"):
                current["id"] = line[6:].strip()
            elif line.startswith("#%"):
                current.setdefault("references", []).append(line[2:].strip())
            elif line.startswith("#!"):
                current["abstract"] = line[2:]
        save()
    return papers


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/aminer_v1")
    args = parser.parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    archive = root / "citation-network1.zip"
    if not archive.exists():
        print(f"DOWNLOADING={URL}", flush=True)
        urllib.request.urlretrieve(URL, archive)
    with zipfile.ZipFile(archive) as zipped:
        text_members = [member for member in zipped.namelist() if member.endswith(".txt")]
        if not text_members:
            raise RuntimeError(f"No .txt record file in {zipped.namelist()[:10]}")
        source = root / Path(text_members[0]).name
        if not source.exists():
            zipped.extract(text_members[0], root)
            extracted = root / text_members[0]
            if extracted != source:
                extracted.rename(source)
    parsed = root / "papers.pkl"
    if parsed.exists():
        with parsed.open("rb") as handle:
            papers = pickle.load(handle)
    else:
        papers = parse(source)
        with parsed.open("wb") as handle:
            pickle.dump(papers, handle, protocol=pickle.HIGHEST_PROTOCOL)

    text_ids = {pid for pid, paper in papers.items() if paper["title"] and paper["abstract"]}
    eligible = [
        pid
        for pid, paper in papers.items()
        if pid in text_ids and paper["references"] and all(ref in text_ids for ref in paper["references"])
    ]
    manifest = {
        "source_url": URL,
        "records": len(papers),
        "records_with_title_and_abstract": len(text_ids),
        "eligible_query_papers_all_references_have_text": len(eligible),
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("AMINER_V1_MANIFEST=" + json.dumps(manifest, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
