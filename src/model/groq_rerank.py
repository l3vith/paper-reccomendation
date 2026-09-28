"""Groq LLM pointwise rerank over an existing TREC run (SciNUP cascade).

Takes the top-K docs per author from a base runfile, scores each 0-10
against the NL profile with an instruction-tuned open model, and writes a
new TREC run. Unscored tail keeps base-run order below rescored head.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_scinup import read_trec  # noqa: E402

API_URL = "https://api.groq.com/openai/v1/chat/completions"
SCORE_RE = re.compile(r"(\d{1,2})\s*$")
_LAST_CALL = [0.0]


def _pace(min_interval: float) -> None:
    now = time.monotonic()
    wait = min_interval - (now - _LAST_CALL[0])
    if wait > 0:
        time.sleep(wait)
    _LAST_CALL[0] = time.monotonic()


def score_one(api_key: str, model: str, profile: str, title: str, abstract: str, retries: int = 6, min_interval: float = 8.0) -> float:
    prompt = (
        "Rate how relevant this scientific paper is to the researcher profile. "
        "Reply with ONLY an integer 0-10 on the last line.\n\n"
        f"Profile: {profile[:600]}\n\n"
        f"Paper title: {title}\nAbstract: {abstract[:800]}\n\nScore:"
    )
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 512,
        "temperature": 0,
    }).encode()
    for attempt in range(retries):
        _pace(min_interval)
        try:
            req = urllib.request.Request(
                API_URL, data=body,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "User-Agent": "scinup-rerank/1.0",
                },
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                payload = json.loads(resp.read())
            text = payload["choices"][0]["message"]["content"].strip()
            match = SCORE_RE.search(text)
            if match:
                return float(max(0, min(10, int(match.group(1)))))
            return 0.0
        except Exception as exc:
            wait = min(60.0, 5.0 * (attempt + 1))
            print(f"  retry in {wait}s ({exc})", flush=True)
            time.sleep(wait)
    return 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="base TREC runfile")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--run-name", default="groq_rerank")
    parser.add_argument("--model", default="openai/gpt-oss-20b")
    parser.add_argument("--depth", type=int, default=30)
    parser.add_argument("--authors", type=int, default=0, help="0 = all")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=100)
    args = parser.parse_args()

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise SystemExit("GROQ_API_KEY not set")

    base = read_trec(args.run, qrel=False)
    records: dict = {}
    with open(args.dataset, encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            records[record["author_id"]] = record

    author_ids = sorted(base)[args.offset : (args.offset + args.authors) if args.authors else None]
    print(f"authors={len(author_ids)} depth={args.depth} model={args.model}", flush=True)

    out_lines: list[str] = []
    done = 0
    for author_id in author_ids:
        ranked = sorted(base[author_id].items(), key=lambda x: x[1], reverse=True)
        head = [doc for doc, _ in ranked[: args.depth]]
        tail = [doc for doc, _ in ranked[args.depth :]]
        record = records[author_id]
        texts = {c["article_id"]: c for c in record["candidate_items"]}

        jobs = [
            (doc, texts[doc]["title"], texts[doc]["abstract"])
            for doc in head if doc in texts
        ]
        base_scores = dict(ranked)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            scores = list(pool.map(
                lambda job: score_one(api_key, args.model, record["nl_profile"], job[1], job[2]),
                jobs,
            ))
        rescored = sorted(
            zip([j[0] for j in jobs], scores),
            key=lambda x: (-x[1], -base_scores.get(x[0], 0.0)),
        )
        final = [doc for doc, _ in rescored] + [d for d in tail if d not in dict(rescored)]
        rank = 1
        score_map = dict(rescored)
        for doc in final[: args.top_k]:
            out_lines.append(
                f"{author_id} Q0 {doc} {rank} {score_map.get(doc, -1.0):.4f} {args.run_name}\n"
            )
            rank += 1
        done += 1
        if done % 10 == 0:
            print(f"  {done}/{len(author_ids)}", flush=True)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text("".join(out_lines), encoding="utf-8")
    print(f"WROTE {len(out_lines)} -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
