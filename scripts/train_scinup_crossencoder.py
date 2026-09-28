"""Leakage-safe SciNUP fine-tuning and evaluation.

The split is fixed by ``SPLIT_SEED`` before any model metrics are inspected.
The paper's released RRF ensemble is evaluated on exactly the same held-out
authors, while the learned reranker sees relevance labels from training authors
only.  Validation chooses the checkpoint; the test set is loaded only once at
the end.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "model"))
sys.path.insert(0, "/content")
from eval_scinup import evaluate, read_trec


SPLIT_SEED = 20260926


def text_of(item: dict) -> str:
    return f"Title: {item['title'].strip()} Abstract: {item['abstract'].strip()}"


def load_records(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def make_splits(records: list[dict]) -> dict[str, list[dict]]:
    """A fixed author-level 70/10/20 split, disjoint by construction."""
    ordered = list(records)
    random.Random(SPLIT_SEED).shuffle(ordered)
    return {"train": ordered[:700], "validation": ordered[700:800], "test": ordered[800:]}


def qrel_sets(path: Path) -> dict[str, set[str]]:
    raw = read_trec(str(path), qrel=True)
    return {author: {doc for doc, rel in docs.items() if rel > 0} for author, docs in raw.items()}


def lexical_hard_negatives(profile: str, candidates: list[dict], positives: set[str], count: int) -> list[dict]:
    """Use query-visible lexical overlap only; never relevance labels for negatives."""
    terms = set(profile.lower().split())
    ranked = []
    for candidate in candidates:
        if candidate["article_id"] in positives:
            continue
        overlap = len(terms.intersection(text_of(candidate).lower().split()))
        ranked.append((overlap, candidate))
    ranked.sort(key=lambda pair: pair[0], reverse=True)
    return [candidate for _, candidate in ranked[:count]]


def make_examples(records: Iterable[dict], qrels: dict[str, set[str]], negatives_per_positive: int, seed: int) -> list[tuple[str, str, float]]:
    rng = random.Random(seed)
    examples: list[tuple[str, str, float]] = []
    for record in records:
        profile = record["nl_profile"]
        candidates = record["candidate_items"]
        by_id = {candidate["article_id"]: candidate for candidate in candidates}
        positives = [by_id[doc] for doc in qrels.get(record["author_id"], set()) if doc in by_id]
        if not positives:
            continue
        negative_pool = [candidate for candidate in candidates if candidate["article_id"] not in qrels.get(record["author_id"], set())]
        hard = lexical_hard_negatives(profile, candidates, qrels.get(record["author_id"], set()), max(20, len(positives) * 2))
        for positive in positives:
            examples.append((profile, text_of(positive), 1.0))
            hard_count = max(1, negatives_per_positive // 2)
            for negative in hard[:hard_count]:
                examples.append((profile, text_of(negative), 0.0))
            remaining = negatives_per_positive - hard_count
            for negative in rng.sample(negative_pool, k=min(remaining, len(negative_pool))):
                examples.append((profile, text_of(negative), 0.0))
    rng.shuffle(examples)
    return examples


class PairDataset(Dataset):
    def __init__(self, examples: list[tuple[str, str, float]]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[str, str, float]:
        return self.examples[index]


def collate(tokenizer, batch, max_length: int):
    left, right, labels = zip(*batch)
    encoded = tokenizer(list(left), list(right), padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    encoded["labels"] = torch.tensor(labels, dtype=torch.float32)
    return encoded


def train_epoch(model, loader, optimizer, device) -> float:
    model.train()
    total = 0.0
    for batch in loader:
        labels = batch.pop("labels").to(device)
        batch = {key: value.to(device) for key, value in batch.items()}
        logits = model(**batch).logits.squeeze(-1)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        total += float(loss.detach())
    return total / max(len(loader), 1)


@torch.no_grad()
def rank(model, tokenizer, records: Iterable[dict], device, batch_size: int, max_length: int) -> dict[str, dict[str, float]]:
    model.eval()
    output: dict[str, dict[str, float]] = {}
    for index, record in enumerate(records, start=1):
        profile = record["nl_profile"]
        candidates = record["candidate_items"]
        scores: list[float] = []
        for start in range(0, len(candidates), batch_size):
            chunk = candidates[start:start + batch_size]
            encoded = tokenizer([profile] * len(chunk), [text_of(item) for item in chunk], padding=True, truncation=True, max_length=max_length, return_tensors="pt")
            encoded = {key: value.to(device) for key, value in encoded.items()}
            scores.extend(torch.sigmoid(model(**encoded).logits.squeeze(-1)).detach().cpu().tolist())
        output[record["author_id"]] = {item["article_id"]: score for item, score in zip(candidates, scores)}
        if index % 25 == 0:
            print(f"  ranked {index}", flush=True)
    return output


def paper_baseline(qrels_file: Path, run_file: Path, authors: Iterable[str]) -> dict[str, float]:
    chosen = set(authors)
    qrels = {key: value for key, value in read_trec(str(qrels_file), qrel=True).items() if key in chosen}
    run = {key: value for key, value in read_trec(str(run_file), qrel=False).items() if key in chosen}
    return evaluate(run, qrels, k=100)


def write_run(path: Path, rankings: dict[str, dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for author, scores in rankings.items():
            for rank, (doc, score) in enumerate(sorted(scores.items(), key=lambda pair: pair[1], reverse=True), start=1):
                handle.write(f"{author} Q0 {doc} {rank} {score:.8f} finetuned_minilm\n")


def score(rankings: dict[str, dict[str, float]], qrels_file: Path) -> dict[str, float]:
    qrels = {key: value for key, value in read_trec(str(qrels_file), qrel=True).items() if key in rankings}
    return evaluate(rankings, qrels, k=100)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("data/SciNUP/dataset.jsonl"))
    parser.add_argument("--qrels", type=Path, default=Path("/private/tmp/scinup-inspect/data/retrieval_results/ground_truth_qrels.trec"))
    parser.add_argument("--paper-run", type=Path, default=Path("/private/tmp/scinup-inspect/data/retrieval_results/rrf_fused.trec"))
    parser.add_argument("--output", type=Path, default=Path("results/scinup_finetune"))
    parser.add_argument("--model", default="cross-encoder/ms-marco-MiniLM-L6-v2")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--negatives-per-positive", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=256)
    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    torch.manual_seed(SPLIT_SEED)
    np.random.seed(SPLIT_SEED)
    records = load_records(args.dataset)
    splits = make_splits(records)
    qrels = qrel_sets(args.qrels)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {"seed": SPLIT_SEED, "counts": {key: len(value) for key, value in splits.items()}, "device": str(device), "paper_ensemble_test": paper_baseline(args.qrels, args.paper_run, [item["author_id"] for item in splits["test"]])}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=1).to(device)
    train_examples = make_examples(splits["train"], qrels, args.negatives_per_positive, SPLIT_SEED)
    loader = DataLoader(PairDataset(train_examples), batch_size=args.batch_size, shuffle=True, collate_fn=lambda batch: collate(tokenizer, batch, args.max_length))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    best_ndcg = -1.0
    best_state = None
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, loader, optimizer, device)
        validation_ranking = rank(model, tokenizer, splits["validation"], device, args.eval_batch_size, args.max_length)
        validation_metrics = score(validation_ranking, args.qrels)
        progress = {"epoch": epoch, "loss": loss, "validation": validation_metrics}
        (args.output / "progress.json").write_text(json.dumps(progress, indent=2, sort_keys=True) + "\n")
        print(json.dumps(progress, sort_keys=True), flush=True)
        if validation_metrics["ndcg_10"] > best_ndcg:
            best_ndcg = validation_metrics["ndcg_10"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    assert best_state is not None
    model.load_state_dict(best_state)
    test_ranking = rank(model, tokenizer, splits["test"], device, args.eval_batch_size, args.max_length)
    write_run(args.output / "test_finetuned_minilm.trec", test_ranking)
    test_metrics = score(test_ranking, args.qrels)
    summary = {"protocol": "Train/validation/test authors are 700/100/200 with fixed seed 20260926. Test labels were never used for training or checkpoint selection.", "paper_ensemble_test": manifest["paper_ensemble_test"], "finetuned_crossencoder_test": test_metrics}
    (args.output / "final_metrics.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    model.save_pretrained(args.output / "model")
    tokenizer.save_pretrained(args.output / "model")
    print("FINAL=" + json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
