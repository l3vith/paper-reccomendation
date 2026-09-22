"""Fine-tuning and fair held-out evaluation for the SciBERT study variants."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import gc
from typing import Any, Dict

from .variants import MODEL_VARIANTS, get_variant, has_trained_checkpoint

RESULTS_PATH = "results/variant_evaluation_metrics.json"


def fine_tune_variant(variant_key: str, triplets_path: str = "data/triplets_enhanced.csv", db_path: str = "data/papers.db", epochs: int = 2, batch_size: int = 4, use_qlora: bool = True, device: str | None = None, steps_per_epoch: int | None = None) -> str:
    """Fine-tune a selected profile and create its matching FAISS index.

    This function is intended to run in the isolated worker process so the
    trainer can use MPS without sharing memory with Streamlit's recommenders.
    """
    from .train import create_model, load_training_data, train
    from src.recommender.indexer import PaperIndex

    variant = get_variant(variant_key)
    train_examples, _ = load_training_data(triplets_path)
    if not train_examples:
        raise ValueError("No training triplets are available.")
    model = create_model(use_qlora=use_qlora, use_lora=True, pooling=variant.pooling, device=device)
    train(model, train_examples, output_path=variant.model_path, epochs=epochs, batch_size=batch_size, steps_per_epoch=steps_per_epoch)
    # Indexing loads the saved checkpoint. Explicitly release the training
    # model first so the worker never keeps two full SciBERT copies alive.
    del model
    gc.collect()
    PaperIndex(variant.model_path, variant.index_path, variant.mapping_path, pooling=variant.pooling).build_index(db_path)
    return variant.model_path


def fine_tune_variant_subprocess(
    variant_key: str,
    epochs: int = 2,
    batch_size: int = 4,
    use_qlora: bool = True,
) -> str:
    """Train outside Streamlit so a native/OOM failure cannot kill the UI."""
    variant = get_variant(variant_key)
    command = [
        sys.executable,
        "-m",
        "src.model.train_variant_worker",
        "--variant", variant_key,
        "--epochs", str(int(epochs)),
        "--batch-size", str(int(batch_size)),
    ]
    if not use_qlora:
        command.append("--no-qlora")
    env = os.environ.copy()
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    result = subprocess.run(command, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        details = (result.stderr or result.stdout or "worker exited without output").strip()
        raise RuntimeError(f"training worker exited with code {result.returncode}: {details[-2000:]}")
    if not has_trained_checkpoint(variant):
        raise RuntimeError("training worker completed but did not create model weights")
    return variant.model_path


def evaluate_variants(triplets_path: str = "data/triplets.csv", db_path: str = "data/papers.db", results_path: str = RESULTS_PATH) -> Dict[str, Any]:
    """Evaluate every trained checkpoint on one identical held-out split."""
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sentence_transformers import SentenceTransformer
    from .evaluate import compute_hybrid_triplet_retrieval_metrics, compute_triplet_retrieval_metrics, compute_triplet_accuracy

    df = pd.read_csv(triplets_path)
    # Match the training split and keep every row for a query paper in one
    # partition. Row-wise splitting inflates scores when anchors repeat.
    anchors = df["anchor_text"].drop_duplicates().tolist()
    _, test_anchors = train_test_split(anchors, test_size=0.2, random_state=42)
    test_df = df[df["anchor_text"].isin(set(test_anchors))]
    test_triplets = [tuple(row) for row in test_df.to_numpy()]
    report: Dict[str, Any] = {"schema_version": 1, "models": {}}
    for key, variant in MODEL_VARIANTS.items():
        row: Dict[str, Any] = {"label": variant.label, "architecture": variant.description, "pooling": variant.pooling, "model_path": variant.model_path}
        if not has_trained_checkpoint(variant):
            row["status"] = "not trained"
        else:
            model = SentenceTransformer(variant.model_path)
            row.update({
                "status": "evaluated",
                "triplet_accuracy": compute_triplet_accuracy(model, test_triplets),
                **compute_triplet_retrieval_metrics(model, test_triplets, k_values=[5, 10]),
                **compute_hybrid_triplet_retrieval_metrics(model, test_triplets, k_values=[5, 10]),
                "retrieval_basis": "held-out triplets",
                "retrieval_queries": len({triplet[0] for triplet in test_triplets}),
            })
        report["models"][key] = row
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return report
