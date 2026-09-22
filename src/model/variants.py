"""Controlled SciBERT sentence-embedding variants for the model study."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class ModelVariant:
    key: str
    label: str
    description: str
    pooling: str
    model_path: str
    index_path: str
    mapping_path: str


# All profiles use the current SciBERT encoder and identical ranking-loss data.
# Only the pooling head changes, making the comparison controlled and reusable.
MODEL_VARIANTS: Dict[str, ModelVariant] = {
    "mean": ModelVariant("mean", "SciBERT Mean Pooling (current)", "Original 768-d mean-token pooling architecture.", "mean", "models/scibert-finetuned-papers", "data/faiss_index.bin", "data/paper_id_map.json"),
    "cls": ModelVariant("cls", "SciBERT [CLS] Pooling", "Uses the final [CLS] token as a 768-d paper representation.", "cls", "models/variants/scibert-cls-finetuned-papers", "data/variants/scibert-cls.faiss", "data/variants/scibert-cls-paper-id-map.json"),
    "mean_max": ModelVariant("mean_max", "SciBERT Mean + Max Pooling", "Concatenates mean and max pooling into a 1,536-d representation.", "mean_max", "models/variants/scibert-mean-max-finetuned-papers", "data/variants/scibert-mean-max.faiss", "data/variants/scibert-mean-max-paper-id-map.json"),
    "weighted_mean": ModelVariant("weighted_mean", "SciBERT Weighted-Mean Pooling", "Uses position-weighted mean token pooling for a 768-d representation.", "weighted_mean", "models/variants/scibert-weighted-mean-finetuned-papers", "data/variants/scibert-weighted-mean.faiss", "data/variants/scibert-weighted-mean-paper-id-map.json"),
}


def get_variant(key: str) -> ModelVariant:
    try:
        return MODEL_VARIANTS[key]
    except KeyError as exc:
        raise ValueError(f"Unknown model variant: {key!r}") from exc


def has_trained_checkpoint(variant: ModelVariant) -> bool:
    return any(os.path.isfile(os.path.join(variant.model_path, name)) for name in ("model.safetensors", "pytorch_model.bin"))
