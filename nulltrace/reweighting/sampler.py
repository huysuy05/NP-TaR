"""Sampling utilities for TaRP-style data reweighting."""
from __future__ import annotations

from typing import Iterable, List, Mapping, MutableMapping, Sequence


def compute_sampling_weight(
    bias_score: float,
    bias_threshold: float = 0.5,
    min_weight: float = 0.1,
    max_weight: float = 1.0,
) -> float:
    """Downweight documents whose bias score exceeds the threshold."""
    bias_score = float(max(0.0, min(1.0, bias_score)))
    if bias_score <= bias_threshold:
        return max_weight
    # Linearly decay weight after the threshold
    scale = (1.0 - bias_score) / (1.0 - bias_threshold + 1e-6)
    weight = min_weight + (max_weight - min_weight) * max(scale, 0.0)
    return float(max(min_weight, min(weight, max_weight)))


def upweight_minority_labels(
    base_weight: float,
    doc_label_counts: Mapping[str, int],
    minority_labels: Sequence[str] | None = None,
    bonus: float = 0.25,
) -> float:
    """Increase the sampling weight for documents that mention minority labels."""
    if not minority_labels:
        return base_weight
    if any(doc_label_counts.get(label, 0) > 0 for label in minority_labels):
        return float(base_weight * (1.0 + bonus))
    return base_weight


def build_reweighted_manifest(
    bias_metadata: Iterable[Mapping[str, object]],
    bias_threshold: float = 0.5,
    min_weight: float = 0.1,
    max_weight: float = 1.0,
    minority_labels: Sequence[str] | None = None,
    minority_bonus: float = 0.25,
) -> List[MutableMapping[str, object]]:
    """Return a manifest containing per-document sampling weights."""
    manifest: List[MutableMapping[str, object]] = []
    for entry in bias_metadata:
        doc_id = str(entry["doc_id"])
        bias_score = float(entry["bias_score"])
        label_counts = entry.get("label_counts", {})

        weight = compute_sampling_weight(
            bias_score,
            bias_threshold=bias_threshold,
            min_weight=min_weight,
            max_weight=max_weight,
        )
        weight = upweight_minority_labels(
            base_weight=weight,
            doc_label_counts=label_counts,
            minority_labels=minority_labels,
            bonus=minority_bonus,
        )

        manifest.append(
            {
                "doc_id": doc_id,
                "bias_score": bias_score,
                "sampling_weight": weight,
                "metadata": entry.get("metadata", {}),
            }
        )
    return manifest
