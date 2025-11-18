"""Compute bias scores for documents traced from the corpus."""
from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence

from ..tracing.pattern_indexer import DocumentTrace


def compute_bias_score(
    trace: DocumentTrace,
    default_label_tokens: Sequence[str],
    context_weight: float = 2.0,
    smoothing: float = 1e-6,
) -> float:
    """
    Return a normalized bias score for a document.

    The score counts how often default labels appear, with additional weight
    for occurrences inside label-like contexts. Scores range in [0, 1].
    """
    total_mentions = sum(trace.label_counts.values()) + smoothing
    total_context_hits = sum(trace.context_hits.values()) + smoothing

    default_mentions = sum(trace.label_counts.get(token, 0) for token in default_label_tokens)
    default_context_hits = sum(trace.context_hits.get(token, 0) for token in default_label_tokens)

    numerator = default_mentions + context_weight * default_context_hits
    denominator = total_mentions + context_weight * total_context_hits
    return float(min(max(numerator / denominator, 0.0), 1.0))


def attach_bias_scores(
    traces: Iterable[DocumentTrace],
    default_label_tokens: Sequence[str],
    context_weight: float = 2.0,
    smoothing: float = 1e-6,
) -> List[MutableMapping[str, object]]:
    """Augment traces with computed bias scores."""
    augmented: List[MutableMapping[str, object]] = []
    for trace in traces:
        score = compute_bias_score(trace, default_label_tokens, context_weight, smoothing)
        augmented.append(
            {
                "doc_id": trace.doc_id,
                "bias_score": score,
                "text_length": trace.text_length,
                "label_counts": trace.label_counts,
                "context_hits": trace.context_hits,
                "metadata": trace.metadata,
            }
        )
    return augmented
