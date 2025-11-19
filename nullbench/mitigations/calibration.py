"""Calibration-style mitigations for NullBench decoder inference."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, MutableMapping, Sequence

import numpy as np

PredictFn = Callable[[Sequence[str]], np.ndarray]
GeneratorMap = Mapping[str, Callable[[Sequence[str] | int], Sequence[str]]]


@dataclass
class MitigationConfig:
    method: str
    reference_generator: str = "placeholder"
    sample_size: int | None = 512
    epsilon: float = 1e-9


class MitigationError(RuntimeError):
    pass


def build_mitigated_predict_fn(
    base_predict_fn: PredictFn,
    *,
    config: MitigationConfig,
    generators: GeneratorMap,
    task_texts: Sequence[str],
    formatter: Callable[[str], str] | None = None,
) -> PredictFn:
    """Return a predict_fn with the requested mitigation applied.

    The wrapped function is fully compatible with NullBench and only activates when
    `config.method` is not "none".
    """

    method = (config.method or "none").lower()
    if method in {"", "none"}:
        return base_predict_fn

    if config.reference_generator not in generators:
        raise MitigationError(
            f"Generator '{config.reference_generator}' not available; choose from {list(generators.keys())}."
        )

    formatter_fn: Callable[[str], str] = formatter if formatter is not None else (lambda x: x)
    reference_inputs = _materialize_reference_inputs(
        generators[config.reference_generator],
        task_texts,
        config.sample_size,
    )
    formatted_inputs = [formatter_fn(text) for text in reference_inputs]
    reference_probs = base_predict_fn(formatted_inputs)
    reference_probs = _normalize(reference_probs, config.epsilon)

    if method == "cc":
        bias = _contextual_bias(reference_probs, config.epsilon)
        return _wrap_with_bias(base_predict_fn, bias, mode="logit", epsilon=config.epsilon)
    if method == "dc":
        bias = reference_probs.mean(axis=0)
        return _wrap_with_bias(base_predict_fn, bias, mode="subtract", epsilon=config.epsilon)
    if method == "looc":
        bias = _leave_one_out_bias(reference_probs, config.epsilon)
        return _wrap_with_bias(base_predict_fn, bias, mode="logit", epsilon=config.epsilon)

    raise MitigationError(f"Unsupported mitigation method '{config.method}'.")


def _materialize_reference_inputs(
    generator_fn: Callable[[Sequence[str] | int], Sequence[str]],
    fallback_texts: Sequence[str],
    sample_size: int | None,
) -> Sequence[str]:
    total = sample_size if sample_size and sample_size > 0 else len(fallback_texts)
    if total <= 0:
        total = len(fallback_texts)
    try:
        candidates = generator_fn(total)
    except TypeError:
        candidates = generator_fn(fallback_texts)

    if sample_size is not None and len(candidates) > sample_size:
        return list(candidates)[: sample_size]
    return list(candidates)


def _normalize(probs: np.ndarray, epsilon: float) -> np.ndarray:
    probs = np.asarray(probs, dtype=np.float64)
    row_sums = probs.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0.0] = 1.0
    probs = probs / row_sums
    return np.clip(probs, epsilon, 1.0)


def _contextual_bias(reference_probs: np.ndarray, epsilon: float) -> np.ndarray:
    bias = reference_probs.mean(axis=0)
    return np.clip(bias, epsilon, None)


def _leave_one_out_bias(reference_probs: np.ndarray, epsilon: float) -> np.ndarray:
    count = reference_probs.shape[0]
    if count <= 1:
        return _contextual_bias(reference_probs, epsilon)
    summed = reference_probs.sum(axis=0)
    looc = summed / max(count - 1, 1)
    return np.clip(looc, epsilon, None)


def _wrap_with_bias(
    base_predict_fn: PredictFn,
    bias: np.ndarray,
    *,
    mode: str,
    epsilon: float,
) -> PredictFn:
    bias = np.asarray(bias, dtype=np.float64)

    def mitigated(texts: Sequence[str]) -> np.ndarray:
        base = _normalize(base_predict_fn(texts), epsilon)
        if mode == "logit":
            adjusted = np.exp(np.log(base) - np.log(bias))
        elif mode == "subtract":
            adjusted = np.clip(base - bias, epsilon, None)
        else:
            raise MitigationError(f"Unknown bias mode '{mode}'")
        adjusted = np.clip(adjusted, epsilon, None)
        row_sums = adjusted.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0.0] = 1.0
        return adjusted / row_sums

    return mitigated
