import glob
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np


def _json_default(obj: Any):
    """
    Ensure numpy numbers can be serialized to JSON.
    """
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    raise TypeError


def save_results_json(
    results: Iterable[Dict[str, Any]],
    experiments_dir: str = "experiments",
    filename: str = "nullbench_results.json",
) -> str:
    """
    Persist a list of NullBench result dicts to experiments/ as JSON.

    Args:
        results: Iterable of result dicts produced by NullBench.evaluate with
            additional metadata such as {"model": "...", "task": "..."}.
        experiments_dir: Directory to place the JSON file.
        filename: Name of the JSON file.

    Returns:
        Full path to the written JSON file.
    """
    os.makedirs(experiments_dir, exist_ok=True)
    path = os.path.join(experiments_dir, filename)
    with open(path, "w") as f:
        json.dump(list(results), f, indent=2, default=_json_default)
    return path


def load_results_from_files(patterns: Sequence[str]) -> List[Dict[str, Any]]:
    matched: List[str] = []
    for pattern in patterns:
        matched.extend(glob.glob(pattern))

    unique_paths = sorted({Path(path).as_posix() for path in matched})
    results: List[Dict[str, Any]] = []

    for path in unique_paths:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)

        def _ingest(entry: Dict[str, Any]):
            enriched = dict(entry)
            enriched.setdefault("_source", path)
            results.append(enriched)

        if isinstance(payload, list):
            for item in payload:
                if isinstance(item, dict):
                    _ingest(item)
        elif isinstance(payload, dict):
            _ingest(payload)

    return results


def _prepare_entries(
    results: Iterable[Dict[str, Any]], metric: str
) -> List[Dict[str, Any]]:
    entries = []
    for res in results:
        if metric not in res:
            continue
        entries.append(
            {
                "model": res.get("model", "model"),
                "task": res.get("task", "task"),
                "value": float(res[metric]),
            }
        )
    return sorted(entries, key=lambda x: x["value"], reverse=True)


def plot_overall_leaderboard(
    results: Iterable[Dict[str, Any]],
    metric: str = "nrs_overall",
    output_dir: str = "docs",
    title: Optional[str] = None,
) -> Optional[str]:
    """
    Plot a horizontal bar leaderboard across all (model, task) pairs.

    Args:
        results: Iterable of NullBench result dicts containing the metric.
        metric: Metric key to rank (e.g., "nrs_overall").
        output_dir: Directory where the plot is saved.
        title: Optional title for the chart.

    Returns:
        Path to the saved plot, or None if no entries.
    """
    entries = _prepare_entries(results, metric)
    if not entries:
        return None

    os.makedirs(output_dir, exist_ok=True)
    labels = [f"{e['model']} · {e['task']}" for e in entries]
    values = [e["value"] for e in entries]

    fig, ax = plt.subplots(figsize=(8, 0.4 * len(entries) + 2))
    ax.barh(labels, values, color="#4c72b0")
    ax.invert_yaxis()
    ax.set_xlabel(metric)
    ax.set_title(title or f"NullBench leaderboard ({metric})")
    plt.tight_layout()

    path = os.path.join(output_dir, f"leaderboard_{metric}.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_task_leaderboards(
    results: Iterable[Dict[str, Any]],
    metric: str = "nrs_overall",
    output_dir: str = "docs",
) -> List[str]:
    """
    Plot per-task leaderboards (one plot per task).

    Args:
        results: Iterable of NullBench result dicts containing the metric.
        metric: Metric key to rank (e.g., "nrs_overall").
        output_dir: Directory where the plots are saved.

    Returns:
        List of paths to the saved plots (one per task).
    """
    os.makedirs(output_dir, exist_ok=True)
    plots = []

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for res in results:
        task = res.get("task", "task")
        grouped.setdefault(task, []).append(res)

    for task, task_results in grouped.items():
        entries = _prepare_entries(task_results, metric)
        if not entries:
            continue

        labels = [e["model"] for e in entries]
        values = [e["value"] for e in entries]

        fig, ax = plt.subplots(figsize=(6, 0.4 * len(entries) + 1.5))
        ax.barh(labels, values, color="#55a868")
        ax.invert_yaxis()
        ax.set_xlabel(metric)
        ax.set_title(f"{task} leaderboard ({metric})")
        plt.tight_layout()

        path = os.path.join(output_dir, f"{task}_{metric}.png")
        fig.savefig(path, dpi=200)
        plt.close(fig)
        plots.append(path)

    return plots


def _tokenize_name(name: str) -> Set[str]:
    return {token for token in re.split(r"[/_\\.\-]+", name.lower()) if token}


def _build_base_catalog(results: Sequence[Dict[str, Any]]) -> List[Tuple[str, str]]:
    catalog: List[Tuple[str, str]] = []
    for res in results:
        raw_model = str(res.get("model", "model"))
        if "checkpoints" in raw_model:
            continue
        display_name = str(res.get("model_display_name", raw_model))
        catalog.append((display_name, raw_model))
    return catalog


def _infer_base_display(checkpoint_model: str, catalog: Sequence[Tuple[str, str]]) -> Optional[str]:
    checkpoint_tokens = _tokenize_name(Path(checkpoint_model).name)
    best_display: Optional[str] = None
    best_overlap = 0
    for display_name, raw_model in catalog:
        candidate_tokens = _tokenize_name(raw_model)
        overlap = len(checkpoint_tokens & candidate_tokens)
        if overlap > best_overlap and overlap > 0:
            best_display = display_name
            best_overlap = overlap
    return best_display


def _resolve_display_name(result: Dict[str, Any], catalog: Sequence[Tuple[str, str]], include_task: bool) -> str:
    task = str(result.get("task", "task"))
    if "model_display_name" in result:
        base_label = str(result["model_display_name"])
    else:
        raw_model = str(result.get("model", "model"))
        if "checkpoints" in raw_model or "tarp" in raw_model.lower():
            inferred = _infer_base_display(raw_model, catalog)
            if inferred:
                base_label = f"{inferred}-tarp" if not inferred.endswith("-tarp") else inferred
            else:
                base_label = Path(raw_model).name
        else:
            base_label = raw_model

    if include_task:
        return f"{base_label} ({task})"
    return base_label


def plot_metric_grid(
    results: Sequence[Dict[str, Any]],
    output_path: str = "docs/models_leaderboard.png",
    metrics: Optional[Sequence[Tuple[str, str]]] = None,
) -> Optional[str]:
    if not results:
        return None

    if metrics is None:
        metrics = [
            ("accuracy_test", "Accuracy"),
            ("nrs_overall", "Null Risk Score"),
            ("dcb_overall", "Default Class Bias"),
            ("null_entropy_overall", "Null Entropy"),
        ]

    os.makedirs(Path(output_path).parent, exist_ok=True)

    tasks = {str(res.get("task", "task")) for res in results}
    include_task = len(tasks) > 1
    base_catalog = _build_base_catalog(results)

    labels = []
    for res in results:
        labels.append(_resolve_display_name(res, base_catalog, include_task))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes = axes.flatten()

    for idx, (metric_key, title) in enumerate(metrics):
        ax = axes[idx]
        metric_labels: List[str] = []
        metric_values: List[float] = []
        for label, res in zip(labels, results):
            if metric_key in res:
                metric_labels.append(label)
                metric_values.append(float(res[metric_key]))

        if not metric_values:
            ax.set_visible(False)
            continue

        order = np.argsort(metric_values)[::-1]
        ordered_values = [metric_values[i] for i in order]
        ordered_labels = [metric_labels[i] for i in order]
        positions = np.arange(len(ordered_labels))

        ax.bar(positions, ordered_values, color="#4c72b0")
        ax.set_xticks(positions)
        ax.set_xticklabels(ordered_labels, rotation=35, ha="right")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2, linestyle="--")

        for pos, value in zip(positions, ordered_values):
            ax.text(pos, value, f"{value:.3f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle("Model Leaderboard", fontsize=16)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path
