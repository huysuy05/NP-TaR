import json
import os
from typing import Any, Dict, Iterable, List, Optional

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
