import os
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np


def plot_model_metrics(
    model_name: str,
    results: Dict[str, Any],
    output_dir: str = "docs",
    metrics: Optional[List[str]] = None,
) -> str:
    """
    Plot detailed metrics for a single model across different tasks or generators.
    Each graph shows multiple metrics (null score, dcb, entropy, recall per class)
    with different line styles and colors.

    Args:
        model_name: Name of the model being evaluated.
        results: Dictionary containing evaluation results with metrics.
                 Expected to have keys for different metrics and possibly tasks.
        output_dir: Directory where the plot is saved.
        metrics: List of metric keys to plot. If None, uses default set.

    Returns:
        Path to the saved plot.
    """
    if metrics is None:
        # Default metrics to plot
        metrics = ["nrs_overall", "dcb", "entropy"]
    
    os.makedirs(output_dir, exist_ok=True)

    # Define line styles and colors for different metrics
    style_map = {
        "nrs_overall": {"color": "#4c72b0", "linestyle": "-", "marker": "o", "label": "Null Score"},
        "dcb": {"color": "#dd8452", "linestyle": "--", "marker": "s", "label": "DCB"},
        "entropy": {"color": "#55a868", "linestyle": "-.", "marker": "^", "label": "Entropy"},
        "recall_per_class": {"color": "#c44e52", "linestyle": ":", "marker": "D", "label": "Avg Recall"},
        "nrs_per_generator": {"color": "#8172b3", "linestyle": "-", "marker": "v", "label": "NRS per Generator"},
    }

    # Extract data for plotting
    # Check if results contain per-generator scores or other categorical data
    categories = []
    metric_data = {metric: [] for metric in metrics}
    
    # Try to extract nrs_per_generator if available
    if "nrs_per_generator" in results:
        gen_scores = results["nrs_per_generator"]
        if isinstance(gen_scores, dict):
            categories = list(gen_scores.keys())
            for metric in metrics:
                if metric == "nrs_per_generator":
                    metric_data[metric] = [gen_scores[cat] for cat in categories]
                elif metric in results:
                    # Replicate scalar metrics across categories
                    metric_data[metric] = [results[metric]] * len(categories)
                else:
                    metric_data[metric] = [0.0] * len(categories)
    else:
        # If no per-generator data, just use the metrics as single points
        categories = ["Overall"]
        for metric in metrics:
            if metric in results:
                metric_data[metric] = [float(results[metric])]
            else:
                metric_data[metric] = [0.0]
    
    # Handle recall_per_class specially
    if "recall_per_class" in results:
        recalls = results["recall_per_class"]
        if isinstance(recalls, (list, np.ndarray)):
            if len(categories) == 1:
                # Show individual class recalls
                categories = [f"Class {i}" for i in range(len(recalls))]
                metric_data["recall_per_class"] = [float(r) for r in recalls]
                # Replicate other metrics
                for metric in metrics:
                    if metric != "recall_per_class" and metric in results:
                        metric_data[metric] = [float(results[metric])] * len(categories)
            else:
                avg_recall = float(np.mean(recalls))
                metric_data["recall_per_class"] = [avg_recall] * len(categories)
        else:
            metric_data["recall_per_class"] = [float(recalls)] * len(categories)
    
    x_positions = np.arange(len(categories))
    
    fig, ax = plt.subplots(figsize=(max(10, len(categories) * 1.5), 6))

    # Plot each metric
    for metric in metrics:
        values = metric_data.get(metric, [])
        
        if values and any(v != 0.0 for v in values):  # Only plot if there are non-zero values
            style = style_map.get(metric, {"color": "gray", "linestyle": "-", "marker": "o", "label": metric})
            ax.plot(
                x_positions,
                values,
                color=style["color"],
                linestyle=style["linestyle"],
                marker=style["marker"],
                linewidth=2,
                markersize=8,
                label=style["label"],
            )

    # Also plot recall per class if not in metrics list but available
    if "recall_per_class" not in metrics and "recall_per_class" in metric_data:
        values = metric_data["recall_per_class"]
        if values and any(v != 0.0 for v in values):
            style = style_map["recall_per_class"]
            ax.plot(
                x_positions,
                values,
                color=style["color"],
                linestyle=style["linestyle"],
                marker=style["marker"],
                linewidth=2,
                markersize=8,
                label=style["label"],
            )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(categories, rotation=45, ha="right")
    ax.set_xlabel("Category")
    ax.set_ylabel("Metric Value")
    ax.set_title(f"Metrics for {model_name}")
    ax.legend(loc="best", frameon=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout()

    # Sanitize model name for filename
    safe_model_name = model_name.replace("/", "_").replace(" ", "_")
    path = os.path.join(output_dir, f"{safe_model_name}_metrics_plot.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    
    return path


def plot_multi_model_metrics(
    results_list: List[Dict[str, Any]],
    output_dir: str = "docs",
    metrics: Optional[List[str]] = None,
) -> List[str]:
    """
    Plot detailed metrics for multiple models, one plot per model.

    Args:
        results_list: List of result dicts, each containing:
                      {"model": "model_name", "task": "task_name", ...metrics...}
        output_dir: Directory where the plots are saved.
        metrics: List of metric keys to plot. If None, uses default set.

    Returns:
        List of paths to the saved plots (one per model).
    """
    if metrics is None:
        metrics = ["nrs_overall", "dcb", "entropy"]
    
    os.makedirs(output_dir, exist_ok=True)
    plots = []

    # Group results by model
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for res in results_list:
        model = res.get("model", "model")
        grouped.setdefault(model, []).append(res)

    # Define line styles and colors for different metrics
    style_map = {
        "nrs_overall": {"color": "#4c72b0", "linestyle": "-", "marker": "o", "label": "Null Score"},
        "dcb": {"color": "#dd8452", "linestyle": "--", "marker": "s", "label": "DCB"},
        "entropy": {"color": "#55a868", "linestyle": "-.", "marker": "^", "label": "Entropy"},
        "recall_per_class": {"color": "#c44e52", "linestyle": ":", "marker": "D", "label": "Avg Recall"},
    }

    for model_name, model_results in grouped.items():
        if not model_results:
            continue

        # Sort by task name for consistent ordering
        model_results = sorted(model_results, key=lambda x: x.get("task", ""))
        
        tasks = [res.get("task", "task") for res in model_results]
        x_positions = np.arange(len(tasks))

        fig, ax = plt.subplots(figsize=(max(10, len(tasks) * 1.5), 6))

        # Plot each metric
        for metric in metrics:
            values = []
            for res in model_results:
                if metric in res:
                    values.append(float(res[metric]))
                else:
                    values.append(0.0)
            
            if any(values):  # Only plot if there are non-zero values
                style = style_map.get(metric, {"color": "gray", "linestyle": "-", "marker": "o", "label": metric})
                ax.plot(
                    x_positions,
                    values,
                    color=style["color"],
                    linestyle=style["linestyle"],
                    marker=style["marker"],
                    linewidth=2,
                    markersize=8,
                    label=style["label"],
                )

        # Also plot average recall per class if recall_per_class exists
        recall_values = []
        for res in model_results:
            if "recall_per_class" in res:
                recalls = res["recall_per_class"]
                if isinstance(recalls, (list, np.ndarray)):
                    avg_recall = float(np.mean(recalls))
                else:
                    avg_recall = float(recalls)
                recall_values.append(avg_recall)
            else:
                recall_values.append(0.0)
        
        if any(recall_values) and "recall_per_class" not in metrics:
            style = style_map["recall_per_class"]
            ax.plot(
                x_positions,
                recall_values,
                color=style["color"],
                linestyle=style["linestyle"],
                marker=style["marker"],
                linewidth=2,
                markersize=8,
                label=style["label"],
            )

        ax.set_xticks(x_positions)
        ax.set_xticklabels(tasks, rotation=45, ha="right")
        ax.set_xlabel("Task")
        ax.set_ylabel("Metric Value")
        ax.set_title(f"Metrics for {model_name}")
        ax.legend(loc="best", frameon=True, shadow=True)
        ax.grid(True, alpha=0.3, linestyle="--")
        plt.tight_layout()

        # Sanitize model name for filename
        safe_model_name = model_name.replace("/", "_").replace(" ", "_")
        path = os.path.join(output_dir, f"{safe_model_name}_metrics_plot.png")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        plots.append(path)

    return plots
