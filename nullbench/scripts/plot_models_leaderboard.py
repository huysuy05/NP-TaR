"""Generate a multi-metric leaderboard plot across all stored NullBench results."""
from __future__ import annotations

import argparse
from typing import List, Sequence

from nullbench.plotting.leaderboard import load_results_from_files, plot_metric_grid


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot accuracy/NRS/DCB/entropy bars for all stored models.")
    parser.add_argument(
        "--results",
        nargs="+",
        default=["experiments/*/*_results.json"],
        help="Glob patterns pointing to NullBench result JSON files",
    )
    parser.add_argument(
        "--output",
        default="docs/models_leaderboard.png",
        help="Path where the combined leaderboard PNG will be written",
    )
    args = parser.parse_args()

    results = load_results_from_files(args.results)
    if not results:
        print("No result files matched the provided patterns; nothing to plot.")
        return

    output_path = plot_metric_grid(results, output_path=args.output)
    if output_path:
        print(f"Leaderboard saved to {output_path}")
    else:
        print("Unable to render leaderboard due to missing metrics.")


if __name__ == "__main__":
    main()
