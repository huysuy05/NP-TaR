"""Trace-and-Reweight pre-training pipeline entry point."""
from __future__ import annotations

import argparse
import json
import glob
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Set

from ..reweighting.bias_scores import attach_bias_scores
from ..reweighting.sampler import build_reweighted_manifest
from ..tracing.pattern_indexer import DocumentTrace, trace_corpus
from ..utils.io import stream_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the TaRP data pipeline.")
    parser.add_argument(
        "--results",
        nargs="+",
        default=[],
        help="Paths to NullBench result JSON files (accepts glob patterns)",
    )
    parser.add_argument(
        "--results-glob",
        action="append",
        dest="results_glob",
        help="Additional glob patterns for NullBench result JSON files",
    )
    parser.add_argument("--corpus", required=True, help="Path to the pre-training corpus in JSONL format")
    parser.add_argument("--output-dir", default="experiments/tarp", help="Where to store trace artifacts")
    parser.add_argument("--text-key", default="text", help="Field that contains document text")
    parser.add_argument("--doc-id-key", default="id", help="Field that contains document ids")
    parser.add_argument("--label-token", action="append", dest="label_tokens", help="Manually specify label tokens")
    parser.add_argument("--context-pattern", action="append", dest="context_patterns", help="Custom context patterns")
    parser.add_argument("--context-weight", type=float, default=2.0, help="Weight for context hits in bias score")
    parser.add_argument("--bias-threshold", type=float, default=0.5, help="Downweight bias score threshold")
    parser.add_argument("--min-weight", type=float, default=0.1, help="Minimum sampling weight")
    parser.add_argument("--max-weight", type=float, default=1.0, help="Maximum sampling weight")
    parser.add_argument("--minority-label", action="append", dest="minority_labels", help="Labels to upsample")
    parser.add_argument("--minority-bonus", type=float, default=0.25, help="Upsampling multiplier for minority labels")
    parser.add_argument(
        "--task",
        default=None,
        help="If set, only use NullBench results for this task when inferring label tokens",
    )
    return parser.parse_args()


def resolve_result_paths(paths: Sequence[str], glob_patterns: Sequence[str] | None) -> List[str]:
    resolved: List[str] = []
    for candidate in paths:
        matches = glob.glob(candidate)
        if matches:
            resolved.extend(matches)
        else:
            resolved.append(candidate)

    if glob_patterns:
        for pattern in glob_patterns:
            resolved.extend(glob.glob(pattern))

    # Deduplicate while preserving order
    seen: Set[str] = set()
    ordered: List[str] = []
    for path in resolved:
        norm = str(Path(path))
        if norm in seen:
            continue
        seen.add(norm)
        ordered.append(norm)
    return ordered


def load_results(paths: Sequence[str]) -> List[Mapping[str, object]]:
    payloads: List[Mapping[str, object]] = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as handle:
            payloads.append(json.load(handle))
    return payloads


def extract_default_labels(results: Iterable[Mapping[str, object]]) -> Dict[str, Set[str]]:
    task_to_labels: Dict[str, Set[str]] = {}
    for result in results:
        task = str(result.get("task", "task"))
        task_to_labels.setdefault(task, set())
        for key, value in result.items():
            if not key.startswith("default_class_"):
                continue
            if not isinstance(value, Mapping):
                continue
            label = value.get("label")
            if label:
                task_to_labels[task].add(str(label))
    return task_to_labels


def document_traces(
    corpus_path: str,
    label_tokens: Sequence[str],
    text_key: str,
    doc_id_key: str,
    context_patterns: Sequence[str] | None,
) -> List[DocumentTrace]:
    documents = stream_jsonl(corpus_path)
    traces = list(
        trace_corpus(
            documents=documents,
            label_tokens=label_tokens,
            context_patterns=context_patterns,
            text_key=text_key,
            doc_id_key=doc_id_key,
        )
    )
    return traces


def traces_to_dicts(traces: Sequence[DocumentTrace]) -> List[Mapping[str, object]]:
    serialized: List[Mapping[str, object]] = []
    for trace in traces:
        serialized.append(
            {
                "doc_id": trace.doc_id,
                "text_length": trace.text_length,
                "label_counts": trace.label_counts,
                "context_hits": trace.context_hits,
                "metadata": trace.metadata,
            }
        )
    return serialized


def main() -> None:
    args = parse_args()
    result_paths = resolve_result_paths(args.results, args.results_glob)
    if not result_paths:
        raise ValueError("No NullBench result files matched --results/--results-glob inputs")

    results = load_results(result_paths)
    if args.task:
        results = [record for record in results if record.get("task") == args.task]
        if not results:
            raise ValueError(f"No NullBench results matched task '{args.task}'")
    task_label_map = extract_default_labels(results)

    if args.label_tokens:
        label_tokens = sorted(set(args.label_tokens))
    else:
        combined: Set[str] = set()
        for labels in task_label_map.values():
            combined.update(labels)
        if not combined:
            raise ValueError("Could not infer label tokens; please pass --label-token")
        label_tokens = sorted(combined)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Tracing corpus {args.corpus} with {len(label_tokens)} label tokens ...")
    traces = document_traces(
        corpus_path=args.corpus,
        label_tokens=label_tokens,
        text_key=args.text_key,
        doc_id_key=args.doc_id_key,
        context_patterns=args.context_patterns,
    )
    print(f"Found {len(traces)} documents.")

    trace_path = output_dir / "document_traces.jsonl"
    write_jsonl(trace_path, traces_to_dicts(traces))
    print(f"Trace stats saved to {trace_path}")

    bias_metadata = attach_bias_scores(
        traces,
        default_label_tokens=label_tokens,
        context_weight=args.context_weight,
    )
    bias_path = output_dir / "bias_scores.jsonl"
    write_jsonl(bias_path, bias_metadata)
    print(f"Bias scores saved to {bias_path}")

    manifest = build_reweighted_manifest(
        bias_metadata,
        bias_threshold=args.bias_threshold,
        min_weight=args.min_weight,
        max_weight=args.max_weight,
        minority_labels=args.minority_labels,
        minority_bonus=args.minority_bonus,
    )
    manifest_path = output_dir / "reweighted_manifest.jsonl"
    write_jsonl(manifest_path, manifest)
    print(f"Sampling manifest saved to {manifest_path}")

    print("\nSummary of default labels:")
    for task, labels in task_label_map.items():
        label_list = ", ".join(sorted(labels)) or "(none)"
        print(f"  - {task}: {label_list}")

    print("\nTaRP pipeline completed.")


if __name__ == "__main__":
    main()
