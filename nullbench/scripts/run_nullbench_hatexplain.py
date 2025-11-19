import argparse
import json
import os

from nullbench import NullBench
from nullbench.generators.empty import generate_empty_inputs
from nullbench.generators.low_signal import generate_low_signal
from nullbench.generators.noise import generate_noise_inputs
from nullbench.generators.placeholder import generate_placeholders
from nullbench.plotting.leaderboard import load_results_from_files, plot_metric_grid
from nullbench.scripts.decoder_eval_utils import (
    build_answer_token_ids,
    build_decoder_predict_fn,
    load_decoder_model,
)
from nullbench.tasks.hate_xplain import HateXplainTask


def parse_args():
    parser = argparse.ArgumentParser(description="Run NullBench on HateXplain with decoder-only models")
    parser.add_argument(
        "--model-name",
        default="google/gemma-3-1b-it",
        help="Hugging Face model id or local path (AutoModelForCausalLM-compatible)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum total tokens (prompt + answer) fed into the model",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for decoder forward passes",
    )
    parser.add_argument(
        "--display-name",
        default=None,
        help="Optional label stored with the results for plotting",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "validation", "test"],
        help="Which HateXplain split to evaluate",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model_name = args.model_name
    display_name = args.display_name or model_name

    print("\n" + "=" * 60)
    print(f"NULLBENCH EVALUATION: {model_name} on HateXplain ({args.split})")
    print("=" * 60)

    print("\nLoading task...")
    task = HateXplainTask(split=args.split)
    print(f"Task: {task.name}")
    print(f"Examples: {len(task.get_test_texts())}")
    print(f"Classes: {task.labels}")

    generators = {
        "empty": generate_empty_inputs,
        "placeholder": generate_placeholders,
        "noise": lambda texts_or_n: generate_noise_inputs(task.get_test_texts()),
        "low_signal": lambda texts_or_n: generate_low_signal(task.get_test_texts()),
    }

    print(f"\nGenerators: {list(generators.keys())}")
    print("\nStarting evaluation (this may take several minutes)...\n")

    tokenizer, model, device = load_decoder_model(model_name)
    answer_token_ids = build_answer_token_ids(tokenizer, task.decoder_choice_texts)
    predict_proba_fn = build_decoder_predict_fn(
        tokenizer,
        model,
        device,
        answer_token_ids,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )

    bench = NullBench(task, generators, abstention_threshold=0.3)
    scores = bench.evaluate(predict_proba_fn)
    scores["model"] = model_name
    scores["model_display_name"] = display_name
    scores["task_split"] = args.split

    safe_model_name = model_name.replace("/", "_")
    output_file = f"experiments/hateXplain/{safe_model_name}_hatexplain_{args.split}_results.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(scores, f, indent=2)
    print(f"\nResults saved to: {output_file}")

    result_records = load_results_from_files(["experiments/hateXplain/*_results.json"])
    task_results = [record for record in result_records if record.get("task") == task.name]
    if task_results:
        leaderboard_path = f"docs/{task.name}_leaderboard.png"
        plot_metric_grid(task_results, output_path=leaderboard_path)
        print(f"Leaderboard updated: {leaderboard_path}")


if __name__ == "__main__":
    main()
