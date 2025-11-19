import argparse
import json

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
from nullbench.tasks.ag_news import AGNewsTask
from nullbench.mitigations import MitigationConfig, build_mitigated_predict_fn


def parse_args():
    parser = argparse.ArgumentParser(description="Run NullBench on AG News with decoder-only models")
    parser.add_argument(
        "--model-name",
        default="google/gemma-3-1b-it",
        help="Hugging Face model id or local path (must be AutoModelForCausalLM-compatible)",
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
        help="Optional label stored in the results JSON for plotting/leaderboards",
    )
    parser.add_argument(
        "--mitigation",
        default="none",
        choices=["none", "cc", "looc", "dc"],
        help="Optional mitigation to wrap around the decoder logits",
    )
    parser.add_argument(
        "--mitigation-generator",
        default="placeholder",
        help="Generator name to use for calibration references",
    )
    parser.add_argument(
        "--mitigation-samples",
        type=int,
        default=512,
        help="Number of reference samples for mitigation calibration",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model_name = args.model_name
    if args.display_name:
        display_name = args.display_name
    else:
        # Use the last part of the path/name for display
        base_display = model_name.rstrip("/").split("/")[-1]
        display_name = f"{base_display}-{args.mitigation}" if args.mitigation != "none" else base_display

    print("\n" + "=" * 60)
    print(f"NULLBENCH EVALUATION: {model_name} on AG News")
    print("=" * 60)

    print("\nLoading task...")
    task = AGNewsTask(split="test")
    print(f"Task: {task.name}")
    print(f"Test examples: {len(task.get_test_texts())}")
    print(f"Classes: {task.labels}")
    print("Instructional prompting: enabled (decoder-friendly prompts wrap each input).")

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

    mitigation_config = MitigationConfig(
        method=args.mitigation,
        reference_generator=args.mitigation_generator,
        sample_size=args.mitigation_samples,
    )
    formatter = getattr(task, "format_with_instruction", None)
    predict_proba_fn = build_mitigated_predict_fn(
        predict_proba_fn,
        config=mitigation_config,
        generators=generators,
        task_texts=task.get_test_texts(),
        formatter=formatter,
    )

    bench = NullBench(task, generators, abstention_threshold=0.3)
    scores = bench.evaluate(predict_proba_fn)
    scores["model"] = model_name if args.mitigation == "none" else f"{model_name}-{args.mitigation}"
    scores["model_display_name"] = display_name
    scores["mitigation"] = args.mitigation

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(json.dumps(scores, indent=2))

    safe_model_name = model_name.replace("/", "_")
    if args.mitigation != "none":
        safe_model_name += f"_{args.mitigation}"
    output_file = f"experiments/ag_news/{safe_model_name}_ag_news_results.json"
    import os

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(scores, f, indent=2)
    print(f"\nResults saved to: {output_file}")

    # Auto-refresh dataset leaderboard plot
    result_records = load_results_from_files(["experiments/ag_news/*_results.json"])
    task_results = [record for record in result_records if record.get("task") == task.name]
    if task_results:
        leaderboard_path = f"docs/{task.name}_leaderboard.png"
        plot_metric_grid(task_results, output_path=leaderboard_path)
        print(f"Leaderboard updated: {leaderboard_path}")


if __name__ == "__main__":
    main()
