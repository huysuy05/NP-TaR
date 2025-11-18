import argparse
import json

from nullbench import NullBench
from nullbench.generators.empty import generate_empty_inputs
from nullbench.generators.low_signal import generate_low_signal
from nullbench.generators.noise import generate_noise_inputs
from nullbench.generators.placeholder import generate_placeholders
from nullbench.scripts.decoder_eval_utils import (
    build_answer_token_ids,
    build_decoder_predict_fn,
    load_decoder_model,
)
from nullbench.tasks.ag_news import AGNewsTask


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
    return parser.parse_args()


def main():
    args = parse_args()
    model_name = args.model_name

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

    bench = NullBench(task, generators, abstention_threshold=0.3)
    scores = bench.evaluate(predict_proba_fn)
    scores["model"] = model_name

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(json.dumps(scores, indent=2))

    safe_model_name = model_name.replace("/", "_")
    output_file = f"experiments/{safe_model_name}_ag_news_results.json"
    import os

    os.makedirs("experiments", exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(scores, f, indent=2)
    print(f"\n✅ Results saved to: {output_file}")


if __name__ == "__main__":
    main()
