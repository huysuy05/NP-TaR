import json
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

from nullbench import NullBench
from nullbench.tasks.ag_news import AGNewsTask
from nullbench.generators.empty import generate_empty_inputs
from nullbench.generators.placeholder import generate_placeholders
from nullbench.generators.noise import generate_noise_inputs
from nullbench.generators.low_signal import generate_low_signal


# 1. Load a model (here we assume a 4-class AG News classifier)
MODEL_NAME = "textattack/distilbert-base-uncased-ag-news"


def load_model_and_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


tokenizer, model = load_model_and_tokenizer(MODEL_NAME)


def predict_proba_fn(texts):
    """Batch predict with HuggingFace model, return [N, K] numpy array."""
    all_probs = []
    batch_size = 32
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            logits = model(**enc).logits  # [B, K]
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        all_probs.append(probs)
    return np.concatenate(all_probs, axis=0)


def main():
    task = AGNewsTask()

    generators = {
        "empty": generate_empty_inputs,                              # uses n
        "placeholder": generate_placeholders,                        # uses n
        "noise": lambda texts_or_n: generate_noise_inputs(task.get_test_texts()),
        "low_signal": lambda texts_or_n: generate_low_signal(task.get_test_texts()),
    }

    bench = NullBench(task, generators, abstention_threshold=0.3)
    scores = bench.evaluate(predict_proba_fn)

    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
