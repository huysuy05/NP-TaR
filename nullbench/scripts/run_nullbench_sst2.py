import json
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

from nullbench import NullBench
from nullbench.tasks.sst2 import SST2Task
from nullbench.generators.empty import generate_empty_inputs
from nullbench.generators.placeholder import generate_placeholders
from nullbench.generators.noise import generate_noise_inputs
from nullbench.generators.low_signal import generate_low_signal


# Load a model for SST-2 sentiment classification (2-class: positive, negative)
MODEL_NAME = "google/gemma-3-1b-it"


def load_model_and_tokenizer(model_name: str):
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Model loaded on {device}")
    
    return tokenizer, model, device


tokenizer, model, device = load_model_and_tokenizer(MODEL_NAME)


def predict_proba_fn(texts):
    """Batch predict with HuggingFace model, return [N, K] numpy array."""
    all_probs = []
    batch_size = 16  # Smaller batch size for larger models
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tokenizer(
            batch, 
            padding=True, 
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            logits = model(**enc).logits  # [B, K]
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        all_probs.append(probs)
        
        if (i // batch_size + 1) % 10 == 0:
            print(f"  Processed {min(i + batch_size, len(texts))}/{len(texts)} texts")
    
    return np.concatenate(all_probs, axis=0)


def main():
    print("\n" + "="*60)
    print(f"NULLBENCH EVALUATION: {MODEL_NAME} on SST-2")
    print("="*60)
    
    print("\nLoading task...")
    task = SST2Task(split='validation')
    print(f"Task: {task.name}")
    print(f"Test examples: {len(task.get_test_texts())}")
    print(f"Classes: {task.labels}")

    generators = {
        "empty": generate_empty_inputs,
        "placeholder": generate_placeholders,
        "noise": lambda texts_or_n: generate_noise_inputs(task.get_test_texts()),
        "low_signal": lambda texts_or_n: generate_low_signal(task.get_test_texts()),
    }
    
    print(f"\nGenerators: {list(generators.keys())}")
    print("\nStarting evaluation (this may take several minutes)...\n")

    bench = NullBench(task, generators, abstention_threshold=0.3)
    scores = bench.evaluate(predict_proba_fn)
    
    # Add model metadata
    scores['model'] = MODEL_NAME

    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(json.dumps(scores, indent=2))
    
    # Save results
    output_file = f"experiments/{MODEL_NAME}_sst2_results.json"
    import os
    os.makedirs("experiments", exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(scores, f, indent=2)
    print(f"\n✅ Results saved to: {output_file}")


if __name__ == "__main__":
    main()
