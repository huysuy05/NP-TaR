"""Run continued pre-training with TaRP reweighting using MLX (Apple Silicon optimized)."""
import argparse
import math
import time
from pathlib import Path
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx_lm import load
from mlx_lm.utils import save_model
from transformers import AutoTokenizer
from tqdm import tqdm

# Import data processing logic from the PyTorch script to ensure consistency
from nulltrace.finetune.run_reweighted_finetune import (
    read_jsonl,
    compute_class_weights,
    assemble_decoder_example,
)

def get_batch_iterator(
    corpus,
    manifest_path,
    tokenizer,
    batch_size,
    max_length,
    class_weight_power,
    minority_class_boost
):
    # Load manifest and compute weights
    manifest = read_jsonl(manifest_path)
    weight_map = {str(entry["doc_id"]): float(entry["sampling_weight"]) for entry in manifest}
    class_weights = compute_class_weights(corpus, class_weight_power, minority_class_boost)
    
    if class_weights:
        print("Class weights applied:")
        for label, weight in sorted(class_weights.items()):
            print(f"  label {label}: {weight:.4f}")

    # Compute individual weights for the corpus
    weights = []
    for record in corpus:
        doc_id = str(record.get("id", len(weights)))
        w = float(weight_map.get(doc_id, 1.0))
        label = int(record.get("label", 0))
        if class_weights:
            w *= float(class_weights.get(label, 1.0))
        weights.append(w)
    
    # Normalize weights for numpy.random.choice
    weights = np.array(weights)
    weights /= weights.sum()
    
    indices = np.arange(len(corpus))
    
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    
    while True:
        # Sample indices based on weights
        batch_indices = np.random.choice(indices, size=batch_size, p=weights, replace=True)
        batch_records = [corpus[i] for i in batch_indices]
        
        input_ids_list = []
        labels_list = []
        
        for rec in batch_records:
            text = str(rec.get("text", ""))
            label = int(rec.get("label", 0))
            # Reuse the assembly logic
            i_ids, _, lbls = assemble_decoder_example(
                text, label, tokenizer, max_length, append_eos=True
            )
            input_ids_list.append(i_ids)
            labels_list.append(lbls)
            
        # Pad batch
        max_len_batch = max(len(x) for x in input_ids_list)
        
        padded_input_ids = []
        padded_labels = []
        
        for i_ids, lbls in zip(input_ids_list, labels_list):
            pad_len = max_len_batch - len(i_ids)
            padded_input_ids.append(i_ids + [pad_id] * pad_len)
            padded_labels.append(lbls + [-100] * pad_len)
            
        yield np.array(padded_input_ids), np.array(padded_labels)

def loss_fn(model, input_ids, labels):
    logits = model(input_ids)
    # Shift for causal LM loss: logits[..., :-1, :] predicts labels[..., 1:]
    logits = logits[:, :-1, :]
    labels = labels[:, 1:]
    
   
    loss = nn.losses.cross_entropy(logits, labels, reduction="none")
    
    mask = (labels != -100)
    loss = loss * mask
    
    # Normalize by valid tokens
    return loss.sum() / mask.sum()

def train():
    parser = argparse.ArgumentParser(description="MLX Fine-tuning for TaRP")
    parser.add_argument("--base-model", required=True, help="Base Hugging Face model or path")
    parser.add_argument("--corpus", required=True, help="Path to corpus JSONL")
    parser.add_argument("--manifest", required=True, help="Path to reweighted manifest JSONL")
    parser.add_argument("--output-dir", required=True, help="Where to save the checkpoint")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Mini-batch size")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--max-length", type=int, default=256, help="Max sequence length")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay")
    parser.add_argument("--save-every", type=int, default=1000, help="Steps between saves")
    parser.add_argument("--class-weight-power", type=float, default=1.0)
    parser.add_argument("--minority-class-boost", type=float, default=1.0)
    args = parser.parse_args()

    print(f"Loading model {args.base_model} with MLX...")
    model, _ = load(args.base_model)
    
    # Use HF tokenizer for consistency with data prep
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.train()

    optimizer = optim.AdamW(learning_rate=args.learning_rate, weight_decay=args.weight_decay)
    
    # Compile the step function
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    # @mx.compile
    def step(input_ids, labels):
        loss, grads = loss_and_grad_fn(model, input_ids, labels)
        optimizer.update(model, grads)
        return loss

    corpus = read_jsonl(args.corpus)
    steps_per_epoch = len(corpus) // args.batch_size
    total_steps = args.epochs * steps_per_epoch
    
    batch_iter = get_batch_iterator(
        corpus, args.manifest, tokenizer, args.batch_size, args.max_length,
        args.class_weight_power, args.minority_class_boost
    )
    
    print(f"Starting training for {total_steps} steps...")
    pbar = tqdm(range(total_steps))
    
    for i in pbar:
        input_ids_np, labels_np = next(batch_iter)
        
        # Convert to MLX arrays
        input_ids = mx.array(input_ids_np)
        labels = mx.array(labels_np)
        
        loss = step(input_ids, labels)
        
        # Evaluate loss to sync and update progress bar
        loss_val = loss.item()
        pbar.set_postfix(loss=f"{loss_val:.4f}")
        
        if (i + 1) % args.save_every == 0:
            save_path = Path(args.output_dir) / f"checkpoint-step-{i+1}"
            save_path.mkdir(parents=True, exist_ok=True)
            save_model(save_path, model)
            tokenizer.save_pretrained(save_path)

    # Final save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_model(output_dir, model)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved final weights to {output_dir}")

if __name__ == "__main__":
    train()
