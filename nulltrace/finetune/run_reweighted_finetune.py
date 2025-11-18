"""Run continued pre-training with TaRP reweighted sampling (decoder-only models)."""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

from nullbench.tasks.ag_news import format_ag_news_instruction, label_to_choice_text


def read_jsonl(path: str | Path) -> List[MutableMapping[str, object]]:
    records: List[MutableMapping[str, object]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, Mapping):
                records.append(dict(payload))
            else:
                raise ValueError(f"Invalid JSONL entry type: {type(payload)}")
    return records


class ReweightedDataset(Dataset):
    def __init__(
        self,
        records: List[Mapping[str, object]],
        weight_map: Dict[str, float],
        default_weight: float = 1.0,
        class_weights: Mapping[int, float] | None = None,
    ):
        self.records = records
        self.weights: List[float] = []
        for record in records:
            doc_id = str(record.get("id", len(self.weights)))
            weight = float(weight_map.get(doc_id, default_weight))
            label = int(record.get("label", 0))
            if class_weights is not None:
                weight *= float(class_weights.get(label, 1.0))
            self.weights.append(weight)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Mapping[str, object]:
        return self.records[idx]


def compute_class_weights(
    records: Iterable[Mapping[str, object]],
    power: float,
    minority_boost: float,
) -> Dict[int, float]:
    labels = [int(example.get("label", 0)) for example in records]
    counts = Counter(labels)
    if not counts:
        return {}
    total = sum(counts.values())
    num_classes = len(counts)
    base_weights: Dict[int, float] = {}
    for label, count in counts.items():
        # Higher weight for rarer classes (inverse frequency)
        base_weights[label] = total / (num_classes * count)

    if minority_boost > 1.0:
        minority_label = min(counts, key=counts.get)
        base_weights[minority_label] *= minority_boost

    if power != 1.0:
        for label, value in base_weights.items():
            base_weights[label] = value**power

    return base_weights


def assemble_decoder_example(
    text: str,
    label: int,
    tokenizer,
    max_length: int,
    append_eos: bool,
) -> Tuple[List[int], List[int], List[int]]:
    prompt = format_ag_news_instruction(text)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    answer_text = label_to_choice_text(label)
    answer_ids = tokenizer.encode(answer_text, add_special_tokens=False)
    if not answer_ids:
        raise ValueError(f"Answer text '{answer_text}' produced no tokens; adjust template spacing.")

    input_ids = prompt_ids + answer_ids
    labels = [-100] * len(prompt_ids) + answer_ids
    if append_eos and tokenizer.eos_token_id is not None:
        input_ids.append(tokenizer.eos_token_id)
        labels.append(-100)

    attention_mask = [1] * len(input_ids)
    if len(input_ids) > max_length:
        excess = len(input_ids) - max_length
        if excess >= len(prompt_ids):
            raise ValueError(
                "max_length is too small to retain the answer token. Increase --max-length."
            )
        input_ids = input_ids[excess:]
        attention_mask = attention_mask[excess:]
        labels = labels[excess:]
    return input_ids, attention_mask, labels


def pad_batch(sequences: Sequence[Tuple[List[int], List[int], List[int]]], pad_token_id: int):
    max_len = max(len(seq[0]) for seq in sequences)
    input_ids_batch = []
    attention_mask_batch = []
    labels_batch = []
    for input_ids, attention_mask, labels in sequences:
        pad_len = max_len - len(input_ids)
        input_ids_batch.append(input_ids + [pad_token_id] * pad_len)
        attention_mask_batch.append(attention_mask + [0] * pad_len)
        labels_batch.append(labels + [-100] * pad_len)

    return (
        torch.tensor(input_ids_batch, dtype=torch.long),
        torch.tensor(attention_mask_batch, dtype=torch.long),
        torch.tensor(labels_batch, dtype=torch.long),
    )


def build_dataloader(
    corpus_path: str,
    manifest_path: str,
    tokenizer,
    batch_size: int,
    max_length: int,
    class_weight_power: float,
    minority_class_boost: float,
) -> tuple[DataLoader, Dict[int, float]]:
    corpus = read_jsonl(corpus_path)
    manifest = read_jsonl(manifest_path)
    weight_map = {str(entry["doc_id"]): float(entry["sampling_weight"]) for entry in manifest}
    class_weights = compute_class_weights(corpus, class_weight_power, minority_class_boost)

    dataset = ReweightedDataset(corpus, weight_map, class_weights=class_weights)
    append_eos = tokenizer.eos_token_id is not None
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if pad_token_id is None:
        raise ValueError("Tokenizer must have either a pad token or an EOS token defined.")

    def collate_fn(batch: List[Mapping[str, object]]):
        encoded: List[Tuple[List[int], List[int], List[int]]] = []
        for example in batch:
            text = str(example.get("text", ""))
            label = int(example.get("label", 0))
            encoded.append(
                assemble_decoder_example(
                    text=text,
                    label=label,
                    tokenizer=tokenizer,
                    max_length=max_length,
                    append_eos=append_eos,
                )
            )

        input_ids, attention_mask, labels = pad_batch(encoded, pad_token_id=pad_token_id)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    sampler = WeightedRandomSampler(weights=dataset.weights, num_samples=len(dataset), replacement=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, collate_fn=collate_fn)
    return dataloader, class_weights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Continue decoder-only pre-training with TaRP reweighting on AG News"
    )
    parser.add_argument("--base-model", required=True, help="Base Hugging Face model or path")
    parser.add_argument("--corpus", required=True, help="Path to corpus JSONL used for continued pre-training")
    parser.add_argument("--manifest", required=True, help="Path to reweighted manifest JSONL")
    parser.add_argument("--output-dir", required=True, help="Where to save the fine-tuned checkpoint")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Mini-batch size")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--max-length", type=int, default=256, help="Max sequence length")
    parser.add_argument("--warmup-steps", type=int, default=0, help="Number of warmup steps")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="AdamW weight decay")
    parser.add_argument("--save-every", type=int, default=1000, help="Steps between intermediate saves")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1, help="Number of steps to accumulate before optimizer update")
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="Gradient clipping value (set <=0 to disable)")
    parser.add_argument("--class-weight-power", type=float, default=1.0, help="Exponent for class-balancing weights")
    parser.add_argument("--minority-class-boost", type=float, default=1.0, help="Additional multiplier for the minority class")
    return parser.parse_args()


def train():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    pad_added = False
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            tokenizer.add_special_tokens({"pad_token": "<pad>", "eos_token": "</s>"})
        tokenizer.pad_token = tokenizer.eos_token
        pad_added = True

    model = AutoModelForCausalLM.from_pretrained(args.base_model)
    if pad_added:
        model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    dataloader, class_weights = build_dataloader(
        corpus_path=args.corpus,
        manifest_path=args.manifest,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        class_weight_power=args.class_weight_power,
        minority_class_boost=args.minority_class_boost,
    )

    if class_weights:
        print("Class weights applied:")
        for label, weight in sorted(class_weights.items()):
            print(f"  label {label}: {weight:.4f}")

    grad_accum_steps = max(1, args.gradient_accumulation_steps)
    num_update_steps_per_epoch = math.ceil(len(dataloader) / grad_accum_steps)
    total_steps = args.epochs * num_update_steps_per_epoch

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps,
    )

    step = 0
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        accum_counter = 0
        for batch_idx, batch in enumerate(dataloader, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / grad_accum_steps
            loss.backward()
            running_loss += loss.item()
            accum_counter += 1

            should_update = batch_idx % grad_accum_steps == 0 or batch_idx == len(dataloader)
            if should_update:
                if args.max_grad_norm and args.max_grad_norm > 0:
                    clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step += 1

                avg_loss = running_loss / max(1, accum_counter)
                if step % 50 == 0 or batch_idx == len(dataloader):
                    print(f"Epoch {epoch+1}, update {step}/{total_steps}, loss={avg_loss:.4f}")

                running_loss = 0.0
                accum_counter = 0

                if step % args.save_every == 0:
                    save_path = Path(args.output_dir) / f"checkpoint-step-{step}"
                    save_path.mkdir(parents=True, exist_ok=True)
                    model.save_pretrained(save_path)
                    tokenizer.save_pretrained(save_path)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Final checkpoint saved to {output_dir}")


if __name__ == "__main__":
    train()
