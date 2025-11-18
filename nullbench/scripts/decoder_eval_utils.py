"""Shared helpers for decoder-only NullBench evaluations."""
from __future__ import annotations

from typing import Callable, List, Sequence

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_decoder_model(model_name: str):
    """Load a causal LM along with its tokenizer and target device."""
    print(f"Loading decoder model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    pad_added = False
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            tokenizer.add_special_tokens({"pad_token": "<pad>", "eos_token": "</s>"})
        tokenizer.pad_token = tokenizer.eos_token
        pad_added = True

    model = AutoModelForCausalLM.from_pretrained(model_name)
    if pad_added:
        model.resize_token_embeddings(len(tokenizer))

    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    model = model.to(device)
    print(f"Model loaded on {device}")
    return tokenizer, model, device


def build_answer_token_ids(tokenizer, choice_texts: Sequence[str]) -> List[int]:
    """Map decoder choice texts to single-token ids."""
    ids: List[int] = []
    for text in choice_texts:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if len(token_ids) != 1:
            raise ValueError(
                "Each decoder choice text must map to exactly one token. "
                f"Got {token_ids} for choice '{text}'. Consider adjusting spacing."
            )
        ids.append(token_ids[0])
    return ids


def build_decoder_predict_fn(
    tokenizer,
    model,
    device,
    answer_token_ids: Sequence[int],
    max_length: int,
    batch_size: int,
) -> Callable[[Sequence[str]], np.ndarray]:
    """Return a predict_proba_fn compatible with NullBench."""

    def predict_proba_fn(texts: Sequence[str]):
        all_probs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                outputs = model(**enc)
                logits = outputs.logits  # [B, T, V]

            attention_mask = enc["attention_mask"]
            last_indices = attention_mask.sum(dim=1) - 1
            next_token_logits = logits[torch.arange(len(batch)), last_indices]
            candidate_logits = next_token_logits[:, answer_token_ids]
            probs = torch.softmax(candidate_logits, dim=-1).cpu().numpy()
            all_probs.append(probs)

            if (i // batch_size + 1) % 10 == 0:
                print(f"  Processed {min(i + batch_size, len(texts))}/{len(texts)} texts")

        return np.concatenate(all_probs, axis=0)

    return predict_proba_fn
