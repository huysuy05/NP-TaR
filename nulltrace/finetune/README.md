# TaRP Fine-tuning Utilities

This module contains helper scripts to continue pre-training (or lightweight fine-tuning) a classification model using the TaRP reweighted manifest.

## Files

- `run_reweighted_finetune.py` – entry script that loads a base model, applies weighted sampling using `reweighted_manifest.jsonl`, and saves checkpoints.
- `checkpoints/` – placeholder directory for saving local checkpoints produced by the script.

## Usage

1. Generate NullBench results and run `nulltrace.scripts.run_tarp_pipeline` to obtain `reweighted_manifest.jsonl`.
2. Run the fine-tune script:

```bash
python -m nulltrace.finetune.run_reweighted_finetune \
  --base-model google/gemma-3-1b-it \
  --corpus data/ag_news_corpus.jsonl \
  --manifest experiments/tarp_ag_news/reweighted_manifest.jsonl \
  --output-dir nulltrace/finetune/checkpoints/gemma-3-1b-it-tarp \
  --epochs 1 \
  --batch-size 8 \
  --learning-rate 5e-5
```

3. Evaluate the newly saved checkpoint with the NullBench scripts to compare against the base model.
