# Decoder-Only Workflow

The AG News pipeline now assumes decoder-only (causal LM) checkpoints for both
NullBench evaluation and TaRP fine-tuning.

## NullBench Evaluation

```bash
python -m nullbench.scripts.run_nullbench_ag_news \
  --model-name google/gemma-3-1b-it \
  --display-name google/gemma-3-1b-it \
  --max-length 512 \
  --batch-size 4
  # Optional: add --mitigation cc --mitigation-generator placeholder --mitigation-samples 512
```

- Each article is wrapped with the dataset-specific instruction prompt
  ("You are an expert news editor...").
- Class probabilities come from the decoder logits for the answer tokens
  (letters `A`/`B`/`C`/`D`). Ensure your tokenizer maps each `" A"` style token
  to a single vocab entry; adjust spacing if needed.
- Mitigations (LOOC, contextual calibration, direct counting) are opt-in via
  `--mitigation {cc|looc|dc}`. The command keeps writing results under
  `experiments/ag_news/` with the method suffix (e.g., `gemma-3-1b-it_cc`).

## TaRP Continued Pre-training

```bash
python -m nulltrace.finetune.run_reweighted_finetune \
  --base-model google/gemma-3-1b-it \
  --corpus data/ag_news_corpus.jsonl \
  --manifest experiments/tarp_ag_news/reweighted_manifest.jsonl \
  --output-dir nulltrace/finetune/checkpoints/gemma-agnews-tarp \
  --epochs 1 --batch-size 4 --max-length 512
```

- The finetune script builds the same instruction prompt and appends the
  gold letter token, supervising only that answer while masking the prompt.
- Keep `--max-length` large enough to hold the prompt + answer; the script will
  raise if it needs to truncate the gold token.
- Weighted sampling, gradient accumulation, and class balancing continue to
  work as before.

## HateXplain Evaluation

```bash
python -m nullbench.scripts.run_nullbench_hatexplain \
  --model-name google/gemma-3-1b-it \
  --display-name gemma-3-1b-it \
  --split test \
  --max-length 512 \
  --batch-size 4
```

- Uses the same decoder-only instruction stack (“You are moderating an online forum…”) so
  the logits come from the answer letters `" A"/" B"/" C"`.
- Results are stored under `experiments/hateXplain/*_hatexplain_<split>_results.json` and automatically
  refresh `docs/hatexplain_leaderboard.png` so every past run stays visible.
- Optional mitigations mirror AG News: use `--mitigation cc|looc|dc` to wrap the
  logits without touching the base script.

## SST-2 (GLUE) Evaluation

```bash
python -m nullbench.scripts.run_nullbench_sst2 \
  --model-name google/gemma-3-1b-it \
  --display-name gemma-3-1b-it \
  --split validation \
  --max-length 512 \
  --batch-size 4
```

- Instruction prompt frames the sample as a movie review and expects only the choice letter.
- Outputs land in `experiments/sst2/*_sst2_<split>_results.json` and trigger
  `docs/sst2_leaderboard.png` updates after each evaluation.
- Mitigation flags work the same here if you want to calibrate SST-2 runs.

> **Note:** All dataset scripts accept `--display-name`, letting you label fine-tuned checkpoints
> as `base-model-tarp` (or any custom string) for the per-task leaderboards. Run
> `python -m nullbench.scripts.plot_models_leaderboard` whenever you want a combined
> multi-task view in `docs/models_leaderboard.png`.

## End-to-End Pipeline Runner

To automate the entire workflow (base eval → TaRP reweighting → fine-tune → eval fine-tune → plot):

```bash
chmod +x run_full_pipeline.sh
./run_full_pipeline.sh ag_news data/ag_news_corpus.jsonl

# Example overriding split/base model
./run_full_pipeline.sh hatexplain data/hatexplain_corpus.jsonl test google/gemma-3-1b-it
```

- Arguments: `DATASET` (`ag_news`, `hatexplain`, `sst2`), `CORPUS_JSONL`, optional `SPLIT`, optional
  `BASE_MODEL` (defaults to `google/gemma-3-1b-it`).
- Environment variables allow tuning hyperparameters (e.g., `FINETUNE_EPOCHS`, `FINETUNE_BATCH_SIZE`,
  `MAX_LENGTH`). See the `run_full_pipeline.sh` header for the complete list.
- The script reuses the per-dataset evaluation modules, runs `nulltrace.scripts.run_tarp_pipeline`, kicks
  off `nulltrace.finetune.run_reweighted_finetune`, evaluates the fine-tuned checkpoint, and finally
  refreshes `docs/<dataset>_leaderboard.png` via the plotting CLI.
