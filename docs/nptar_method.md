# NP-TaR Method Guide

This document details how the NP-TaR stack links NullBench diagnostics to TaRP (Trace-and-Reweight Pretraining) mitigation. It summarizes the benchmark setup, derives the scoring formulas, and gives hands-on steps for tracing documents and comparing against external tracing systems such as OLMo Trace.

## 1. NullBench Benchmark

NullBench evaluates decoder-only checkpoints on dataset-specific instruction prompts. Supported tasks live under `nullbench/tasks/` (AG News, HateXplain, SST-2) and are invoked via `nullbench/scripts/run_nullbench_<task>.py`.

Each run logs metrics to `experiments/<TaskName>/<model>_<task>_<split>_results.json` and refreshes the per-task leaderboard PNG under `docs/`.

### Metrics

Given null inputs \(x_i^{\text{null}}\) produced by dataset generators and model logits \(p_\theta(y=k\mid x)\):

- **Default-Class Bias (DCB)**

  \[
  \text{DCB} = \frac{1}{N}\sum_{i=1}^{N} \max_k p_\theta \bigl(y = k \mid x_i^{\text{null}}\bigr)
  \]

  Measures how strongly the model collapses onto a single class when signal is absent (lower is better).

- **Null Entropy**

  \[
  H_{\text{null}} = -\frac{1}{N}\sum_{i=1}^{N}\sum_k p_\theta(y=k \mid x_i^{\text{null}})\log p_\theta(y=k \mid x_i^{\text{null}})
  \]

  Captures uncertainty on null inputs (higher is better).

- **Null Risk Score (NRS)**  
  Combines DCB and entropy to summarize collapse robustness (higher is better); see `docs/nullbench_tarp_paper.txt` for derivation.

Standard accuracy/recall is logged for the labeled split (`test`, `validation`, etc.), enabling before/after comparisons once mitigation is applied.

### Entry Commands

```bash
# Example: HateXplain, Gemma 3B, decoder-only format
python -m nullbench.scripts.run_nullbench_hatexplain \
  --model-name google/gemma-3-1b-it \
  --display-name google/gemma-3-1b-it \
  --split test \
  --max-length 512 \
  --batch-size 4
```

Replace the script for AG News or SST-2 as needed. Optional flags `--mitigation {cc|looc|dc}` let you layer contextual calibration or leave-one-out corrections directly within NullBench.

## 2. TaRP Mitigation Pipeline

`nulltrace/scripts/run_tarp_pipeline.py` converts NullBench findings into corpus-level sampling weights. It expects:

- One or more NullBench JSON result files (`--results` or `--results-glob`).
- A corpus JSONL (`--corpus`) containing `{id, text, label, ...}` records (see `data/*.jsonl` exports).
- Optional knobs for label tokens, context patterns, and reweighting strength.

### 2.1 Tracing Documents

`trace_corpus` (`nulltrace/tracing/pattern_indexer.py`) streams each document and counts:

- `label_counts[t]`: frequency of explicit label tokens \(t\) (e.g., `"normal"`, `"offensive"`, `"hate"`). Counts rely on whole-word regex `\b token \b`.
- `context_hits[t]`: counts of label-in-context expressions, using patterns like `"label <token>"`, `"category: <token>"`, etc. (customizable via `--context-pattern`).

Each trace stores `(doc_id, text_length, label_counts, context_hits, metadata)` in `document_traces.jsonl`.

### 2.2 Bias Score Formula

`nulltrace/reweighting/bias_scores.py` normalizes how “default-class heavy” a document is:

\[
b_d = \frac{\sum_{t\in \mathcal{D}} c_{d,t} + \lambda \sum_{t\in \mathcal{D}} h_{d,t}}
           {\sum_{t} c_{d,t} + \lambda \sum_{t} h_{d,t}}
\]

- \(c_{d,t}\): raw mentions of token \(t\) in document \(d\).
- \(h_{d,t}\): context hits for token \(t\).
- \(\mathcal{D}\): set of default label tokens inferred from NullBench `default_class_*` entries (or provided via `--label-token`).
- \(\lambda = \text{context_weight}\) (default `2.0`) emphasizes structured cues such as “label: Hate”.
- Scores are clamped to \([0, 1]\) and stored alongside metadata in `bias_scores.jsonl`.

### 2.3 Sampling Weight Formula

`nulltrace/reweighting/sampler.py` maps each bias score to a weight:

\[
w_d = 
\begin{cases}
w_{\max}, & b_d \le \tau \\
\max\bigl(w_{\min},\, w_{\min} + (w_{\max} - w_{\min}) \cdot \frac{1 - b_d}{1 - \tau + \epsilon}\bigr), & b_d > \tau
\end{cases}
\]

where:

- \(\tau = \text{bias\_threshold}\) (default `0.5`) marks the collapse boundary.
- \(w_{\min}\) / \(w_{\max}\) clamp the weighting range (defaults `0.1` and `1.0`).
- `--minority-label` with `--minority-bonus` multiplies \(w_d\) by \((1 + \text{bonus})\) for documents containing specified minority labels.

Output `reweighted_manifest.jsonl` holds `{doc_id, bias_score, sampling_weight, metadata}` per document.

### 2.4 Pipeline Command

```bash
python -m nulltrace.scripts.run_tarp_pipeline \
  --results experiments/hateXplain/google_gemma-3-1b-it_hatexplain_test_results.json \
  --task hatexplain \
  --corpus data/hatexplain_corpus.jsonl \
  --output-dir experiments/tarp_hatexplain \
  --context-pattern "label" --context-pattern "category" \
  --bias-threshold 0.5 --min-weight 0.2 --max-weight 1.0
```

`--task` filters result files so only HateXplain labels influence tracing, ensuring the inferred tokens match the dataset’s letter/label mapping.

## 3. Reweighted Fine-tuning

`nulltrace/finetune/run_reweighted_finetune.py` continues decoder-only pretraining with the TaRP manifest.

- Supports `--task {ag_news,hatexplain,sst2}` to pick the correct instruction template and decoder answer letters (see `TASK_PROMPT_CONFIGS`).
- Reads the corpus and manifest JSONLs, builds a `WeightedRandomSampler` where each document’s sampling probability is proportional to `sampling_weight`.
- Computes inverse-frequency class weights with optional exponential scaling (`--class-weight-power`) and minority boost (`--minority-class-boost`), ensuring rare gold labels still surface after down-weighting collapsed docs.
- Each example becomes `[instruction prompt tokens][answer letter tokens]`, masking the prompt (label `-100`) so loss focuses solely on the choice letter (plus optional EOS).

### Training Command

```bash
python -m nulltrace.finetune.run_reweighted_finetune \
  --task hatexplain \
  --base-model google/gemma-3-1b-it \
  --corpus data/hatexplain_corpus.jsonl \
  --manifest experiments/tarp_hatexplain/reweighted_manifest.jsonl \
  --output-dir nulltrace/finetune/checkpoints/gemma-3-1b-it-hatexplain-tarp \
  --epochs 1 \
  --batch-size 4 \
  --learning-rate 5e-5 \
  --max-length 512 \
  --gradient-accumulation-steps 2
```

Intermediate snapshots live under `checkpoint-step-*` every `--save-every` updates; the script also saves a final `output-dir` checkpoint suitable for `AutoModelForCausalLM.from_pretrained`.

### Post-finetune Evaluation

Evaluate the new checkpoint via NullBench to log improved collapse/accuracy metrics and refresh the leaderboard:

```bash
python -m nullbench.scripts.run_nullbench_hatexplain \
  --model-name nulltrace/finetune/checkpoints/gemma-3-1b-it-hatexplain-tarp \
  --display-name gemma-3-1b-it-hatexplain-tarp \
  --split test --max-length 512 --batch-size 4
```

## 4. Document Tracing Workflow

1. **Export corpus:** use scripts in `scripts/` (e.g., `export_hatexplain_validation.py`) to turn HF datasets into JSONL corpora with consistent `id`, `text`, and `label` fields.
2. **Baseline NullBench:** choose the relevant `run_nullbench_*` CLI to produce `*_results.json` (ensuring `default_class_*` fields capture the collapsed labels).
3. **Run TaRP tracing:** execute `run_tarp_pipeline` as shown above; inspect `document_traces.jsonl` and `bias_scores.jsonl` to spot documents saturated with default labels.
4. **Fine-tune:** feed the generated `reweighted_manifest.jsonl` into the finetune script.
5. **Re-evaluate:** NullBench the TaRP checkpoint and plot combined leaderboards via `python -m nullbench.scripts.plot_models_leaderboard`.

## 5. Comparing with OLMo Trace

OLMo Trace (from AI2) derives document influence scores using gradient-based tracing on OLMo checkpoints. To compare against TaRP:

1. **Generate TaRP bias data:** keep `bias_scores.jsonl` and `reweighted_manifest.jsonl` from the steps above.
2. **Run OLMo Trace:** obtain per-document influence scores \(I_d\) for the same corpus (consult OLMo documentation for setup, typically via `olmo trace --corpus ...`).
3. **Normalize for comparison:** align both measures (e.g., z-score each set) and build scatter plots of \((b_d, I_d)\) to see whether both pipelines flag the same documents.
4. **Cross-manifest testing:** convert OLMo influence scores into TaRP-compatible manifests (`{"doc_id": ..., "sampling_weight": ...}`) to reuse `run_reweighted_finetune.py` without code changes. This yields an A/B comparison:  
   - **TaRP weights:** heuristic token/context tracing.  
   - **OLMo weights:** gradient-based influence.
5. **Evaluate:** NullBench both fine-tuned models and contrast DCB/NRS/accuracy to quantify which tracing approach reduces collapse more effectively.

By capturing both the mathematical underpinnings and the concrete command lines, this guide should let you run the full NP-TaR pipeline end-to-end and fairly benchmark it against alternative tracing techniques like OLMo Trace.
