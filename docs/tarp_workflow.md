# TaRP End-to-End Workflow

Use this diagram for any supported dataset (AG News, HateXplain, SST-2). Substitute dataset-specific corpus paths, NullBench scripts, and display names as needed.

```mermaid
flowchart LR
    subgraph Inputs
        A[Dataset corpus JSONL]
        B[Base decoder checkpoint]
    end

    A --> C[1. NullBench Evaluation<br/>run nullbench.scripts.run_nullbench_*]
    B --> C

    C --> D[2. TaRP Tracing & Reweighting<br/>run nulltrace.scripts.run_tarp_pipeline]
    D --> E[3. Reweighted Fine-tune<br/>run nulltrace.finetune.run_reweighted_finetune]
    E --> F[4. Post-Finetune Evaluation<br/>rerun NullBench script]
    F --> G[5. Plot Leaderboards<br/>python -m nullbench.scripts.plot_models_leaderboard]

    style Inputs fill:#fdfdfd,stroke:#666
    style C fill:#e3f2fd,stroke:#1e88e5,color:#0d47a1
    style D fill:#e8f5e9,stroke:#388e3c,color:#1b5e20
    style E fill:#fff8e1,stroke:#f9a825,color:#8c6d1f
    style F fill:#f3e5f5,stroke:#8e24aa,color:#4a148c
    style G fill:#eceff1,stroke:#455a64,color:#263238
```

**Step details**
- **1. NullBench Evaluation:** Generate collapse diagnostics (`experiments/<dataset>/*_results.json`) for the base model.
- **2. TaRP Tracing & Reweighting:** Feed those results plus the dataset corpus into the TaRP pipeline to write `document_traces.jsonl`, `bias_scores.jsonl`, and `reweighted_manifest.jsonl` under `experiments/tarp_<dataset>`.
- **3. Reweighted Fine-tune:** Continue pre-training the base checkpoint using the corpus + `reweighted_manifest.jsonl` to create `<model>-<dataset>-tarp` checkpoints.
- **4. Post-Finetune Evaluation:** Rerun the matching NullBench script to log the improved metrics for the TaRP checkpoint.
- **5. Plot Leaderboards:** Aggregate and visualize the latest runs with `python -m nullbench.scripts.plot_models_leaderboard` (per-dataset PNGs refresh automatically when evaluations finish).
