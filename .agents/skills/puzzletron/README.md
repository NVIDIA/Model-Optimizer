# Puzzletron Agent Skill

Puzzletron is an end-to-end workflow for model pruning and MIP-based architecture optimization.
This skill exposes it as a slash command for AI coding agents and via natural language conversation.

For full environment setup, model configuration, and algorithm details see
[examples/puzzletron/README.md](../../examples/puzzletron/README.md).

> **Experimental:** AI agent integration is an experimental feature and may change.

Run `/puzzletron` with no arguments to see available commands.

## Running the full pipeline

To run the full 8-step pipeline, use the slash command (where the number is GPUs per node):

```text
/puzzletron all 2
```

Or in natural language:

```text
run puzzletron all for Llama-3.1-8B on 2 GPUs
```

Check progress with:

```text
/puzzletron all progress
```

Example output while running:

```text
Overall: Puzzletron full pipeline (steps 1–8)
────────────────────────────────────────────────────────────────────
  Status      Step  Description                          Elapsed
────────────────────────────────────────────────────────────────────
  [DONE]            1/8: starting puzzletron pipeline      0m 0s
  [DONE]            2/8: converting model to Puzzletron heterogeneous format (single-gpu)    0m 26s
  [DONE]            3/8: scoring pruning activations (multi-gpu)     9m 9s
  [DONE]            4/8: pruning the model and saving pruned checkpoints (single-gpu)    0m 57s
  [DONE]            5/8: building replacement library and subblock statistics (single-gpu)    0m 26s
  [RUNNING]         6/8: calculating one block scores (multi-gpu) (270/352 solutions)   100m 6s
  [ ]               7/8: running MIP and realizing models (multi-gpu)
  [ ]               8/8: puzzletron pipeline completed (multi-gpu)
────────────────────────────────────────────────────────────────────
  Started:   00:08:50
  Finished:  01:59:54 (in progress)
  Elapsed:   111m 4s
  Completed: 5/8 steps
  Remaining: 56m 24s estimated
```

Step 6 progress is tracked via completed `solution_N.json` files on disk for an accurate
remaining estimate. Step 7 (MIP sweep) shows per-rate progress once it starts.

## Running the MIP step

Start the MIP step by telling the agent how many GPUs per node to use:

```text
/puzzletron mip 4
```

Output is streamed live and also written to `./log.txt`. While it runs (or after it finishes),
check progress with:

```text
/puzzletron mip progress
```

Example output when complete:

```text
Overall: Puzzletron step 7/8 — MIP sweep (6 compression rates)
──────────────────────────────────────────────────────────────
  Status      Phase                              Elapsed
──────────────────────────────────────────────────────────────
  [DONE]      Prep (teacher memory + rate list)       <1s
  [DONE]      compression_rate=0.5                3m 52s
  [DONE]      compression_rate=0.6                4m 41s
  [DONE]      compression_rate=0.7                4m 46s
  [DONE]      compression_rate=0.8                3m 55s
  [DONE]      compression_rate=0.9                3m 55s
  [DONE]      compression_rate=1.0                3m 59s
──────────────────────────────────────────────────────────────
  Started:   08:05:30
  Finished:  08:30:38
  Elapsed:   25m 8s
  Completed: 6/6 compression rates
  Remaining: done estimated

  Results:   /workspace/puzzle_dir/mip_sweep_results.csv
```

While running, the report shows which rate is active, sub-step detail (MIP solver node count
or validation batch progress), and an estimated time remaining based on completed rates.

## Checking compressed model accuracy

Two commands are available depending on whether you ran a single constrained MIP solve or a sweep:

**Single constrained run** — teacher vs. solution_0 at the configured target memory:

```text
/puzzletron mip losses
```

**Sweep** — accuracy across all compression rates from the sweep CSV:

```text
/puzzletron mip sweep losses
```

Example `mip losses` output for Qwen3.5-0.8B (target 10,000 MiB):

| Metric | Teacher | Compressed (solution_0) |
|---|---|---|
| `target_memory` | 20,389 MiB | 10,000 MiB |
| `lm_loss` | 1.1067 | 3.8808 |
| `token_accuracy_top_1` | 0.7365 | 0.2915 |
| `token_accuracy_top_5` | 0.9079 | 0.5500 |
| `token_accuracy_top_10` | 0.9399 | 0.6451 |

## Evaluating compressed models on MMLU

### Listing checkpoints

After the MIP sweep, list all available checkpoints (teacher + all compressed solutions) with their index numbers:

```text
/puzzletron eval list
```

Example output:

```text
#     Label           MMLU    Path
--------------------------------------------------------------------------------------------
0     teacher                 /workspace/hf_models/meta-llama/Llama-3.2-3B-Instruct
1     10,000 MiB              .../target_memory_10000MiB.../solution_0
2     12,233 MiB              .../target_memory_12233_.../solution_0
3     14,272 MiB              .../target_memory_14272_.../solution_0
4     16,311 MiB              .../target_memory_16310_.../solution_0

Usage: /puzzletron eval mmlu <index>
       /puzzletron eval mmlu <index> --limit 10   (smoke test, default)
```

### Running MMLU evaluation

Evaluate a checkpoint by index (from `eval list`) or direct path. The default `--limit 10` runs a quick smoke test; omit it or use a larger value for a full run:

```text
/puzzletron eval mmlu 0                          # teacher, smoke test (10 questions)
/puzzletron eval mmlu 1 --limit 1000             # compressed, 1000 questions
/puzzletron eval mmlu /path/to/hf/model          # direct path
```

Results are saved to `<checkpoint>/eval_results/mmlu/`.

### Checking eval progress

When running multiple checkpoints sequentially, monitor progress with:

```text
/puzzletron eval progress
```

Running rows show the current lm-eval phase and that phase's ETA. During the final loglikelihood phase, they also show the overall ETA, including result-saving time. Context-building progress is aggregated across MMLU subtasks instead of resetting for every task. Saved full evaluations take precedence over smoke tests; when only a limited result exists, its accuracy is labeled `(limit=N)`.

Example output mid-run:

```text
MMLU eval progress  (3/8 done)
──────────────────────────────────────────────────────────────────
  Status      Checkpoint       MMLU acc  Path
──────────────────────────────────────────────────────────────────
  [DONE]      teacher            0.5038  /workspace/hf_models/Qwen/Qwen3.5-0.8B
  [DONE]      10,000 MiB         0.2365  .../target_memory_10000MiB.../solution_0
  [DONE]      10,194 MiB         0.2417  .../target_memory_10194_.../solution_0
  [RUNNING]   12,233 MiB            ...  .../target_memory_12233_.../solution_0
  [ ]         14,272 MiB        pending
  [ ]         16,311 MiB        pending
──────────────────────────────────────────────────────────────────
  Done:    3/8
  Running: 12,233 MiB
  Pending: 14,272 MiB, 16,311 MiB
```

### Viewing combined sweep + MMLU results

Ask Claude to show both internal sweep losses and MMLU scores in one table. It joins the sweep CSV with per-checkpoint MMLU JSON results:

```text
  rate    target_mem    actual_mem     num_params   lm_loss   top_1   top_5  top_10    MMLU
----------------------------------------------------------------------------------------------------
  teacher                                                                               0.5038
    0.50      10,194.4      10,143.3    888,813,280    3.2367  0.3663  0.6384  0.7251  0.2417
    0.60      12,233.2      11,719.5    909,901,856    2.6377  0.4434  0.7198  0.7981  0.2446
    0.70      14,272.1      14,083.8    941,534,720    1.8532  0.5855  0.8176  0.8735  0.2374
    0.80      16,311.0      15,660.1    962,623,296    1.5385  0.6448  0.8576  0.9046  0.2636
    0.90      18,349.9      18,024.4    994,256,160    1.2447  0.7064  0.8914  0.9278  0.3119
    1.00      20,388.7      20,388.7  1,025,889,024    1.1067  0.7365  0.9079  0.9399  0.5038
```

## Distillation

Distillation fine-tunes the compressed student model to recover accuracy after pruning. It uses the teacher model as a supervision signal (KL-divergence on logits).

### Tokenizing a dataset

Tokenize a HuggingFace dataset into Megatron binary format before running distillation:

```text
/puzzletron distill tokenize \
  --hf_dataset nvidia/Nemotron-Post-Training-Dataset-v2 \
  --output_dir /workspace/hf_datasets/tokenized_nemotron \
  --tokenizer /workspace/hf_models/meta-llama/Llama-3.2-3B-Instruct
```

The command prints the output prefix and token count (`.bin` file size ÷ 4 bytes).

### Running distillation

Run distillation for a MIP solution. `--train_iters` is auto-computed as 1 epoch when `--data_paths` is provided:

```text
/puzzletron distill run \
  --puzzle_dir /workspace/puzzle_dir_llama3_2-3b \
  --ratio 0.9 \
  --nproc_per_node 8 \
  --output_dir /workspace/puzzle_dir_llama3_2-3b/distillation/0.9x-nemotron \
  --data_paths 1.0 /workspace/hf_datasets/tokenized_nemotron/train_text_document
```

The job runs in the background, writing to `./log.txt`. Progress is shown immediately after launch.

For a quick smoke test with mock data:

```text
/puzzletron distill run \
  --puzzle_dir /workspace/puzzle_dir_llama3_2-3b \
  --ratio 0.9 \
  --nproc_per_node 8 \
  --train_iters 10 \
  --use_mock_data
```

### Listing distillation runs

```text
/puzzletron distill list
```

Example output:

```text
Distillation runs — /workspace/puzzle_dir_llama3_2-3b
────────────────────────────────────────────────────────────────────
  0.9x          10 iters   HF exported   0.9x/hf
  0.9x-nemotron 10 iters   HF exported   0.9x-nemotron/hf
  0.9x-real     10 iters   HF exported   0.9x-real/hf
```

### Checking distillation progress

```text
/puzzletron distill progress
```

Example output for a running full-epoch distillation:

```text
Distillation progress — /workspace/puzzle_dir_llama3_2-3b
────────────────────────────────────────────────────────────────────

  Ratio:      0.9x-nemotron-full
  Output dir: /workspace/puzzle_dir_llama3_2-3b/distillation/0.9x-nemotron-full
  Status:     RUNNING  (iter 2410/3719)
  Started:    07:12:25
  Elapsed:    14m 28s
  Iter time:  0.3s/iter (avg last 5)
  Remaining:  ~6m 13s (1309 iters left)
  HF export:  not yet
  Log file:   /workspace/Model-Optimizer/log.txt

  Train loss: █▅▅▄▄▄▄▃▃▃▃▃▂▂▂▂▂▂▁▁▁▁▁▁▁  (0.278 → 0.131)
  Val loss:   █▇▆▆▅▇▄▄▄▄▄▄▃▃▂▂▂▂▂▁▁▁▁▁  (0.237 → 0.134)
  Convergence: CONVERGING  (-4.0% over last 3 checkpoints)
  Student CE: █▇▆▅▅▄▄▃▃▂▂▁  (2.840 → 2.410)
```

The sparklines show objective-loss and validation student-CE history — bars descend as
training improves. `Student CE` is parsed from the validation `lm loss value`; it is separate
from the objective loss optimized by distillation. Convergence verdict:
- `CONVERGING` — >2% improvement over last 3 validation checkpoints
- `DIMINISHING RETURNS` — 0.5–2% improvement
- `PLATEAU` — <0.5% change (safe to stop)
- `DIVERGING` — loss increasing (consider stopping)

Loss data is read live from `./log.txt` while the run is active; falls back to TensorBoard for stopped runs.

## Adding support for a new model

See [adding_new_model_tutorial.md](adding_new_model_tutorial.md) for a step-by-step walkthrough
covering: diagnosing why a model isn't supported, upgrading Transformers, writing a model
descriptor and converter, creating YAML configs, and a final checklist.
