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
  [ ]               7/8: pending
  [ ]               8/8: pending
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

## Adding support for a new model

See [adding_new_model_tutorial.md](adding_new_model_tutorial.md) for a step-by-step walkthrough
covering: diagnosing why a model isn't supported, upgrading Transformers, writing a model
descriptor and converter, creating YAML configs, and a final checklist.
