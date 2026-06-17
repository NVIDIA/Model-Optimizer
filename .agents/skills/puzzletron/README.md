# Puzzletron Agent Skill

Puzzletron is an end-to-end workflow for model pruning and MIP-based architecture optimization.
This skill exposes it as a slash command for AI coding agents.

For full environment setup, model configuration, and algorithm details see
[examples/puzzletron/README.md](../../examples/puzzletron/README.md).

> **Experimental:** AI agent integration is an experimental feature and may change.

Run `/puzzletron` with no arguments to see available commands.

## Running the MIP sweep

Start the sweep by telling the agent how many GPUs per node to use:

```text
/puzzletron mip_sweep 4
```

Output is streamed live and also written to `./log.txt`. While it runs (or after it finishes),
check progress with:

```text
/puzzletron mip_sweep progress
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
  Completed: 6/6 compression rates  (avg 4m 11s/rate)
  Remaining: done estimated

  Results:   /workspace/puzzle_dir/mip_sweep_results.csv
```

While running, the report shows which rate is active, sub-step detail (MIP solver node count
or validation batch progress), and an estimated time remaining based on completed rates.
