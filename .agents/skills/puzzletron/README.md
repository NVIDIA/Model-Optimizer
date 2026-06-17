# Puzzletron Agent Skill

Puzzletron is an end-to-end workflow for model pruning and MIP-based architecture optimization.
This skill exposes it as a slash command for AI coding agents.

For full environment setup, model configuration, and algorithm details see
[examples/puzzletron/README.md](../../examples/puzzletron/README.md).

## Using with AI agents

> **Experimental:** AI agent integration is an experimental feature and may change.

| Agent | How to invoke |
|---|---|
| **Claude Code** | `/puzzletron <command>` in the chat |

## Commands

### `mip_sweep <nproc_per_node>`

Runs the MIP sweep across multiple compression rates. `nproc_per_node` is the number of GPUs per node.

```text
/puzzletron mip_sweep 4
```

Output is streamed live and also written to `./log.txt`.

### `mip_sweep progress`

Parses `./log.txt` and prints a structured progress report: prep steps, per-compression-rate
sub-steps, and a timing summary with elapsed and estimated remaining time.

```text
/puzzletron mip_sweep progress
```

Example output:

```text
Overall: Puzzletron step 7/8 — MIP sweep (6 compression rates)
──────────────────────────────────────────────────────────────
  Status      Phase                             Elapsed
──────────────────────────────────────────────────────────────
  [DONE]      Prep (teacher memory + rate list)     <1s
  [DONE]      compression_rate=0.5               3m 52s
  [RUNNING]   compression_rate=0.6 — validating (47/128 batches)   1m 14s
  [ ]         compression_rate=0.7               pending
  [ ]         compression_rate=0.8               pending
  [ ]         compression_rate=0.9               pending
  [ ]         compression_rate=1.0               pending
──────────────────────────────────────────────────────────────
  Started:   08:05:30
  Now:       08:10:56
  Elapsed:   5m 26s
  Completed: 1/6 compression rates  (avg 3m 52s/rate)
  Remaining: ~19m 22s estimated
```
