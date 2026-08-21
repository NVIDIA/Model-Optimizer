# Stage 3 — Triage a failed run

Diagnose a failure in the draft-training pipeline: identify the failing task, find
the root cause, and give a fix plus a re-run command.

## Step 0 — Locate the experiment

Ask the user for one of:

- The experiment directory (e.g. the `--job-dir` passed to `launch.py` / `slurm.py`)
- The model name / YAML they ran

Find recent experiments under the job directory:

```bash
ls -td experiments/cicd/cicd_* | head -10
# or wherever --job-dir was pointed
```

Each experiment directory contains one subdirectory per task, each with a log file
whose name varies by launch mode (Slurm: `sbatch_*.out`, local Docker: `*.log`).

## Step 1 — Fetch logs for the failed task

Match the log files generally and read the tail of each — errors appear at the end:

```bash
find experiments/<exp_id>/ -type f \( -name '*.out' -o -name '*.log' \) | sort | while read -r f; do
  echo "=== $f ==="; tail -200 "$f"; echo
done
```

Find the first task with a non-zero exit code or an error message. Later tasks
usually fail only because an upstream artifact is missing, so fix the first one.

## Step 2 — Diagnose

Work through two tables. Start here — these failures are independent of the
algorithm and account for most runs:

| Error pattern | Root cause | Fix |
| --- | --- | --- |
| Server never becomes healthy (hangs at the health check) | Model too large for the allocated GPUs, or a server startup crash | Compare BF16 weight size against total allocated GPU memory; increase TP and/or nodes |
| `CUDA out of memory` during model load | Insufficient GPU memory | Reduce `--max-model-len`, or increase `--tensor-parallel-size` |
| `CUDA out of memory` during the hidden-state dump | Model too large for the chosen backend | Switch to a `device_map="auto"` backend, or increase TP |
| `CUDA out of memory` during training | Batch or sequence length too large | Reduce the recipe's training batch size or sequence length (see the algorithm sheet's *Recipe and training knobs*) |
| `CUDA out of memory` at benchmark | Target plus draft exceeds GPU memory | Increase TP |
| `pyxis: child terminated with signal 15` | SIGTERM — usually OOM | Increase TP or switch backends |
| `NCCL timeout` / `NCCL error` | Multi-node communication failure | Retry; reduce EP |
| `CANCELLED ... DUE TO TIME LIMIT` | Slurm wall-clock limit too short | Increase `--time`. Note that `afterany` dependencies let the next task start anyway. |
| `trust_remote_code` error | Model needs custom code but the flag isn't set | Add the flag to the serving task args (before the `--` separator) **and** to the benchmark task args |
| Vocab / tokenizer error | Missing tokenizer cache (e.g. a tiktoken cache) | Point the relevant cache env var at a pre-populated path |
| Architecture not supported by the serving engine | Engine version too old for this model | Try a newer container image |

Then check *Known failures* in `../algorithms/<algorithm>.md` for failures specific
to this algorithm — wrong script paths, missing scratchspace artifacts, export
failures, draft-config incompatibilities.

## Step 3 — Check for new-model issues

If the user is adding support for a new model, re-read *Per-model adjustments* in
`../algorithms/<algorithm>.md` and confirm each applicable knob was set — attention
type, MoE dimensions, custom tokenizer, and `trust_remote_code` are the usual
offenders.

If the architecture isn't recognized by the training code at all, that needs changes
in `modelopt/torch/speculative/` and a separate ModelOpt PR — no YAML change fixes it.

## Step 4 — Suggest fix and next steps

Provide:

1. **Root cause** — one-line summary
2. **Fix** — the specific config change, code edit, or command
3. **How to re-run** — skip earlier successful tasks by pointing at the existing
   scratchspace artifacts

To skip the first two tasks and re-run from the third:

```bash
uv run launch.py --yaml examples/<Org>/<Model>/<config>.yaml \
    pipeline.task_0.skip=true \
    pipeline.task_1.skip=true \
    --yes
```

To run a single task standalone, skip every other one:

```bash
uv run launch.py --yaml examples/<Org>/<Model>/<config>.yaml \
    pipeline.task_0.skip=true \
    pipeline.task_2.skip=true \
    pipeline.task_3.skip=true \
    --yes
```

## Step 5 — Record the failure pattern

If you hit a failure pattern not seen before, capture it in the team's internal
triage tracker — symptom, root cause, and fix — so the next engineer benefits. If
it's algorithm-specific, add a row to *Known failures* in the algorithm sheet; if it
applies to every algorithm, add it to Step 2 above.
