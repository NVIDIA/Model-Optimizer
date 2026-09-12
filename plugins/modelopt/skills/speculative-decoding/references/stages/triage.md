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
| `CUDA out of memory` **while loading weights** (before the KV cache is allocated) | The weights themselves don't fit | Increase `--tensor-parallel-size`, add nodes, or switch backend. `--max-model-len` will **not** help — it doesn't change weight memory. |
| `CUDA out of memory` **after weights load** — KV-cache allocation, or during a forward pass | Activation / KV-cache pressure | Reduce `--max-model-len`, batch size, or concurrency; raising TP also helps by splitting the cache |
| `CUDA out of memory` during the hidden-state dump | Model too large for the chosen backend | Switch to a `device_map="auto"` backend, or increase TP |
| `CUDA out of memory` during training | Batch or sequence length too large | Reduce the recipe's training batch size or sequence length (see the algorithm sheet's *Recipe and training knobs*) |
| `CUDA out of memory` at benchmark | Target plus draft exceeds GPU memory | Increase TP |
| `pyxis: child terminated with signal 15` | SIGTERM — usually OOM | Increase TP or switch backends |
| `NCCL timeout` / `NCCL error` | Multi-node communication failure | Retry; reduce EP |
| `CANCELLED ... DUE TO TIME LIMIT` | Slurm wall-clock limit too short | Increase `--time`. Note that `afterany` dependencies let the next task start anyway. |
| `trust_remote_code` error | Model needs custom code but the flag isn't set for **that** task | Set it on **every** task that loads the model — see the spellings below |
| Vocab / tokenizer error | Missing tokenizer cache (e.g. a tiktoken cache) | Point the relevant cache env var at a pre-populated path |
| Architecture not supported by the serving engine | Engine version too old for this model | Try a newer container image |

### `trust_remote_code` spellings

The flag is spelled differently per task type, so setting it once is not enough — a
custom-code model needs it everywhere it is loaded:

| Task type | How to set it |
| --- | --- |
| Serving / benchmark | CLI flag before the `--` separator: `--trust-remote-code` (vLLM) or `--trust_remote_code` (trtllm-serve) |
| Hidden-state dump | `TRUST_REMOTE_CODE: "1"` in the task `environment` |
| Training | `model.trust_remote_code=true` as an OmegaConf override |
| Streaming serve replicas | `SERVE_EXTRA_ARGS: "--trust-remote-code"` |
| Export | `EXPORT_EXTRA_ARGS: "--trust_remote_code"` |

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

Re-runs work by adding `pipeline.task_N.skip=true` for each task you want to skip.
**Read the task list out of the config first** — task count varies (EAGLE3 offline is
4, DFlash offline is 2, Domino is 2), so there is no fixed set of skip flags:

```bash
grep -n '^  task_[0-9]*:' examples/<Org>/<Model>/<config>.yaml
```

To resume from a failed task, skip every task before it. For a 4-task EAGLE3 offline
config whose `task_2` failed:

```bash
uv run launch.py --yaml examples/<Org>/<Model>/<config>.yaml \
    pipeline.task_0.skip=true \
    pipeline.task_1.skip=true \
    --yes
```

To run one task standalone, skip every other task in that config. For the same 4-task
config, running only `task_1`:

```bash
uv run launch.py --yaml examples/<Org>/<Model>/<config>.yaml \
    pipeline.task_0.skip=true \
    pipeline.task_2.skip=true \
    pipeline.task_3.skip=true \
    --yes
```

Both are EAGLE3-offline examples — translate the flags to the config at hand rather
than copying them verbatim.

## Step 5 — Record the failure pattern

If you hit a failure pattern not seen before, capture it in the team's internal
triage tracker — symptom, root cause, and fix — so the next engineer benefits. If
it's algorithm-specific, add a row to *Known failures* in the algorithm sheet; if it
applies to every algorithm, add it to Step 2 above.
