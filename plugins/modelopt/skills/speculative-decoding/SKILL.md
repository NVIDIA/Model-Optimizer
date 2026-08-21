---
name: speculative-decoding
description: >-
  Train, debug, and validate a speculative decoding draft model (EAGLE3, DFlash,
  DSpark, Domino) through the ModelOpt launcher pipeline. Use when the user wants
  to add a new model to a draft-training pipeline, asks why a pipeline run failed,
  wants experiment logs reviewed, or wants to check whether a run's acceptance rate
  meets threshold. Triggers on "EAGLE3", "DFlash", "DSpark", "draft model",
  "acceptance rate", "speculative decoding pipeline". Do NOT use for quantizing a
  model (use ptq) or serving a checkpoint (use deployment).
user_invocable: true
---

# Speculative Decoding Draft-Model Training

Everything needed to take a target model from "no draft head" to "validated
acceptance rate" lives in this directory. Work through the stages below in order
for a new model; jump straight to a stage when you already know which one you need.

## Two axes: stage and algorithm

The pipeline is the same shape for every draft-model algorithm — synthesize data,
dump base-model hidden states, train the draft, benchmark acceptance rate. What
changes between algorithms is which training script and recipe run, which knobs
matter, and which failures are typical.

So this skill is split along those two axes, and **you almost always read one file
from each**:

| Axis | Directory | What it holds |
| --- | --- | --- |
| Stage | `references/stages/` | The procedure — algorithm-independent |
| Algorithm | `references/algorithms/` | The data sheet — scripts, recipe, knobs, thresholds, known failures |

Read the stage file for *what to do*, and the algorithm sheet for *the values to
plug in*. When a stage file says "see the algorithm sheet", it means the section of
`references/algorithms/<algorithm>.md` with the matching heading.

## Stages

| Stage | Reference | Use when |
| --- | --- | --- |
| 1. Configure | `references/stages/configure.md` | Adding a model that has no pipeline YAML yet |
| 2. Review logs | `references/stages/review-logs.md` | A run finished (or died) and you want a pass/fail summary |
| 3. Triage | `references/stages/triage.md` | A task failed and you need root cause plus a fix |
| 4. Validate | `references/stages/validate.md` | All tasks passed and you need to confirm the acceptance rate gate |

Review-logs and triage overlap by design: review-logs is the fast sweep across all
tasks, triage is the deep dive into one failing task. Start with review-logs unless
the user already knows which task broke.

## Algorithms

All recipes live in `modelopt_recipes/general/speculative_decoding/<algorithm>.yaml`.

| Algorithm | Sheet | Family |
| --- | --- | --- |
| EAGLE3 | `references/algorithms/eagle3.md` | Autoregressive draft head |
| DFlash | `references/algorithms/dflash.md` | Block diffusion |
| DSpark | `references/algorithms/dspark.md` | DFlash backbone + Markov head + confidence head |
| Domino | `references/algorithms/domino.md` | DFlash backbone + GRU causal correction head |

DSpark and Domino are **DFlash variants**, not separate pipelines: same
`recipe_type: speculative_dflash`, same training script, same `dflash.*` config
namespace, selected by `dflash_architecture_config.projector_type`. Read
`references/algorithms/dflash.md` first, then the variant's sheet for the delta.

If the user's algorithm has no sheet yet, the stage procedures still apply — derive
the missing values from an existing launcher example for that algorithm
(`tools/launcher/examples/*/*/hf_*_<algorithm>.yaml`) and its recipe, then write the
sheet as you go. `references/algorithms/README.md` defines what a sheet must contain.

## End-to-end: a new model

1. Confirm the algorithm and find the closest existing launcher example.
2. **Configure** — write the pipeline YAML (`references/stages/configure.md`).
3. Preview with `--dryrun`, then submit:

   ```bash
   cd tools/launcher
   uv run launch.py --yaml examples/<Org>/<Model>/<config>.yaml --yes
   ```

4. Register the job and set up monitoring per the **monitor skill**.
5. **Review logs** when it finishes (`references/stages/review-logs.md`).
6. **Triage** anything that failed (`references/stages/triage.md`), fix, re-run only
   the failed tasks onward via `pipeline.task_N.skip=true`.
7. **Validate** once all tasks pass (`references/stages/validate.md`).

Model-support gaps that need code changes land in `modelopt/torch/speculative/` and
require a separate ModelOpt PR — the pipeline YAML alone cannot fix an unrecognized
architecture.
