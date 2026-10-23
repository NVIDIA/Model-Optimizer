# Stage 1 — Configure a new model

Create `tools/launcher/examples/<Org>/<Model>/<config>.yaml` by **copying the closest
existing example and adapting it**. Pick a reference with the same algorithm and the
same shape as the target (dense vs MoE, similar size) from `tools/launcher/examples/`
— e.g. the Qwen3-8B config for a dense model.

The task structure, args, containers, and GPU/node sizing are all visible in the
existing examples — infer them from a reference rather than hand-rolling. This file
covers only what the examples don't make obvious.

## Step 1 — Pick the algorithm and the variant

Example filenames encode both: `hf_<mode>_<algorithm>.yaml`, where mode is `offline`
(dump hidden states first, then train on them), `online` (forward the base model at
training time), or `streaming`.

```bash
ls tools/launcher/examples/*/*/hf_*_<algorithm>.yaml
```

Offline is the default choice when the target model is too large to forward
alongside training. Task count follows from the variant, not from the algorithm — do
not assume a fixed number of tasks; copy the reference's layout.

## Step 2 — Fill in the algorithm-specific values

From the algorithm sheet (`../algorithms/<algorithm>.md`):

- **Pipeline tasks** — which script each task runs, and the artifact paths they pass
  between each other.
- **Recipe and training knobs** — the recipe path for the training task, and which
  overrides this model needs.
- **Per-model adjustments** — the non-obvious knobs that vary by target model.

For offline variants, the hidden-state dump task usually offers more than one
backend (vLLM / HF / TRT-LLM). The sheet's *Pipeline tasks* section says how to pick.

## Step 3 — Size the job

Copy node/GPU counts from the reference example, then sanity-check against the
target: the base model's BF16 weights must fit in the allocated GPU memory for the
serving and dump tasks, so scale `tensor-parallel-size`, `gpus_per_node`, or `nodes`
if the target is larger than the reference. Then fit every task to the cluster:

- Clusters that only accept whole nodes reject smaller jobs at `sbatch` (`QOSMinGRES` or
  `Requested node configuration is not available`; the launcher's `--mem=0` also trips
  clusters that scale RAM per GPU). Set every task's `gpus_per_node` to the node size
  (`sinfo -o '%P %G'`); serving tasks keep their TP.
- Training starts one process per visible GPU, so when a training task's GPU count
  changes, set `training.gradient_accumulation_steps` to keep nodes × GPUs × per-device
  batch × accumulation unchanged. If no integer does, ask the user.

## Step 4 — Preview

```bash
cd tools/launcher
uv run launch.py --yaml examples/<Org>/<Model>/<config>.yaml --dryrun
```

Check the resolved scripts, paths, and containers before submitting for real.
