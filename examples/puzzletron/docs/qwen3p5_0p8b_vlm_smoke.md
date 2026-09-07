# Qwen 3.5 0.8B VLM pruning example

The Qwen 3.5 0.8B VLM example has two experiment files with distinct jobs:

| Experiment | Purpose | Execution profile |
| --- | --- | --- |
| `full_vlm_smoke.yaml` | Check the complete lifecycle on one GPU | `execution.single_gpu.yaml` |
| `vlm_campaign.yaml` | Run an illustrative multi-axis campaign | `qwen3p5_0p8b/execution.vlm_campaign.yaml` |

Start with `full_vlm_smoke.yaml`. It uses small workloads to check dataset
preparation, pruning, MIP, materialization, checkpoint evaluation, serving,
two-step VLM distillation, final evaluation, and resume. Its scores and
throughput are integration observations, not model-quality or production
performance results.

The campaign is a larger illustrative experiment. It is not a recommended
pruning recipe, training duration, or candidate-selection policy.

Unit tests compile these recipes and verify their stage and resource contracts.
They do not replace an end-to-end GPU run against the current model, data,
evaluator, and runtime dependencies.

## Prerequisites

Prepare the setup and worker environments described in
[environment setup](environment_setup.md). Workers need access to:

- the pinned `Qwen/Qwen3.5-0.8B` revision;
- the pinned Nemotron-VLM data revision;
- the evaluator datasets and media described in
  [VLM checkpoint evaluation](vlm_checkpoint_evaluation.md);
- a shared campaign output directory.

If workers cannot access the network, populate those caches before launch and
mount them through the runner. Do not replace the configured model identity
with a machine-specific path.

## Run the lifecycle smoke

Set paths visible to every worker:

```bash
export PUZZLETRON_DATASET_PATH=/path/to/qwen3p5-vlm-smoke-data
export PUZZLETRON_DATASET_REVISION=51f4f4d219315c3283950994d4eb3d7fc30aa87b
export PUZZLETRON_RUN_ROOT=/path/to/qwen3p5_0p8b_vlm_smoke
```

Prepare the eight image-conversation samples. This command is safe to rerun
when the existing manifest matches the same request.

```bash
python examples/puzzletron/materialize_dataset.py nemotron_vlm_v2 \
  --output "$PUZZLETRON_DATASET_PATH" \
  --revision "$PUZZLETRON_DATASET_REVISION" \
  --subsets sparsetables plotqa_cot wiki_en \
  --num-samples 8 \
  --max-shards-per-subset 1
```

Use the maintained experiment with the shared single-GPU execution profile and
a site-specific runner:

```bash
EXPERIMENT=examples/puzzletron/configs/families/qwen3_5/qwen3p5_0p8b/runs/full_vlm_smoke.yaml
EXECUTION=examples/puzzletron/configs/orchestration/execution.single_gpu.yaml
RUNNER=/path/to/site-specific/runner.slurm.yaml

python examples/puzzletron/orchestrate.py \
  --experiment "$EXPERIMENT" \
  --runner "$RUNNER" \
  --execution "$EXECUTION" \
  --stage full --dry-run
```

Inspect the resolved paths, resources, and stages. Then omit `--dry-run` to
launch. Rerun that same launch command to resume compatible completed work.

After completion, inspect the campaign report and verify that:

- image-text evaluation processed real image tensors;
- the selected checkpoint was physically materialized and reloaded;
- the fixed 24-row RealWorldQA, MMMU, and MVBench evaluation completed before
  and after KD;
- the serving requests and two KD steps completed with finite measurements;
- resuming submits no work for compatible completed stages.

The smoke deliberately uses tiny workloads. Run a separate benchmark with
representative requests before drawing performance conclusions.

## Inspect the illustrative campaign

The campaign enables hidden width, heterogeneous FFN width, depth,
grouped-attention geometry, and GDN geometry. All retained candidates use the
same frozen evaluator rows, teacher checkpoint, and 128-step example KD budget.

The flow is intentionally compact:

1. Generate parameter-constrained multi-axis candidates.
2. Retain five searched candidates by image-text LM loss.
3. Materialize and evaluate every retained checkpoint before KD.
4. Run 128 KD steps and evaluate again on the same 344 rows.
5. Apply the configured aggregate-rank rule to the searched candidates and
   measure serving for the selected result.

The 128-step value demonstrates the integration. It is not a convergence
criterion or recommended training duration. The aggregate-rank rule is also an
example policy rather than a general definition of the best model.

Compile the campaign before allocating resources:

```bash
EXPERIMENT=examples/puzzletron/configs/families/qwen3_5/qwen3p5_0p8b/runs/vlm_campaign.yaml
EXECUTION=examples/puzzletron/configs/orchestration/qwen3p5_0p8b/execution.vlm_campaign.yaml
RUNNER=/path/to/site-specific/runner.slurm.yaml
export PUZZLETRON_RUN_ROOT=/path/to/qwen3p5_0p8b_vlm_campaign

python examples/puzzletron/orchestrate.py \
  --experiment "$EXPERIMENT" \
  --runner "$RUNNER" \
  --execution "$EXECUTION" \
  --stage full --dry-run
```

The campaign has no checked-in expected-result baseline. Evaluate its output
against decision thresholds chosen for the target workload.

## Change the example

Keep omitted architecture dimensions at their teacher values. When adding an
axis, verify measurement and physical slicing on the target checkpoint before
expanding the campaign. Use [MIP profiles](mip_profiles.md#search-space) for the
search-space syntax and [configuration overrides](configuration_overrides.md)
for temporary changes.

Keep the model revision, frozen evaluator profile, KD exposure, and serving
workload fixed when comparing candidates. Changing any of them creates a
different experiment.
