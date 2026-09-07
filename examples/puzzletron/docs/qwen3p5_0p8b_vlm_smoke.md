# Qwen 3.5 0.8B VLM pruning example

The Qwen 3.5 0.8B VLM example has two experiment files with distinct purposes:

| Experiment | Purpose | Execution profile |
| --- | --- | --- |
| `full_vlm_smoke.yaml` | Check the complete lifecycle in one reusable node allocation | `qwen3p5_0p8b/execution.vlm_smoke.yaml` |
| `vlm_campaign.yaml` | Run the longer scheduled multi-axis example | `qwen3p5_0p8b/execution.vlm_campaign.yaml` |

Both recipes select vLLM's Triton GDN prefill backend. On a fresh worker, the
FlashInfer GDN kernels can still be compiling when the server readiness check
expires. Selecting Triton avoids that cold-start failure and makes startup
predictable in the reviewed worker image; it is not a general performance
recommendation.

Start with `full_vlm_smoke.yaml`. It uses small workloads to check dataset
preparation, pruning, MIP, materialization, checkpoint evaluation, serving,
two-step VLM distillation, final evaluation, and resume. Its scores and
throughput are integration observations, not model-quality or production
performance results.

The campaign is a larger illustrative experiment intended for scheduled
integration validation, not routine development or presubmit use. It is not a
recommended pruning recipe, training duration, or candidate-selection policy.

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
mount them through the runner. Keep the experiment's public model repository
and revision unchanged; a local cache is only where workers obtain those files.

The accelerated profiles are qualified for the documented one-node/eight-GPU
environment. Other Slurm sites may need adaptation; see [Slurm
configuration](slurm_configuration.md#reusable-single-node-allocations).

## Run the lifecycle smoke

Set the shared cache, source identity, and run paths:

```bash
export HF_HOME=/path/to/huggingface-cache
export PUZZLETRON_SOURCE_REVISION="$(git rev-parse HEAD)"
export PUZZLETRON_DATASET_REVISION=51f4f4d219315c3283950994d4eb3d7fc30aa87b
export PUZZLETRON_RUN_ROOT=/path/to/qwen3p5_0p8b_vlm_smoke
export PUZZLETRON_RUN_ROOT="$(python -c 'import os; print(os.path.realpath(os.environ["PUZZLETRON_RUN_ROOT"]))')"
DATASET_PATH="$PUZZLETRON_RUN_ROOT/datasets/nemotron_vlm_v2"
```

`HF_HOME` is the shared destination for the evaluator cache and must be visible
to workers. With network access, the campaign's `prepare_dataset` stage fills
and validates that cache. For offline workers, populate it before launch using
the cache command in [VLM checkpoint evaluation](vlm_checkpoint_evaluation.md#cache-benchmark-data).
Keep `PUZZLETRON_SOURCE_REVISION` exported for both dry-run and launch; submitted
workers inherit it so controller and worker artifact identities stay identical.
The run-root normalization keeps provenance checks reproducible on systems
where a shared-storage alias traverses a symbolic link. Both canonical paths
must be visible to workers through the runner's mounts.

The launch includes a `prepare_dataset` worker stage that creates and validates
the eight image-conversation samples at `DATASET_PATH`, inside the campaign
root. It also prepares the configured evaluation cache. You do not need dataset
or model dependencies in the lightweight controller venv, and there is no
separate data-preparation command for the online first-run path.

Unlike the text-only example, this VLM route does not publish a separate
tokenized-dataset artifact. It keeps the image conversations in their native
format so the model processor can construct text and image inputs together.

Use the maintained experiment with its reusable eight-GPU execution profile
and a site-specific runner:

```bash
EXPERIMENT=examples/puzzletron/configs/families/qwen3_5/qwen3p5_0p8b/runs/full_vlm_smoke.yaml
EXECUTION=examples/puzzletron/configs/orchestration/qwen3p5_0p8b/execution.vlm_smoke.yaml
RUNNER=/path/to/site-specific/runner.slurm.yaml

python examples/puzzletron/orchestrate.py \
  --experiment "$EXPERIMENT" \
  --runner "$RUNNER" \
  --execution "$EXECUTION" \
  --stage full --dry-run
```

Inspect the outer one-node allocation plus every logical attempt, resource, and
stage. Then omit `--dry-run` to launch. Rerun that same launch command to
reattach or resume compatible completed work.

After completion, inspect the campaign report and verify that:

- image-text evaluation processed real image tensors;
- the selected checkpoint was physically materialized and reloaded;
- the fixed 24-row RealWorldQA, MMMU, and MVBench evaluation completed before
  and after KD;
- the serving requests and two KD steps completed with finite measurements;
- resuming submits no work for compatible completed stages.

The smoke deliberately uses tiny workloads. Run a separate benchmark with
representative requests before drawing performance conclusions.

## Run the longer illustrative campaign

The campaign enables hidden width (the model's shared transformer hidden size),
heterogeneous FFN width (per-block feed-forward intermediate sizes), depth,
grouped-attention geometry, and GDN geometry. All retained candidates use the
same frozen evaluator rows, teacher checkpoint, and 128-step example KD budget.

The flow is intentionally compact:

1. Generate parameter-constrained multi-axis candidates.
2. Retain five searched candidates by image-text LM loss.
3. Materialize and evaluate every retained checkpoint before KD.
4. Run 128 KD steps and evaluate again on the same 344 rows.
5. Apply the configured aggregate-rank rule to the searched candidates and
   measure serving for the selected result.

The execution profile splits replacement scoring into eight resident workers
across the three embedding widths. It shards initial image evaluation across
five tasks, then runs materialization, pre-KD evaluation, KD, and post-KD
evaluation for the five retained candidates across disjoint GPUs. Each stage
runs concurrently up to the configured capacity; final serving measures only
the selected candidate, and individual stages do not all consume eight GPUs.

The 128-step value demonstrates the integration. It is not a convergence
criterion or recommended training duration. The aggregate-rank rule is also an
example policy rather than a general definition of the best model.

After the lifecycle smoke succeeds, keep its controller venv and runner, choose
a new run root, and inspect the larger plan before allocating resources:

```bash
EXPERIMENT=examples/puzzletron/configs/families/qwen3_5/qwen3p5_0p8b/runs/vlm_campaign.yaml
EXECUTION=examples/puzzletron/configs/orchestration/qwen3p5_0p8b/execution.vlm_campaign.yaml
RUNNER=/path/to/site-specific/runner.slurm.yaml
export HF_HOME=/path/to/huggingface-cache
export PUZZLETRON_SOURCE_REVISION="$(git rev-parse HEAD)"
export PUZZLETRON_RUN_ROOT=/path/to/qwen3p5_0p8b_vlm_campaign
export PUZZLETRON_RUN_ROOT="$(python -c 'import os; print(os.path.realpath(os.environ["PUZZLETRON_RUN_ROOT"]))')"

python examples/puzzletron/orchestrate.py \
  --experiment "$EXPERIMENT" \
  --runner "$RUNNER" \
  --execution "$EXECUTION" \
  --stage full --dry-run
```

Remove `--dry-run` to launch the inspected plan. Rerun that identical launch
command to resume it.

The campaign's `prepare_dataset` worker stage creates its 512-sample dataset
under this new run root. Do not point both recipes at the same run root: the
smoke manifest records an eight-sample request and is intentionally not
rewritten in place.

The campaign has no checked-in expected-result baseline. Evaluate its output
against decision thresholds chosen for the target workload.

## Change the example

Keep omitted architecture dimensions at their teacher values. When adding an
axis, verify measurement and physical slicing on the target checkpoint before
expanding the campaign. Use [MIP profiles](mip_profiles.md#search-space) for the
search-space syntax and [configuration overrides](configuration_overrides.md)
for temporary changes. Change campaign inputs before launch; do not patch
resolved or generated artifacts inside an existing run root.

In this guide, hidden width means the model's transformer hidden size.
Heterogeneous FFN width is a separate axis that selects per-layer FFN
intermediate sizes; changing one does not implicitly change the other.

Keep the model revision, frozen evaluator profile, KD exposure, and serving
workload fixed when comparing candidates. Changing any of them creates a
different experiment.

The campaign evaluates saved checkpoints automatically and records the results
with each candidate. To check a checkpoint outside the campaign, follow
[VLM checkpoint evaluation](vlm_checkpoint_evaluation.md). That guide owns
environment and cache preflight, suite selection, exact commands, result files,
and diagnosis.

Use guided setup when you need help resolving the model, dataset, and site
settings. The [setup wizard guide](setup_wizard.md) owns its invocation,
profiles, inputs, and generated files. Select Qwen 3.5 0.8B and the
Nemotron-VLM v2 image-text dataset to generate this route with site-specific
settings. Hidden (residual/embedding) width, attention, GDN, and depth are
available through guided customization. Among the tracked routes, attention
and GDN reductions are enabled only by `vlm_campaign.yaml`.
Inspect every customized plan with `--dry-run` before launch. See
[configuration and overrides](configuration_overrides.md) for persistent and
temporary changes.
