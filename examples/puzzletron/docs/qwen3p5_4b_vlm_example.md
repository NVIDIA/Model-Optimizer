# Qwen 3.5 4B VLM FFN example

This example defines a bounded Puzzletron route for `Qwen/Qwen3.5-4B`. It pins the public checkpoint revision, spans seven aligned FFN intermediate-size targets from light pruning through roughly 20% whole-model pruning, and separates the default pruning plan from checkpoint evaluation and distillation.

The default `mip_vlm_smoke.yaml` route:

- changes the language-model FFN intermediate size from 9216 to one of seven aligned widths from 8704 through 4608;
- uses eight image-conversation samples for width importance and two for sort sanity, width sanity, and replacement scoring;
- compiles the complete teacher-plus-seven-width FFN candidate grid and searches both an approximately 20%-pruned parameter target and an analytical weight-plus-KV serving-memory target;
- stops at MIP without materializing checkpoints, running benchmark evaluation, or starting KD.

The opt-in `full_vlm_smoke.yaml` route continues the selected candidate through image-text evaluation, physical checkpoint materialization, a fresh-process RealWorldQA smoke evaluation, two VLM KD steps, checkpoint reload, and final image-text evaluation. These limits check integration behavior; they do not establish model quality or performance.

## Prepare the inputs

Prepare the [Puzzletron worker environment](environment_setup.md), then choose worker-visible paths for the normalized dataset and run output. The dataset revision must be an immutable Hugging Face commit SHA.

```bash
export PUZZLETRON_DATASET_PATH=/path/to/qwen3p5-vlm-smoke-data
export PUZZLETRON_DATASET_REVISION=51f4f4d219315c3283950994d4eb3d7fc30aa87b
export PUZZLETRON_RUN_ROOT=/path/to/qwen3p5_4b_vlm_smoke

python examples/puzzletron/materialize_dataset.py nemotron_vlm_v2 \
  --output "$PUZZLETRON_DATASET_PATH" \
  --revision "$PUZZLETRON_DATASET_REVISION" \
  --subsets sparsetables plotqa_cot wiki_en \
  --num-samples 8 \
  --max-shards-per-subset 1
```

The source checkpoint is `Qwen/Qwen3.5-4B` at the immutable revision recorded in `configs/families/qwen3_5/qwen3p5_4b/model.yaml`. Cache that revision before launch when workers cannot access Hugging Face.

## Compile the default plan

Copy the runner template and replace every `REPLACE_WITH_` value with the reviewed scheduler, repository, environment, container, and mount settings for the target site.

```bash
EXPERIMENT=examples/puzzletron/configs/families/qwen3_5/qwen3p5_4b/runs/mip_vlm_smoke.yaml
EXECUTION=examples/puzzletron/configs/orchestration/execution.single_gpu.yaml
RUNNER_TEMPLATE=examples/puzzletron/configs/orchestration/qwen3p5_4b/runner.slurm.yaml
RUNNER="$PUZZLETRON_RUN_ROOT/runner.slurm.yaml"

mkdir -p "$PUZZLETRON_RUN_ROOT"
cp "$RUNNER_TEMPLATE" "$RUNNER"
${EDITOR:-vi} "$RUNNER"

python examples/puzzletron/orchestrate.py \
  --experiment "$EXPERIMENT" \
  --runner "$RUNNER" \
  --execution "$EXECUTION" \
  --stage full --dry-run
```

The compiled default plan should end at `mip` and request one GPU for every enabled stage. Inspect the complete plan, paths, and resource settings before removing `--dry-run`.

## Compile the opt-in lifecycle

Use the same dataset, runner, and execution files with the full experiment:

```bash
EXPERIMENT=examples/puzzletron/configs/families/qwen3_5/qwen3p5_4b/runs/full_vlm_smoke.yaml

python examples/puzzletron/orchestrate.py \
  --experiment "$EXPERIMENT" \
  --runner "$RUNNER" \
  --execution "$EXECUTION" \
  --stage full --dry-run
```

Confirm that the post-MIP order is `image_eval`, `best_vlm_loss`, `materialized`, `checkpoint_eval`, `short_vlm_kd`, `post_kd_checkpoint_eval`, `final_image_eval`, and `best`. `checkpoint_eval` verifies that the physically sliced Hugging Face checkpoint can be loaded by the bounded RealWorldQA evaluator. `post_kd_checkpoint_eval` performs the same reload check on the consolidated KD checkpoint.

The checked-in CPU tests validate configuration resolution, the full FFN candidate grid, resource counts, and both compiled DAGs. A real run is still required to establish checkpoint compatibility, memory use, finite metrics, and end-to-end behavior on the target GPU and worker image.
