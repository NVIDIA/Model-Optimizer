# Legacy Nano campaign

The checked-in Nano experiment uses `mode: online_solutions` with the
`zero_shot_evaluation`, `aiperf`, and global distillation stages. This config
requires explicit commands to create the evaluation plan and materialize
selected finalists. Setup v2 represents those operations as campaign-DAG
nodes; its recommended flow includes evaluation and materialization.

## Prepare the dataset and MIP results

Use site-specific runner and execution configs. The Nano experiment pins its
public model source and revision but inherits a repository-relative
`dataset_path` from `base.yaml`. Override it with a materialized Hugging Face
dataset directory that is visible at the same path on every worker.

If needed, prepare Puzzle-KD from the full worker environment before starting
the controller:

```bash
export PUZZLETRON_DATASET=/shared/datasets/puzzle-kd-v2

python examples/puzzletron/materialize_dataset.py puzzle_kd_v2 \
  --output "$PUZZLETRON_DATASET" \
  --train-samples 8192 \
  --validation-samples 1024 \
  --seed 408
```

Pass the same dataset override to every controller invocation. First run the
campaign through MIP with the downstream stages disabled:

```bash
PUZZLETRON_EXPERIMENT=examples/puzzletron/configs/families/nemotron3/nano_30b_a3b_bf16/runs/default.yaml
PUZZLETRON_RUNNER=/path/to/runner.yaml
PUZZLETRON_EXECUTION=/path/to/execution.yaml
export PUZZLETRON_RUN_ROOT=/shared/puzzle_runs/my_campaign
export PUZZLETRON_DATASET=/shared/datasets/puzzle-kd-v2

python examples/puzzletron/orchestrate.py \
  --experiment "$PUZZLETRON_EXPERIMENT" \
  --runner "$PUZZLETRON_RUNNER" \
  --execution "$PUZZLETRON_EXECUTION" \
  --stage full \
  --override "dataset_path=$PUZZLETRON_DATASET" \
  --override zero_shot_evaluation.enabled=false \
  --override aiperf.enabled=false \
  --override global_distillation_sanity.enabled=false \
  --override global_distillation.enabled=false \
  --override post_distillation_evaluation.enabled=false
```

## Evaluate candidate profiles

Prepare the online evaluation plan, then run and aggregate its shards. These
profile IDs match the Nano example. For another legacy experiment, pass every
entry from its `zero_shot_evaluation.profile_ids` list.

```bash
python examples/puzzletron/run_profile_online_evaluation.py \
  --puzzle-dir "$PUZZLETRON_RUN_ROOT" \
  --profile-id params-075 \
  --profile-id runtime-075 \
  --profile-id memory-075 \
  --profile-id params-075-num-experts-only \
  --profile-id params-075-expert-dim-only \
  --profile-id params-075-num-experts-and-expert-dim \
  --prepare

python examples/puzzletron/orchestrate.py \
  --experiment "$PUZZLETRON_EXPERIMENT" \
  --runner "$PUZZLETRON_RUNNER" \
  --execution "$PUZZLETRON_EXECUTION" \
  --stage zero_shot_evaluation \
  --override "dataset_path=$PUZZLETRON_DATASET"
```

## Materialize finalists and resume

Materialize the evaluated finalists for the configured AIPerf profile from the
full worker environment. If the runner uses a container, enter it with the same
mounts before activating the worker virtual environment. For another legacy
experiment, use its `aiperf.profile_id` value.

```bash
# On the worker host or in the worker container:
cd /path/to/modelopt
source /path/to/full-modelopt-venv/bin/activate
export PUZZLETRON_RUN_ROOT=/shared/puzzle_runs/my_campaign

python examples/puzzletron/prepare_online_profile_finalists.py \
  --puzzle-dir "$PUZZLETRON_RUN_ROOT" \
  --config examples/puzzletron/configs/families/nemotron3/nano_30b_a3b_bf16/runs/default.yaml \
  --profile-id runtime-075 \
  --count 1
```

Return to the login node and resume the remaining enabled stages:

```bash
cd /path/to/modelopt
source .venv-puzzletron-control/bin/activate
PUZZLETRON_EXPERIMENT=examples/puzzletron/configs/families/nemotron3/nano_30b_a3b_bf16/runs/default.yaml
PUZZLETRON_RUNNER=/path/to/runner.yaml
PUZZLETRON_EXECUTION=/path/to/execution.yaml
export PUZZLETRON_RUN_ROOT=/shared/puzzle_runs/my_campaign
export PUZZLETRON_DATASET=/shared/datasets/puzzle-kd-v2

python examples/puzzletron/orchestrate.py \
  --experiment "$PUZZLETRON_EXPERIMENT" \
  --runner "$PUZZLETRON_RUNNER" \
  --execution "$PUZZLETRON_EXECUTION" \
  --stage full \
  --override "dataset_path=$PUZZLETRON_DATASET"
```

The final command resumes from verified completed stages. Do not use it as the
initial command for legacy configs with `mode: online_solutions`.
