# Qwen 3.5 4B VLM FFN-width 10%-to-20% KD search

This campaign compares three smaller versions of `Qwen/Qwen3.5-4B`. Each version
reduces the language model's FFN width while leaving the vision path unchanged.

This page lists the campaign runs and the commands shared by all runs. Each run
summary owns its results, runtime, and limitations so that those details are not
repeated here.

## Runs

| Run | Status | What finished |
|---|---|---|
| [2026-09-01-r2](runs/2026-09-01-r2/summary.md) | Early comparison | Three students, a serving check, 64 KD steps, and two short quality benchmarks |

## Reproduce

Use the model and dataset versions recorded in the run's structured record.
Prepare the dataset where workers can read it, and replace every placeholder in
the runner template before launching.

```bash
export PUZZLETRON_DATASET_PATH=/path/to/qwen3p5-vlm-campaign-data
export PUZZLETRON_DATASET_REVISION=51f4f4d219315c3283950994d4eb3d7fc30aa87b
export PUZZLETRON_RUN_ROOT=/path/to/puzzle_runs/qwen3p5_4b_vlm_campaign

python examples/puzzletron/materialize_dataset.py nemotron_vlm_v2 \
  --output "$PUZZLETRON_DATASET_PATH" \
  --revision "$PUZZLETRON_DATASET_REVISION" \
  --subsets sparsetables plotqa_cot wiki_en \
  --num-samples 64 \
  --max-shards-per-subset 1

EXPERIMENT=examples/puzzletron/configs/families/qwen3_5/qwen3p5_4b/runs/ffn_width_10to20pct_kd_search.yaml
EXECUTION=examples/puzzletron/configs/orchestration/qwen3p5_4b/execution.ffn_width_10to20pct_kd_search.yaml
RUNNER=examples/puzzletron/configs/orchestration/qwen3p5_4b/runner.slurm.yaml

python examples/puzzletron/orchestrate.py \
  --experiment "$EXPERIMENT" \
  --runner "$RUNNER" \
  --execution "$EXECUTION" \
  --stage full --dry-run
```

Run the dry run first and check the commands, paths, and requested GPUs. The
runner file is only a template until its placeholders are replaced. The full
recipe also contains a fresh 256-step KD run and a final teacher comparison;
run `2026-09-01-r2` stopped before those steps.

See the [Qwen 3.5 4B VLM example](../../../../docs/qwen3p5_4b_vlm_example.md)
for environment preparation and lifecycle details.
