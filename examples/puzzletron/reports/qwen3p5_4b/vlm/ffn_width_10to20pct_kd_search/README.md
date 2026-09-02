# Qwen 3.5 4B VLM FFN-width 10%-to-20% KD search

This campaign evaluates language-FFN width pruning for `Qwen/Qwen3.5-4B`
while retaining the multimodal input path. It compares approximately 10%, 15%,
and 20% parameter reductions, materializes each student, measures serving
behavior, runs 64 steps of tensor-parallel knowledge distillation, and compares
the students with the teacher on a bounded two-task evaluation.

The campaign records integration and screening evidence. A run is not a model
quality result unless its leaf record identifies the exact evaluator, pinned
datasets, sampling policy, checkpoint geometry, and completed stages.

## Runs

| Run | Evidence status | Completed scope | Provisional outcome |
|---|---|---|---|
| [2026-09-01-r2](runs/2026-09-01-r2/summary.md) | `preliminary` | Three materialized students, serving, 64-step TP2 KD, post-KD loss, and RealWorldQA/MMMU prefix-100 evaluation repeated twice | The 9.5%-pruned student is the provisional candidate under this screen |

`preliminary` is deliberate because the downstream screen covers only the first
100 RealWorldQA and MMMU rows and lacks a frozen per-item manifest. The
configured 256-step continuation was not run, but it is an exploratory budget,
not a requirement for campaign or scientific completeness. The bounded scores
select a follow-up candidate; they do not establish general VLM quality
preservation.

## Reproduce

Use the pinned model and dataset revisions recorded in the run leaf. Prepare a
worker-visible dataset and replace every placeholder in the checked-in runner
template before launching.

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

Inspect the compiled topology, paths, environment, scheduler resources, and
all nested requests before removing `--dry-run`. The public runner is a
template and is not runnable until its placeholders are replaced. Launching
the complete checked-in campaign continues past the scope retained in the
current run leaf and includes the exploratory finalist continuation.

See the [Qwen 3.5 4B VLM example](../../../../docs/qwen3p5_4b_vlm_example.md)
for environment preparation and lifecycle details.
