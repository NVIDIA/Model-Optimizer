# Maintained recipes

Puzzletron provides five maintained recipes. Every recipe uses the same
`puzzletron.py` commands and a reusable site file. Start with a smoke for the
model and modality before running a campaign.

| Recipe | Purpose | Site resource |
|---|---|---|
| `qwen3p5_0p8b_text_smoke.yaml` | Text pruning lifecycle with bounded evaluation, serving, and distillation | `single-gpu` |
| `qwen3p5_0p8b_vlm_smoke.yaml` | Image-text pruning lifecycle with bounded evaluation, serving, and distillation | `smoke` |
| `qwen3p5_0p8b_vlm_campaign.yaml` | Larger multi-axis image-text integration example | `campaign` |
| `qwen3p5_4b_vlm_smoke.yaml` | 4B all-axis lifecycle with bounded evaluation, serving, and distillation | `smoke` |
| `qwen3p5_4b_vlm_campaign.yaml` | 4B exact-four all-axis search with matched KD128 | `campaign` |

Smoke workloads validate that the configured lifecycle runs, produces usable
checkpoints, and resumes. Their scores and throughput are not model-quality or
production-performance results. The campaigns are scheduled integration
examples, not recommended pruning or training policies.

## Run a recipe

Follow the [quickstart](../README.md#quickstart) for the common site, validation,
launch, and resume commands. Set a new worker-visible `run_root` for every run
and fill in `data.path` and `data.revision` when the recipe contains them.

## Route-specific requirements

### Qwen 3.5 0.8B text smoke

Provide a prepared Puzzle-KD dataset and immutable source revision in the
recipe. The materializer writes the required Hugging Face dataset with a
`messages` column:

```bash
python examples/puzzletron/materialize_dataset.py puzzle_kd_v2 \
  --output /shared/data/puzzle-kd-v2 \
  --train-samples 8 \
  --validation-samples 2 \
  --revision REPLACE_WITH_IMMUTABLE_DATASET_REVISION
```

Set `data.path` to that output directory. Workers must have the pinned text
evaluator available. Each scheduled GPU attempt uses one GPU. The smoke checks
FFN-width pruning, checkpoint materialization and reload, two-sample IFEval, a
small serving measurement, two distillation steps, final evaluation, reporting,
and no-work resume.

### Qwen 3.5 0.8B VLM smoke and campaign

Workers need the pinned model, Nemotron-VLM data, evaluator datasets and media,
and a shared Hugging Face cache. The smoke prepares eight image conversations
and fixed evaluation rows, then checks pruning, MIP selection, checkpoint
materialization and reload, image-text evaluation, serving, two distillation
steps, reporting, and resume in one reusable two-GPU allocation.

The campaign uses an eight-GPU allocation and increases the data, candidate,
evaluation, and distillation budgets. It explores hidden width, heterogeneous
FFN width, depth, grouped attention, and GDN geometry. Run it only after the
smoke succeeds. Its selection rule and 128-step distillation budget are example
settings.

For offline workers, prepare the caches described in [VLM checkpoint
evaluation](vlm_checkpoint_evaluation.md#cache-benchmark-data) before launch.

### Qwen 3.5 4B VLM smoke and campaign

Both 4B recipes require a prepared normalized VLM dataset and immutable
revision. Prepare eight samples for the smoke:

```bash
python examples/puzzletron/materialize_dataset.py nemotron_vlm_v2 \
  --output /shared/data/qwen3p5-vlm \
  --revision 51f4f4d219315c3283950994d4eb3d7fc30aa87b \
  --subsets sparsetables plotqa_cot wiki_en \
  --num-samples 8 \
  --max-shards-per-subset 1
```

Set `data.path` to `/shared/data/qwen3p5-vlm`. The campaign needs at least 512
samples, so prepare a separate directory with the same command and
`--num-samples 512`, then point the campaign recipe to it.

The smoke exercises hidden width, FFN width, grouped attention, GDN geometry,
and depth before checking materialization, checkpoint reload, a frozen 24-row
image-text evaluation, serving, two TP2 distillation steps, structured results,
reporting, and resume on two colocated GPUs.

The campaign generates candidates from parameter and serving-memory budgets
over every supported structural axis. It ranks complete finite pre-KD
image-text loss, fails before fanout unless exactly four distinct architectures
remain, and gives all four the same frozen pre/post evaluation and resumable
128-step, 512-example TP2 distillation exposure. Final quality ranks dominate
the serving tie-breaker. These search, ranking, and training budgets are example
settings. Compare candidates only when their model, data, evaluator, sampling,
generation, teacher, and runtime identities match.

## Inspect results

Use `puzzletron.py inspect <run-root>` and the generated campaign report. For a
smoke, confirm that every planned stage completed, materialized checkpoints
reload, reported metrics are finite, and resume submits no completed work.
Use [run and recovery](orchestration_operations.md) for failure handling and
[evaluate saved checkpoints](post_mip_pipeline.md#evaluate-saved-checkpoints)
for evaluator details.
