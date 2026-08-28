# Qwen 3.5 0.8B VLM pruning smoke

Use this guide to run the checked-in Qwen 3.5 0.8B vision-language pruning
example on one GPU. The example uses eight image-conversation samples and keeps
the work small enough to check the complete workflow:

1. Search two FFN intermediate sizes, `3072` and `2048`.
2. Save the two candidates with the lowest image-text loss.
3. Reload each saved checkpoint and evaluate two RealWorldQA samples.
4. Measure both checkpoints with 1-, 6-, and 12-image AIPerf requests.
5. Distill the faster candidate for two steps, then reload and evaluate the
   resulting checkpoint.

The model, dataset, evaluation task, and small work limits are pinned in the
checked-in configuration. This smoke test checks that pruning, evaluation,
serving, distillation, and resume all work. Its scores and throughput are not
model-quality or production performance results.

## Before you start

Prepare the setup and worker environments described in
[environment setup](environment_setup.md). The worker environment needs
ModelOpt, NeMo AutoModel's Qwen 3.5 VLM support, the Puzzletron requirements,
and the AIPerf/vLLM runtime selected by your runner.

Workers also need access to:

- `Qwen/Qwen3.5-0.8B` at the revision in
  `configs/families/qwen3_5/qwen3p5_0p8b/model.yaml`;
- the pinned RealWorldQA snapshot described in
  [cache benchmark data](vlm_checkpoint_evaluation.md#cache-benchmark-data);
- a shared location for the prepared dataset and campaign output.

If workers cannot access the network, populate the model and benchmark caches
before launch and mount them through the runner. Keep the checked-in model
identity instead of replacing it with a machine-specific path.

## Prepare the dataset

Choose paths visible to every worker. The dataset revision must be an immutable
Hugging Face commit SHA.

```bash
export PUZZLETRON_DATASET_PATH=/path/to/qwen3p5-vlm-smoke-data
export PUZZLETRON_DATASET_REVISION=51f4f4d219315c3283950994d4eb3d7fc30aa87b
export PUZZLETRON_RUN_ROOT=/path/to/qwen3p5_0p8b_vlm_smoke
```

Download and normalize eight image-conversation samples. This is the only step
that needs dataset network access.

```bash
python examples/puzzletron/materialize_dataset.py nemotron_vlm_v2 \
  --output "$PUZZLETRON_DATASET_PATH" \
  --revision "$PUZZLETRON_DATASET_REVISION" \
  --subsets sparsetables plotqa_cot wiki_en \
  --num-samples 8 \
  --max-shards-per-subset 1
```

Check that the manifest has the expected revision, samples, and local images:

```bash
python - "$PUZZLETRON_DATASET_PATH" "$PUZZLETRON_DATASET_REVISION" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
manifest = json.loads((root / "manifest.json").read_text())
assert manifest["acquisition"]["revision"] == sys.argv[2]
assert manifest["sample_count"] == 8
assert manifest["image_count"] >= 8
assert all((root / image["path"]).is_file() for image in manifest["images"])
print(f"prepared {manifest['sample_count']} samples with {manifest['image_count']} images")
PY
```

## Configure the runner

The example provides the experiment and execution settings. Copy the runner
template into the output directory and replace its site-specific placeholders:

```bash
EXPERIMENT=examples/puzzletron/configs/families/qwen3_5/qwen3p5_0p8b/runs/full_vlm_smoke.yaml
EXECUTION=examples/puzzletron/configs/orchestration/qwen3p5_0p8b/execution.full_vlm_smoke.yaml
RUNNER_TEMPLATE=examples/puzzletron/configs/orchestration/qwen3p5_0p8b/runner.slurm.yaml
RUNNER="$PUZZLETRON_RUN_ROOT/runner.slurm.yaml"

mkdir -p "$PUZZLETRON_RUN_ROOT"
if test -e "$RUNNER"; then
  echo "runner already exists: $RUNNER" >&2
  exit 1
fi
cp "$RUNNER_TEMPLATE" "$RUNNER"
${EDITOR:-vi} "$RUNNER"

if rg -n 'REPLACE_WITH_' "$RUNNER"; then
  echo "replace every runner placeholder before continuing" >&2
  exit 1
fi
```

Set the repository, environment, container, mounts, Slurm account, and
partition for your site. The model cache, benchmark cache, prepared dataset,
and output directory must be visible inside the worker environment.

## Inspect and run the smoke test

Compile the plan without submitting work:

```bash
python examples/puzzletron/orchestrate.py \
  --experiment "$EXPERIMENT" \
  --runner "$RUNNER" \
  --execution "$EXECUTION" \
  --stage full --dry-run
```

Before launch, check that model stages use one GPU and that the plan contains
these VLM-specific steps:

- image-text evaluation and no text tokenization stage;
- `checkpoint_eval` before serving;
- chat serving with 1, 6, and 12 images per request;
- `post_kd_checkpoint_eval` after VLM distillation;
- final image-text evaluation and candidate selection.

Run the same plan without `--dry-run`:

```bash
python examples/puzzletron/orchestrate.py \
  --experiment "$EXPERIMENT" \
  --runner "$RUNNER" \
  --execution "$EXECUTION" \
  --stage full
```

To resume after an interruption, rerun this exact command with the same
environment variables and configuration files. Puzzletron reuses compatible
completed stages and reruns failed or incomplete work.

## Check the result

Open the campaign report at
`$PUZZLETRON_RUN_ROOT/artifacts/campaign_report/campaign_report.html`. Confirm
the following before accepting the smoke test:

- width scoring, image-text evaluation, and VLM distillation processed real
  image tensors;
- sorting and physical slicing checks passed;
- both saved pre-distillation checkpoints completed `checkpoint_eval` on two
  RealWorldQA samples with finite metrics;
- AIPerf completed its 1-, 6-, and 12-image requests for both candidates, and
  `fastest_vlm` selected one candidate from the 12-image throughput result;
- the selected post-distillation checkpoint completed
  `post_kd_checkpoint_eval` on two RealWorldQA samples with finite metrics;
- the two distillation steps produced finite CE and KD metrics;
- rerunning the launch command submitted no work for already completed stages.

The evaluation summaries contain aggregate metrics. The raw `image_eval` and
`final_image_eval` records should also report a nonzero vision-forward count
and image-output checksums:

```bash
python - "$PUZZLETRON_RUN_ROOT" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
for node in ("image_eval", "final_image_eval"):
    records = list(
        root.glob(
            f"artifacts/post_mip/nodes/{node}/executions/*/raw/*/solution_0.json"
        )
    )
    assert records, f"missing raw evaluation record for {node}"
    for path in records:
        details = json.loads(path.read_text())["observability"]
        assert details["vision_forward_count"] > 0
        assert details["vision_output_checksums"]
PY
```

The serving step uses one synthetic request for each image count. It checks the
multimodal API and provides a value for candidate selection, but it is not a
performance benchmark. Use more requests, realistic concurrency, and your
deployment image sizes before drawing throughput conclusions.

## Evaluate a saved checkpoint separately

The campaign evaluates saved checkpoints automatically and records the results
with each candidate. To check a checkpoint outside the campaign, validate the
environment and cached benchmark data first:

```bash
python -m examples.puzzletron.evaluation.vlm.run \
  --checkpoint /path/to/pruned-checkpoint \
  --output-dir /path/to/results/vlm-smoke \
  --hf-home /path/to/huggingface-cache \
  --suite short \
  --preflight-only
```

If preflight succeeds, run the same command without `--preflight-only`. The
`short` suite runs the pinned RealWorldQA and MMMU tasks. See
[VLM checkpoint evaluation](vlm_checkpoint_evaluation.md) for other suites,
result files, and cache preparation.

## Plan a larger run

Start from `full_vlm_smoke.yaml` and `execution.full_vlm_smoke.yaml`. Copy both
files, keep the VLM-specific order of evaluation, materialization, image
serving, selection, and distillation, then change each small smoke limit for
your use case.

Use guided setup when you need help resolving the model, dataset, and site
settings:

```bash
python examples/puzzletron/puzzletron_setup_v2.py \
  --defaults examples/puzzletron/configs/setup/defaults.example.yaml
```

Select Qwen 3.5 0.8B and the Nemotron-VLM v2 image-text dataset. The wizard's
pruning profile and post-MIP graph are generic, so do not treat the generated
graph as the maintained VLM example. Apply its site settings to your copied VLM
configuration instead.

Choose larger-run settings deliberately:

- representative dataset subsets and sample counts;
- supported FFN widths, search constraints, and number of solutions;
- evaluation tasks and sample counts for the intended use case;
- serving image sizes, images per request, concurrency, and request count;
- distillation steps, batch sizes, validation, and checkpoint frequency;
- worker resources for every changed or added stage.

Keep this route FFN-only unless another pruning axis has its own correctness
evidence. Inspect the complete plan with `--dry-run` before launch. See
[configuration and overrides](configuration_overrides.md) for persistent and
temporary changes.
