# Qwen 3.5 0.8B VLM pruning smoke

Use this guide to run the Qwen 3.5 0.8B vision-language pruning example on one
GPU. The example uses eight image-conversation samples and keeps
the work small enough to check the complete workflow:

1. Search two FFN intermediate sizes, `3072` and `2048`.
2. Save the two candidates with the lowest image-text loss.
3. Reload each saved checkpoint and evaluate two RealWorldQA samples.
4. Measure both checkpoints with 1-, 6-, and 12-image AIPerf requests.
5. Distill the faster candidate for two steps, then reload and evaluate the
   resulting checkpoint.

The example configuration pins the model, dataset, evaluation task, and small
work limits. This smoke test checks that pruning, evaluation, serving,
distillation, and resume all work. Its scores and throughput are not model
quality or performance results.

## Generate a complete bundle with the setup wizard

For a new run, start with the [setup wizard](setup_wizard.md), select Qwen 3.5
0.8B, and choose a multimodal dataset. The generated smoke and campaign bundles
cover conversion, multimodal pruning and search, MIP selection,
materialization, image-aware serving, VLM distillation, final selection, and a
pinned student-versus-teacher RealWorldQA/MMMU comparison. Inspect the generated
`dry-run-plan.txt` and materialize the site-specific runner settings before
launching.

The model-specific defaults are separate from the wizard implementation. The
same onboarding path can support the 2B and 4B variants by adding their model
inventory, pruning domains, resource defaults, and pinned evaluation settings.
The tracked recipes below remain useful for reproducing the bounded 0.8B run.

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
before launch and mount them through the runner. Keep the configured model
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
EXECUTION=examples/puzzletron/configs/orchestration/execution.single_gpu.yaml
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

## Choose a campaign or quality-comparison route

The routes use the pinned Qwen 3.5 0.8B VLM. The legacy
`full_vlm_smoke.yaml` compatibility route keeps the portable FFN-only lifecycle
check. `vlm_admitted_axes_lifecycle_smoke.yaml` enables
hidden width, FFN, and depth diagnostics, then realizes
one mixed candidate whose exact MIP parameter count must retain 85–95% of the
teacher. That candidate continues through physical materialization, checkpoint
reload, image serving, two-step VLM KD, and image-text evaluation.
The extended grid uses 64-channel hidden-width alignment so its `960` and `896`
endpoints pass the same physical materialization validator used by the smoke.

The older `qwen35_vlm_realworldqa`, `qwen35_vlm_e2e_full_eval`, and
`qwen35_vlm_short_v1` profile names remain registered only as deprecated
compatibility aliases. New recipes use identities that state the task scope and
row-selection policy.

`vlm_campaign.yaml` is the expanded admitted-axis search: hidden width,
heterogeneous FFN width, and depth. It reuses the existing `params-90` MIP
profile and the established image-text LM-loss `top_k: 2` screening step.
Serving measurements are recorded for both retained candidates but do not
affect selection. The `3328` and `3072` FFN widths are separate conservative
controls; the deeper `2816`, `2432`, `2048`, `1664`, and `1408` values remain
search bins rather than justified VLM defaults.

Every LM-loss-retained candidate follows one resumable trajectory from the same
immutable pre-KD materialized checkpoint. AutoModel optimizer state is
preserved while the cumulative step limit advances through 64, 128, and 256
updates. Candidate selection happens only after all three checkpoints have been
evaluated. The selection is an explicit manual gate so the campaign does not
encode an additional sampling interpretation.

The `3328` and `3072` controls follow their own otherwise identical 64/128/256
learning curves. Every milestone uses the same
`qwen35_vlm_realworldqa64_mmmu120_mvbench160_frozen_rows_v1` profile, exact-row
manifest path, and required manifest SHA256. The teacher result is
computed once under that identity and reused for every control, candidate, and
milestone. Each KD record includes global batch size, cumulative examples,
non-padding effective tokens from the training log, a padded-token upper bound,
estimated cumulative GPU-hours, and measured incremental and cumulative
GPU-hours. The selected bounded result freezes the pre-KD checkpoint identity,
all three checkpoint paths, metrics, exposure records, teacher identity, and
row-manifest digest in an immutable learning-curve manifest.

The 512- and 1024-step extensions are not part of the bounded campaign result.
They are available only for the selected finalist, each behind a separate
manual approval gate, and resume the same optimizer trajectory if approved.
They use the same fixed 344-row selection: 64 RealWorldQA rows, 120 MMMU
validation rows, and 160 MVBench rows under the semantic contract
`qwen35-vlm-rwqa64-mmmu120-mvbench160-frozen-v1`. The separate eight-task,
all-rows `qwen35-vlm-judge-free8-all-rows-v1` scope remains an optional later
run.

Grouped-attention and GDN reductions are disabled. The native Qwen
`Qwen3NextAttention` backend rejects compact reduced geometry, and its
CP-aware GDN wrapper has no compact scoring equivalence path. Keep their teacher
shapes until those backends have equivalent scoring and materialization support.
`gdn_value_heads_per_group` is additionally irreducible because its teacher
value is already one.

The opt-in `e2e_vlm_quality_comparison.yaml` route keeps the lifecycle bounded
and compares its final student with the pinned teacher on deterministic
RealWorldQA and MMMU subsets. Repetitions are resumable and are reused only
when checkpoint identity and evaluator artifacts still match. Results include
student and teacher metrics and their deltas, but no quality gate.

Use `e2e_vlm_quality_comparison_extended.yaml` for the same bounded comparison
after the expanded admitted-axis smoke candidate.

Run the comparison route with a site-specific runner and a distinct output
root. This route requires the pinned RealWorldQA and MMMU caches and is
intentionally excluded from default CI:

```bash
EXPERIMENT=examples/puzzletron/configs/families/qwen3_5/qwen3p5_0p8b/runs/e2e_vlm_quality_comparison.yaml
EXECUTION=examples/puzzletron/configs/orchestration/execution.single_gpu.yaml
RUNNER=/path/to/site-specific/runner.slurm.yaml
export PUZZLETRON_RUN_ROOT=/path/to/qwen3p5_0p8b_e2e_vlm_quality_comparison

python examples/puzzletron/orchestrate.py \
  --experiment "$EXPERIMENT" \
  --runner "$RUNNER" \
  --execution "$EXECUTION" \
  --stage full --dry-run
```

Inspect the compiled stage order and one-GPU allocation before launch. Then
omit `--dry-run` to launch or resume the exact same three-input campaign.

Run the campaign with the same execution profile and a distinct output root:

```bash
EXPERIMENT=examples/puzzletron/configs/families/qwen3_5/qwen3p5_0p8b/runs/vlm_campaign.yaml
EXECUTION=examples/puzzletron/configs/orchestration/qwen3p5_0p8b/execution.vlm_admitted_axes_campaign.yaml
RUNNER=/path/to/site-specific/runner.slurm.yaml
export PUZZLETRON_RUN_ROOT=/path/to/qwen3p5_0p8b_vlm_campaign
export PUZZLETRON_VLM_SHORT_V1_MANIFEST=/path/to/frozen-short-v1.json
export PUZZLETRON_VLM_SHORT_V1_SHA256=<64-lowercase-hex-digest>

python examples/puzzletron/orchestrate.py \
  --experiment "$EXPERIMENT" \
  --runner "$RUNNER" \
  --execution "$EXECUTION" \
  --stage full --dry-run
```

Inspect the compiled stage order, identity inputs, and one-GPU allocation. A
successful dry run or smoke test is preparation evidence only. Launching this
real campaign requires separate GPU/scheduler authorization, and the campaign
is not complete until at least one selected post-KD student has comparable
64/128/256 scores on the fixed 344-row selection and an immutable
learning-curve result manifest.

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

## Customize a campaign

Tune site resources and the pinned training dataset for the intended workload.

Use guided setup when you need help resolving the model, dataset, and site
settings:

```bash
python examples/puzzletron/puzzletron_setup_v2.py \
  --defaults examples/puzzletron/configs/setup/defaults.example.yaml
```

Select Qwen 3.5 0.8B and the Nemotron-VLM v2 image-text dataset. Guided setup
can generate the same route with site-specific settings. Hidden width,
attention, GDN, embedding width, and depth are available through guided
customization, but grouped-attention reduction is not admitted by the tracked
runtime-validated routes. Inspect every customized plan
with `--dry-run` before launch. See
[configuration and overrides](configuration_overrides.md) for persistent and
temporary changes.
