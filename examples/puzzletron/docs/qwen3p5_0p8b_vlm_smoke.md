# Qwen 3.5 0.8B vision-language pruning smoke

The checked-in `full_vlm_smoke` recipe runs a small end-to-end test of
vision-language pruning for the public `Qwen/Qwen3.5-0.8B` checkpoint. It uses
real image-conversation examples to search the FFN intermediate sizes
`[3072, 2048]`, evaluate the candidates, and save the two strongest candidates
as physical checkpoints. It reloads each saved directory through vLLM for two
RealWorldQA image samples using the pinned Qwen 3.5 VLM evaluation profile. The
profile verifies the evaluator revision and immutable offline dataset snapshot,
strips inherited Hub credentials, and records preflight provenance before
delegating execution to the shared checkpoint evaluator. The workflow then
measures both checkpoints with 1-, 6-, and 12-image AIPerf requests, distills
the candidate with the highest measured 12-image throughput for two steps, and
runs the pinned RealWorldQA benchmark and internal image-and-text evaluation on
the resulting checkpoint. The immutable revision in `model.yaml` ensures that
repeated runs use the same starting model. See
[evaluate saved checkpoints](post_mip_pipeline.md#evaluate-saved-checkpoints)
for the direct pre-KD and post-KD reload paths; no AnyModel-to-AutoModel
conversion occurs.

These small budgets check that the complete workflow runs and resumes
correctly. They do not establish model quality or production throughput.

## Prepare the worker environment

Prepare the setup and worker environments described in
[environment setup](environment_setup.md). The worker environment must provide
ModelOpt, NeMo AutoModel's Qwen 3.5 VLM support, the Puzzletron requirements,
the pinned `lmms-eval` dependency, and the reviewed AIPerf/vLLM runtime selected
by your runner contract. The worker-visible Hugging Face cache must already
contain the pinned RealWorldQA snapshot. Populate it as described in
[cache benchmark data](vlm_checkpoint_evaluation.md#cache-benchmark-data); the
profile verifies the local snapshot and evaluates it offline.

The worker-visible Hugging Face cache must contain, or be allowed to fetch,
`Qwen/Qwen3.5-0.8B` at the pinned revision in
`configs/families/qwen3_5/qwen3p5_0p8b/model.yaml`. If workers are offline,
populate that cache before launch and mount it through the runner; do not
replace the checked-in model identity with a machine-specific path.

## Materialize the image-text smoke dataset

Choose worker-visible dataset and output paths. The dataset revision must be an
immutable Hugging Face commit SHA, not `main` or `latest`.

```bash
export PUZZLETRON_DATASET_PATH=/path/to/qwen3p5-vlm-smoke-data
export PUZZLETRON_DATASET_REVISION=51f4f4d219315c3283950994d4eb3d7fc30aa87b
export PUZZLETRON_RUN_ROOT=/path/to/qwen3p5_0p8b_full_vlm_smoke

test -n "$PUZZLETRON_DATASET_PATH"
test -n "$PUZZLETRON_DATASET_REVISION"
test -n "$PUZZLETRON_RUN_ROOT"
case "$PUZZLETRON_DATASET_REVISION" in
  *[!0-9a-f]*|'')
    echo "PUZZLETRON_DATASET_REVISION must be a lowercase commit SHA" >&2
    exit 1
    ;;
esac
test "${#PUZZLETRON_DATASET_REVISION}" -eq 40
```

Materialize eight normalized image-conversation samples. Run this networked
preparation step once; campaign workers consume the resulting local files and
manifest without fetching the dataset.

```bash
python examples/puzzletron/materialize_dataset.py nemotron_vlm_v2 \
  --output "$PUZZLETRON_DATASET_PATH" \
  --revision "$PUZZLETRON_DATASET_REVISION" \
  --subsets sparsetables plotqa_cot wiki_en \
  --num-samples 8 \
  --max-shards-per-subset 1
```

Before GPU work, confirm that the immutable acquisition identity, eight
samples, and real image inventory were published:

```bash
python - "$PUZZLETRON_DATASET_PATH" "$PUZZLETRON_DATASET_REVISION" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
expected_revision = sys.argv[2]
manifest = json.loads((root / "manifest.json").read_text())
acquisition = manifest["acquisition"]
assert acquisition["revision"] == expected_revision
assert manifest["sample_count"] == 8
assert manifest["image_count"] >= 8
assert len(manifest["images"]) == manifest["image_count"]
assert all((root / item["path"]).is_file() for item in manifest["images"])
print(json.dumps(manifest, indent=2, sort_keys=True))
PY
```

## Materialize the site runner

The experiment and execution files are canonical inputs. The checked-in Slurm
runner is a portable placeholder: copy it into the campaign directory, fill in
the site contract once, and reuse that same file for dry-run, launch, and
resume.

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

The runner's repository, environment, container, mounts, Slurm account, and
partition must all resolve from every worker. Include the Hugging Face model
cache and the materialized dataset path in the worker/container mount contract.

## Inspect, launch, and resume

Compile and inspect the full plan without submitting work:

```bash
python examples/puzzletron/orchestrate.py \
  --experiment "$EXPERIMENT" \
  --runner "$RUNNER" \
  --execution "$EXECUTION" \
  --stage full --dry-run
```

Confirm that all enabled model stages use one GPU, image-backed stages resolve
`data.modality=multimodal`, and no text tokenization stage is present. Confirm
that two quality candidates reach `post.params-90.checkpoint_eval` for the
bounded RealWorldQA run and then `post.params-90.vlm_serving`, which declares a
`chat` workload with 1, 6, and 12 1280x720 images per request rather than a
text-only serving proxy. The `fastest_vlm` filter selects one candidate from the
12-image throughput metric before KD. Then launch by omitting only `--dry-run`:

```bash
python examples/puzzletron/orchestrate.py \
  --experiment "$EXPERIMENT" \
  --runner "$RUNNER" \
  --execution "$EXECUTION" \
  --stage full
```

After an interruption, rerun that exact launch command with the same
environment variables and the same three input files. Puzzletron reuses
compatible completed stages and reruns failed or incomplete stages; do not
copy partial artifacts into a new output root.

## Acceptance evidence and limits

A successful command exit is necessary but not sufficient. Before treating the
smoke as accepted, verify the cumulative report at
`$PUZZLETRON_RUN_ROOT/artifacts/campaign_report/campaign_report.html` and its
canonical stage summaries:

- width scoring and VLM KD processed real image tensors and report a nonzero
  vision-forward count;
- sorting and physical slicing equivalence passed at the configured tolerance;
- each materialized pre-KD checkpoint has a successful `checkpoint_eval`
  summary whose `checkpoint` field names that saved artifact, whose
  RealWorldQA sample count equals two, and whose metrics are finite; the
  evaluation root's `profile.json` records the pinned dataset revision and
  offline preflight;
- the selected post-KD checkpoint has a successful `post_kd_checkpoint_eval`
  summary for two RealWorldQA samples with finite metrics, and the internal
  final image evaluation reloads the same checkpoint;
- the two-step KD summary contains finite main CE/KD and MTP CE/KD metrics plus
  nonzero trainable-group gradient evidence;
- AIPerf completes every 1-, 6-, and 12-image chat workload for both retained
  candidates without request failures, and `fastest_vlm` selects one candidate
  using `images_12.concurrency_1.image_throughput`;
- rerunning the launch command submits no work for completed compatible stages.

Post-MIP evaluation summaries contain aggregate metrics. Verify their raw
image-evaluation records separately:

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
        observability = json.loads(path.read_text())["observability"]
        assert observability["vision_forward_count"] > 0
        assert observability["vision_output_checksums"]
PY
```

The serving stage uses one synthetic request per candidate and workload, with
1280x720 images in batches of 1, 6, and 12. This exercises the multimodal API,
vision path, and comparative selection while keeping the example bounded; it
does not isolate vision-encoder latency or establish production performance.
Increase request count and concurrency in a separate reviewed performance run
before making throughput claims.

The real-checkpoint lifecycle test is opt-in and is not part of default pytest
or routine CI smoke execution. Point it at the populated cache:

```bash
PUZZLETRON_VLM_BENCHMARK_HF_HOME=/path/to/hf-home \
  pytest --run-manual tests/gpu/torch/puzzletron/test_qwen3p5_0p8b_vlm_smoke.py
```

The checkpoint contract is documented in
[evaluate saved checkpoints](post_mip_pipeline.md#evaluate-saved-checkpoints).
