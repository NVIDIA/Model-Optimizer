# VLM checkpoint evaluation

Use the VLM evaluator to run pinned image and video benchmarks against a local
Qwen 3.5 checkpoint without creating a Puzzletron campaign. Versioned profiles
are the reproducible interface for teacher and student comparisons. The default
`short` suite remains a small smoke evaluation for checkpoint and runtime
checks.

For text-only IFEval and GSM8K evaluation, use the separate
[text checkpoint evaluator](checkpoint_evaluation.md).

## Prepare the worker environment

Run evaluation in the same worker environment used by Puzzletron. Install
ModelOpt and the Puzzletron requirements from the repository root:

```bash
python -m pip install -e '.[hf,puzzletron]' \
  -r examples/puzzletron/requirements.txt
```

The requirements install the upstream EvolvingLMMs `lmms-eval` repository at
the exact revision recorded in `examples/puzzletron/ci_environment.json`. This
is the generic vLLM profile revision based on `lmms-eval` 0.7.0, not a ModelOpt
fork or a locally patched evaluator. It is installed as an editable checkout
because the evaluator loads task templates that are not included in its wheel.
Puzzletron generates its benchmark task adapters separately and does not modify
the evaluator checkout.

The documented [worker environment](environment_setup.md) installs the same
requirements file, and startup validates both the installed revision and that
the checkout has no local changes. To use a native profile, install
`examples/puzzletron/requirements-vlm-native.txt` after the base requirements.
That file selects the separately pinned upstream revision containing the native
Qwen 3.5 backend. Preflight rejects an evaluator revision that does not match
the selected profile.

## Choose a suite

Use a versioned profile when scores will be compared across checkpoints:

| Profile | Coverage | Sampling policy |
| --- | --- | --- |
| `short-v1` | RealWorldQA, MMMU, and MVBench | 344 exact rows; MMMU uses four rows per subject and MVBench uses eight per task |
| `short-native-v1` | Same rows as `short-v1`, using the native Qwen 3.5 backend | 344 exact rows; native video processing preserves Qwen timestamp metadata |
| `short-all-native-v1` | All eight judge-free benchmarks, using the native Qwen 3.5 backend | 690 exact rows; retains the 344-row selection and adds 346 rows across the five additional benchmarks |
| `full-v1` | Eight judge-free image and video benchmarks | Every row from each pinned dataset revision |

For future short teacher and student comparisons through the Qwen-specific
adapter, prefer `short-all-native-v1` when all eight datasets are available.
It checks more kinds of image and video tasks than the 344-row profiles. Use
the 344-row profiles for faster three-benchmark regression checks or when the
same rows must be compared across both execution paths. Neither short profile
replaces `full-v1` for complete benchmark reporting.

All profiles pin the evaluator revision, model family, dataset revisions,
task scoring configurations, preprocessing, generation, and sample selection.
They also pin batch size 1 because changing VLM batching can change deterministic
outputs with the maintained backends. Generic vLLM profiles disable Qwen
thinking through a local, evaluation-owned copy of the checkpoint chat
template; the native profile passes the equivalent backend option directly.
This ensures short-answer tasks spend their token budget on the scored answer
instead of hidden reasoning. Preflight reports the profile schema and SHA-256
contract fingerprint. The fingerprint is also part of the resumable run
identity, so results from a different contract are not reused.

`short-all-native-v1` balances VideoMMMU across its three task leaves,
Video-MME across duration and domain, MLVU across task type, and PerceptionTest
across area and reasoning type. LongVideoBench uses deterministic index-spaced
rows. This profile is a reproducible regression screen, not a full-benchmark
quality estimate.

`full-v1` excludes MMVU because complete MMVU evaluation requires an external
judge. Run MMVU separately through `mmvu-smoke` or the judge-enabled legacy
`full` suite.

The unversioned suites are useful for development and targeted diagnosis:

| Suite | Coverage | Sampling policy |
| --- | --- | --- |
| `short` | RealWorldQA and MMMU | Eight samples per task, repeated twice |
| `quick` | RealWorldQA, MMMU, and MVBench | 344 exact rows from a required versioned manifest |
| `adapter-smoke` | Video-MME and PerceptionTest | Eight samples per task |
| `video-mmmu-smoke` | VideoMMMU | Eight samples |
| `mmvu-smoke` | MMVU | Eight judge-free rows |
| `longvideobench-smoke` | LongVideoBench | Eight samples |
| `mlvu-smoke` | MLVU | Eight samples |
| `full` | All nine pinned image and video benchmarks | Complete datasets; MMVU judge calls required |

All suites pin their dataset and upstream task revisions. Generation is
deterministic. Video suites sample at 2 frames per second with at most 32
frames.

## Cache benchmark data

Evaluation is offline and requires every selected dataset revision to already
exist under an explicit Hugging Face cache root. Populate the pinned image
datasets for the selected profile or suite before starting evaluation. The
repository stores only profile metadata and exact-row selectors; it does not
store benchmark records or media. The example uses `short-v1`:

```bash
export HF_HOME=/path/to/huggingface-cache

python - "$HF_HOME" short-v1 <<'PY'
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

from examples.puzzletron.evaluation.vlm import profile, suites

hf_home = Path(sys.argv[1])
for task in suites.source_tasks(sys.argv[2]):
    dataset = profile.VLM_BENCHMARK_DATASETS[task]
    if dataset.media_dir is not None:
        continue
    snapshot_download(
        repo_id=dataset.repository,
        repo_type="dataset",
        revision=dataset.revision,
        cache_dir=hf_home / "hub",
    )
PY
```

The video preparation command downloads its pinned snapshot and safely extracts
the media:

```bash
python -m examples.puzzletron.evaluation.vlm.preparation.benchmark_data \
  --hf-home "$HF_HOME" \
  --tasks mvbench
```

Pass a comma-separated list to `--tasks` when preparing more than one video
dataset. Use `--download-only` and `--extract-only` to split transfer and
extraction across jobs, or `--range-resume` for a resumable single-writer
download. Run the command with `--help` to list the supported video dataset
names.

Video suites also require an installed `decord`-compatible reader. The
Puzzletron requirements select the supported reader for the current platform
where one is available. Preflight stops before evaluation with an installation
error if the environment has no compatible reader.

## Run a versioned profile

The checkpoint path is an invocation input, not part of either profile
manifest. Use the same command with a teacher or any materialized Qwen 3.5
student checkpoint:

```bash
python -m examples.puzzletron.evaluation.vlm.run \
  --checkpoint /path/to/checkpoint \
  --output-dir /path/to/results/short-v1 \
  --hf-home "$HF_HOME" \
  --profile short-v1
```

Replace `short-v1` with `short-native-v1` after installing the native
requirements. Use `short-all-native-v1` after caching all eight judge-free
benchmarks, or `full-v1` after caching the same datasets for a complete
generic-vLLM evaluation. Use `--preflight-only` before consuming GPU time.

For scheduler-safe parallelism, run one contract task per job with either
`--profile full-v1 --profile-task TASK` or
`--profile short-all-native-v1 --profile-task TASK`. Every shard retains the
complete profile fingerprint while recording its selected source task in the
preflight and resumable identity. A complete result requires exactly one
successful shard for each of the eight tasks declared in the selected manifest.

Grouped tasks can use multiple batch-1 workers without changing inference
batching. Add `--profile-task-shard INDEX/COUNT` to `mvbench` or `video_mmmu`,
where `INDEX` is zero-based. For example, shard MVBench as `0/8` through `7/8`
on an eight-GPU node and combine all leaf-task metrics. Every leaf appears in
exactly one shard and each result records the selected leaves in its resumable
identity.

## Run the default smoke evaluation

```bash
python -m examples.puzzletron.evaluation.vlm.run \
  --checkpoint /path/to/checkpoint \
  --output-dir /path/to/results/vlm-smoke \
  --hf-home "$HF_HOME" \
  --suite short
```

The checkpoint must contain a readable `config.json` matching the supported
Qwen 3.5 VLM family, local multimodal processor assets, and a Qwen chat template
with the standard `enable_thinking` switch. Use
`--preflight-only` first to validate the
checkpoint, evaluator revision, task definitions, cached datasets, media, and
credentials without starting model evaluation.

The `quick` suite additionally requires `--quick-manifest` with its pinned
344-row selection. The `full` suite requires explicit
`--allow-judge-calls`, `--mmvu-judge-api-type`, and `--mmvu-judge-model`
options plus the corresponding OpenAI or Azure credentials. Other suites do
not accept judge options and use a loopback-only disabled judge configuration.

## Interpret results and limitations

Each execution creates an isolated `attempt_<id>/` directory containing the
command, standard output and error, the raw evaluator result, and normalized
`summary.json` metrics. The command result uses the
`modelopt.vlm-evaluation-result/v1` schema, while preflight uses
`modelopt.vlm-evaluation-preflight/v1`. Repeating a run creates new attempt
directories rather than overwriting previous evidence.

`short-v1` and `full-v1` use the generic vLLM backend. `short-native-v1` and
`short-all-native-v1` load the checkpoint through Transformers with the native
Qwen 3.5 wrapper. The native route preserves Qwen video timestamp metadata;
the generic route converts media to generic multimodal messages and does not
preserve that metadata. The routes also use different pinned evaluator
revisions and prompt construction, so their scores are backend-specific
baselines rather than an engine-only comparison.

If preflight fails, address the reported checkpoint, revision, cache, decoder,
or credential mismatch before retrying. Inspect `stderr.txt` in the attempt
directory when the evaluator subprocess starts but does not complete.
