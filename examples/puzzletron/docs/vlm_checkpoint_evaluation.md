# VLM checkpoint evaluation

Use this evaluator to test a local Qwen 3.5 checkpoint on image and video
benchmarks. A profile chooses the benchmarks, examples, model-loading path,
and evaluation settings. Use the same profile for every model being compared.

The common choices are:

- `short-all-native-v1` for the recommended short comparison across eight
  benchmarks;
- `short-v1` or `short-native-v1` for a faster three-benchmark check; and
- `full-v1` to evaluate every available example in eight benchmarks.

For text-only IFEval and GSM8K evaluation, use the separate
[text checkpoint evaluator](checkpoint_evaluation.md).

## Prepare the worker environment

Run evaluation in the same worker environment used by Puzzletron. Install
ModelOpt and the Puzzletron requirements from the repository root:

```bash
python -m pip install -e '.[hf,puzzletron]' \
  -r examples/puzzletron/requirements.txt
```

This installs a pinned upstream `lmms-eval` revision that declares version
0.7.0. It replaces another version already present in the worker image.

Profiles with `native` in their name need one additional installation step:

```bash
python -m pip install -r examples/puzzletron/requirements-vlm-native.txt
```

This temporary requirements file installs upstream `lmms-eval` 0.7.1, which
adds the Qwen 3.5-specific model loader. Neither pinned version is a ModelOpt
fork. Follow-up work should use the `lmms-eval` version supplied by the worker
image once both model-loading paths have been validated with it.

Keep the general vLLM and Qwen-specific paths in separate environments because
they require different `lmms-eval` revisions. If one environment must be
reused, reinstall `examples/puzzletron/requirements.txt` before running
`short-v1` or `full-v1`, and reinstall
`examples/puzzletron/requirements-vlm-native.txt` before running a profile with
`native` in its name. Preflight rejects the wrong installed revision.

## Understand the two execution paths

| Path | Profiles | What it does |
| --- | --- | --- |
| Qwen-specific Transformers | `short-native-v1`, `short-all-native-v1` | Loads the checkpoint directly with the Qwen 3.5 model loader. Video prompts include timestamps for sampled frames. |
| General vLLM | `short-v1`, `full-v1` | Runs the checkpoint through vLLM and converts inputs to general image and video messages. Video prompts do not include frame timestamps. |

The paths also use different `lmms-eval` revisions and produce different
prompts. Their scores represent the complete paths and do not isolate the
effect of the inference engine.

## Choose an evaluation

Use a versioned profile when scores will be compared across checkpoints:

| Profile | Coverage | Examples evaluated |
| --- | --- | --- |
| `short-v1` | RealWorldQA, MMMU, and MVBench | 344 predefined examples; MMMU uses four per subject and MVBench uses eight per category |
| `short-native-v1` | Same examples as `short-v1`, using the Qwen-specific Transformers adapter | The same 344 predefined examples with timestamps added to video prompts |
| `short-all-native-v1` | All eight judge-free benchmarks, using the Qwen-specific Transformers adapter | 690 predefined examples: the same 344 plus 346 from five additional benchmarks |
| `full-v1` | Eight judge-free image and video benchmarks | Every available example in each pinned dataset version |

For future short teacher and student comparisons through the Qwen-specific
adapter, prefer `short-all-native-v1` when all eight datasets are available.
It checks more kinds of image and video tasks than the 344-row profiles. Use
the 344-row profiles for faster three-benchmark regression checks or when the
same rows must be compared across both execution paths. Neither short profile
replaces `full-v1` for complete benchmark reporting.

Profiles keep the evaluator, datasets, selected examples, video sampling,
answer generation, and batch size fixed. They disable Qwen thinking so the
model returns the short answer expected by these benchmarks. Puzzletron stops
before evaluation if the installed evaluator or cached data do not match the
selected profile.

`short-all-native-v1` balances VideoMMMU across its three categories,
Video-MME across duration and domain, MLVU across task type, and PerceptionTest
across area and reasoning type. LongVideoBench uses deterministic index-spaced
rows. This profile is a reproducible regression screen, not a full-benchmark
quality estimate.

`full-v1` excludes MMVU because complete MMVU evaluation requires an external
judge. Run MMVU separately through `mmvu-smoke` or the judge-enabled legacy
`full` suite.

These additional options are intended for setup checks and targeted diagnosis:

| Option | Coverage | Examples evaluated |
| --- | --- | --- |
| `short` | RealWorldQA and MMMU | Eight samples per task, repeated twice |
| `realworldqa-mmmu-prefix100-repeat2` | RealWorldQA and MMMU | First 100 rows of each task, repeated twice |
| `quick` | RealWorldQA, MMMU, and MVBench | 344 predefined examples from a required profile file |
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

## Check the setup before using a GPU

Add `--preflight-only` to any evaluation command to check the model, evaluator,
cached datasets, video files, and credentials without loading model weights or
running inference.

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

To run tasks in parallel, run one profile task per job with either
`--profile full-v1 --profile-task TASK` or
`--profile short-all-native-v1 --profile-task TASK`. A complete result needs
one successful job for each of the eight tasks in the selected profile.

Grouped tasks can use multiple batch-1 workers without changing inference
batching. Add `--profile-task-shard INDEX/COUNT` to `mvbench` or `video_mmmu`,
where `INDEX` is zero-based. For example, split MVBench as `0/8` through `7/8`
on an eight-GPU node and combine all leaf-task metrics.

## Run the default smoke evaluation

```bash
python -m examples.puzzletron.evaluation.vlm.run \
  --checkpoint /path/to/checkpoint \
  --output-dir /path/to/results/vlm-smoke \
  --hf-home "$HF_HOME" \
  --suite short
```

Run with `--preflight-only` first. It reports missing or incompatible model
files, evaluator packages, datasets, video files, and credentials without
starting model evaluation.

The `quick` suite additionally requires `--quick-manifest` with its pinned
344-row selection. The `full` suite requires explicit
`--allow-judge-calls`, `--mmvu-judge-api-type`, and `--mmvu-judge-model`
options plus the corresponding OpenAI or Azure credentials. Other suites do
not accept judge options and use a loopback-only disabled judge configuration.

## Interpret results and limitations

Each execution creates an `attempt_<id>/` directory containing the command,
logs, raw evaluator output, and normalized metrics in `summary.json`. Repeating
a run creates another attempt directory instead of overwriting earlier output.

`short-v1` and `full-v1` use the generic vLLM adapter. `short-native-v1` and
`short-all-native-v1` use the Qwen-specific Transformers adapter. As described
above, the paths use different evaluator revisions and construct different
model inputs. Keep the profile name with every score, and do not combine scores
from the two paths as if only the inference engine changed.

If preflight fails, address the reported checkpoint, revision, cache, decoder,
or credential mismatch before retrying. Inspect `stderr.txt` in the attempt
directory when the evaluator subprocess starts but does not complete.
