# VLM checkpoint evaluation

Use the VLM evaluator to run pinned image and video benchmarks against a local
Qwen 3.5 0.8B checkpoint without creating a Puzzletron campaign. The default
`short` suite is a deterministic smoke evaluation intended to catch checkpoint
or runtime regressions before a larger benchmark run.

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
is the supported older revision based on `lmms-eval` 0.7.0, not a ModelOpt fork
or a locally patched evaluator. It is installed as an editable checkout because
the evaluator loads task templates that are not included in its wheel.
Puzzletron generates its benchmark task adapters separately and does not modify
the evaluator checkout.

The documented [worker environment](environment_setup.md) installs the same
requirements file, and startup validates both the installed revision and that
the checkout has no local changes. Do not substitute another `lmms-eval`
revision in this environment. Support for a newer evaluator and its dependency
set is separate work. This repository does not define a second Puzzletron
container-specific evaluator pin.

## Choose a suite

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
datasets for the selected suite before starting evaluation. The example uses
the default `short` suite; replace `short` with another suite name when needed:

```bash
export HF_HOME=/path/to/huggingface-cache

python - "$HF_HOME" short <<'PY'
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

## Run the default smoke evaluation

```bash
python -m examples.puzzletron.evaluation.vlm.run \
  --checkpoint /path/to/checkpoint \
  --output-dir /path/to/results/vlm-smoke \
  --hf-home "$HF_HOME" \
  --suite short
```

The checkpoint must contain a readable `config.json` matching the supported
Qwen 3.5 0.8B VLM architecture. Use `--preflight-only` first to validate the
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
`summary.json` metrics. Repeating a run creates new attempt directories rather
than overwriting previous evidence.

Qwen 3.5 uses the pinned evaluator's generic vLLM backend. Image and video
scores are useful for repeatable development comparisons, but the generic
backend does not preserve video timestamps as faithfully as the newer native
Qwen 3.5 wrapper. Do not interpret its video scores as backend-equivalence
results.

If preflight fails, address the reported checkpoint, revision, cache, decoder,
or credential mismatch before retrying. Inspect `stderr.txt` in the attempt
directory when the evaluator subprocess starts but does not complete.
