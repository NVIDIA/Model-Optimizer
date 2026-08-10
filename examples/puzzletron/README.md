# Puzzletron v2

Puzzletron searches for smaller and faster variants of a pretrained model by
combining width and depth importance, physical slicing, measured serving costs,
mixed-integer search, evaluation, and optional knowledge distillation. Each
stage writes resumable artifacts and contributes to one self-contained HTML
campaign report.

## Table of Contents

- [Campaign reports](#campaign-reports)
- [Setup wizard](#setup-wizard)
- [Installation](#installation)
- [Run with an agent](#run-with-an-agent)
- [Configuration](#configuration)
- [Run a campaign](#run-a-campaign)
- [Campaign stages](#campaign-stages)
- [Reports](#reports)

## Campaign reports

The [campaign report catalog](docs/campaign_reports.md) records each report's
producer state, reproduction and support status, metadata origin, relationship
to current configuration files, and known limitations. An entry marked as not
reproduced is not a current model-support claim. Detailed run facts remain in
the retained reports.

Each report is a self-contained HTML file that embeds sanity-check outputs,
stage manifests, and evaluation results; it can be 100s of MB. Download it to
disk and open it locally rather than previewing it in a browser tab.

Each report's **Zero-shot Evaluation** section compares the pruned candidate
solutions directly against the full teacher model across multiple benchmarks.
Interpret those results together with the reproduction status and unresolved
correctness findings recorded in the campaign report catalog.

## Setup wizard

The schema-driven Puzzletron v2 setup wizard inspects a local checkpoint or
Hugging Face model configuration and generates self-contained smoke and
production experiment, runner, and execution bundles. Its setup environment
does not require PyTorch or model weights.

Install the setup dependencies, then start the guided flow with the current v2
defaults profile:

```bash
python -m pip install -r examples/puzzletron/requirements-setup.txt
python examples/puzzletron/puzzletron_setup_v2.py \
  --defaults examples/puzzletron/configs/setup/defaults.example.yaml
```

The example defaults use only repository-relative values. Copy the file and add
site-specific data, scheduler, and container settings before selecting it.
Defaults are loaded only when passed explicitly. Selection prompts have
a visible **← Back** action; text and numeric prompts accept `:back`. Every
accepted answer and the exact navigation frame are saved in
`answers_v2.yaml`, so an interrupted session can resume with:

```bash
python examples/puzzletron/puzzletron_setup_v2.py --resume /path/to/campaign
```

The wizard supports reusable named parallel profiles, canonical per-stage execution
strategies, independent instance counts, scheduling-compatible effective batches,
multiple named vLLM workload/topology measurements, independent MIP goals with
internal constraints/variants/matrices, and editable post-MIP flow DAGs. The
recommended flow begins with online evaluation and uses LM loss for filtering;
it does not include Initial Filter. PTQ and downstream evaluation are shown as
reserved but unavailable.

The final review writes `resolved_defaults.yaml`, `README.md`, and validated
`smoke/` and `production/` bundles transactionally. Both bundles are validated
and dry-run, but neither is submitted and production is not gated on smoke. The
wizard never launches the orchestrator.

Wizard discovery and schema generation do not establish model or axis support.
The [campaign report catalog](docs/campaign_reports.md) records retained
reports, their evidence status, and the requirements for a future support
claim.

## Installation

Puzzletron uses one Python environment for ModelOpt, the patched vLLM fork,
AutoModel, and official AIPerf. Install PyTorch first and build every CUDA
extension against that same installation; mixing PyTorch or CUDA builds can
cause import failures or incorrect GPU execution. AIPerf uses the official PyPI
package; no custom AIPerf fork is required.

### 1. Start from the CUDA development image

Use this image for local containers and cluster jobs:

```text
nvcr.io/nvidia/cuda:12.9.2-cudnn-devel-ubuntu24.04
```

For example:

```bash
export PUZZLETRON_WORKSPACE=/absolute/path/to/workspace
docker run --gpus all --ipc=host --rm -it \
  -v "${PUZZLETRON_WORKSPACE}:/workspace" \
  -w /workspace \
  nvcr.io/nvidia/cuda:12.9.2-cudnn-devel-ubuntu24.04 bash
```

Inside the container, install Python 3.12 and the build tools used by editable
packages and optional CUDA extensions:

```bash
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y \
  build-essential cmake git ninja-build \
  python3 python3-dev python3-pip python3-venv
```

### 2. Clone the tracked forks

Keep ModelOpt and the two Puzzletron forks as siblings:

The core compatibility pins used by the CPU CI lane are recorded once in the
machine-readable [CI environment](ci_environment.json). Nox reads that file
directly. The full GPU setup below uses the same core package versions and adds
the CUDA-specific builds, patched vLLM runtime, and AIPerf.

```bash
export MODEL_OPT_ROOT=/workspace/modelopt
export VLLM_ROOT=/workspace/vllm
export AUTOMODEL_ROOT=/workspace/Automodel
export PUZZLETRON_CI_ENVIRONMENT="${MODEL_OPT_ROOT}/examples/puzzletron/ci_environment.json"
export AUTOMODEL_REF="$(python3 -c \
  'import json, sys; print(json.load(open(sys.argv[1]))["nemo_automodel"]["commit"])' \
  "${PUZZLETRON_CI_ENVIRONMENT}")"

git clone --branch feature/add_anymodel_to_vllm --single-branch \
  https://github.com/Separius/vllm.git "${VLLM_ROOT}"
git clone --branch puzzletron --single-branch \
  https://github.com/Separius/Automodel.git "${AUTOMODEL_ROOT}"
git -C "${AUTOMODEL_ROOT}" checkout --detach "${AUTOMODEL_REF}"
```

```text
/workspace/
├── modelopt/
├── vllm/
└── Automodel/
```

### 3. Create the environment and install runtime packages

The patched vLLM branch uses the PyTorch version recorded in the CI environment
with CUDA 12.9. Install that combination before anything that compiles CUDA
code:

```bash
python3 -m venv /workspace/.venv
source /workspace/.venv/bin/activate

export PUZZLETRON_TORCH_VERSION="$(python -c \
  'import json, sys; print(json.load(open(sys.argv[1]))["torch"])' \
  "${PUZZLETRON_CI_ENVIRONMENT}")"
export PUZZLETRON_TORCHVISION_VERSION="$(python -c \
  'import json, sys; print(json.load(open(sys.argv[1]))["torchvision"])' \
  "${PUZZLETRON_CI_ENVIRONMENT}")"
export PUZZLETRON_TRANSFORMERS_VERSION="$(python -c \
  'import json, sys; print(json.load(open(sys.argv[1]))["transformers"])' \
  "${PUZZLETRON_CI_ENVIRONMENT}")"

python -m pip install --upgrade \
  pip "setuptools>=80,<81" "setuptools-scm>=8" setuptools-rust \
  wheel "packaging>=24.2" "cmake>=3.26.1" ninja jinja2

python -m pip install \
  "torch==${PUZZLETRON_TORCH_VERSION}" \
  "torchvision==${PUZZLETRON_TORCHVISION_VERSION}" \
  "torchaudio==${PUZZLETRON_TORCH_VERSION}" \
  --index-url https://download.pytorch.org/whl/cu129

VLLM_USE_PRECOMPILED=1 VLLM_PRECOMPILED_WHEEL_VARIANT=cu129 \
  python -m pip install --no-build-isolation -e "${VLLM_ROOT}"

python -m pip install -e "${AUTOMODEL_ROOT}"
python -m pip install aiperf
python -m pip install -e "${MODEL_OPT_ROOT}[hf,puzzletron]"
python -m pip install "transformers==${PUZZLETRON_TRANSFORMERS_VERSION}"
python -m pip install -r "${MODEL_OPT_ROOT}/examples/puzzletron/requirements.txt"
```

Do not add `--no-deps`: the packages need their declared Python dependencies.
`--no-build-isolation` ensures compiled extensions use the active PyTorch
installation; it does not disable dependency installation.

Install only the model-specific kernels required by the target architecture:

```bash
# Mixture of experts
python -m pip install --no-build-isolation \
  "git+https://github.com/fanshiqing/grouped_gemm@v1.1.4"

# Mamba
python -m pip install "mamba-ssm[causal-conv1d]" --no-build-isolation

# Linear attention
python -m pip install "flash-linear-attention[cuda]"
```

### 4. Verify the exact environment

Run these checks inside the same container and venv used for Puzzletron jobs:

```bash
test "$(git -C "${VLLM_ROOT}" remote get-url origin)" = \
  "https://github.com/Separius/vllm.git"
test "$(git -C "${VLLM_ROOT}" branch --show-current)" = \
  "feature/add_anymodel_to_vllm"
test "$(git -C "${AUTOMODEL_ROOT}" remote get-url origin)" = \
  "https://github.com/Separius/Automodel.git"
test "$(git -C "${AUTOMODEL_ROOT}" rev-parse HEAD)" = "${AUTOMODEL_REF}"

git -C "${MODEL_OPT_ROOT}" rev-parse HEAD
git -C "${VLLM_ROOT}" rev-parse HEAD
git -C "${AUTOMODEL_ROOT}" rev-parse HEAD
```

```bash
python - <<'PY'
import importlib.metadata as metadata
import json
import os

from packaging.version import Version

import aiperf
import modelopt
import nemo_automodel
import torch
import transformers
import vllm

with open(os.environ["PUZZLETRON_CI_ENVIRONMENT"], encoding="utf-8") as stream:
    ci_environment = json.load(stream)

for package in ("torch", "vllm", "nemo-automodel", "aiperf", "nvidia-modelopt"):
    print(package, metadata.version(package))

print("torch CUDA", torch.version.cuda)
print("CUDA available", torch.cuda.is_available())
print("modelopt", modelopt.__file__)
print("vllm", vllm.__file__)

assert Version(torch.__version__).release == Version(ci_environment["torch"]).release
assert Version(metadata.version("torchvision")).release == Version(
    ci_environment["torchvision"]
).release
assert transformers.__version__ == ci_environment["transformers"]
assert Version(metadata.version("nemo-automodel")).base_version == (
    ci_environment["nemo_automodel"]["base_version"]
)
assert torch.version.cuda == "12.9"
assert torch.cuda.is_available()
PY

python -m pip check
```

Record the three source revisions and verification output with the campaign.
Re-run verification after pulling either fork or rebuilding a CUDA extension.

## Run with an agent

The canonical agent workflow is
[`running-puzzletron`](../../.agents/skills/running-puzzletron/SKILL.md). Ask an
agent to use that skill and provide the model, dataset, compute environment,
search space, resource constraints, and required downstream stages. For
example:

```text
Use .agents/skills/running-puzzletron/SKILL.md to run the Puzzletron campaign
at examples/puzzletron/configs/families/nemotron3/nano_30b_a3b_bf16/runs/default.yaml.
Validate the smoke path first, execute the enabled DAG, resume compatible
artifacts, and regenerate and verify the report after every completed stage.
```

`.agents/` is the source of truth. Agent-specific paths such as
`.claude/skills/running-puzzletron` are compatibility symlinks and should not
be edited separately.

## Configuration

Configs use Hydra composition:

```text
examples/puzzletron/configs/
├── base.yaml                         # pipeline-wide defaults
└── families/
    └── <family>/
        ├── family.yaml               # descriptors, hooks, and family axes
        └── <model>/
            ├── model.yaml            # checkpoint metadata and legal domains
            └── runs/default.yaml     # current campaign replay/reconstruction
```

Site-specific paths can be overridden without editing the checked-in config:

```bash
export PUZZLETRON_RUN_ROOT=/shared/puzzle_runs/my_campaign
```

Independent runs, variants, solution pools, resource constraints, and
homogeneous search are documented in [MIP runs](docs/mip_profiles.md). Configure
candidate evaluation, filtering, materialization, and distillation with
[post-MIP pipelines](docs/post_mip_pipeline.md). The
[campaign report catalog](docs/campaign_reports.md) records retained reports and
their evidence status; it is not a current model or pruning-axis support
matrix.

## Run a campaign

The v2 orchestrator lives in
[`modelopt/torch/puzzletron/orchestration/`](../../modelopt/torch/puzzletron/orchestration/)
and is launched through
[`examples/puzzletron/orchestrate.py`](orchestrate.py). It separates:

- experiment semantics (`--experiment`, the existing Puzzletron YAML);
- runner infrastructure (`--runner`, Slurm or bare-metal inventory plus container/venv);
- execution semantics (`--execution`, per-stage strategy, `instances`, and optional mesh overrides).

The controller import path is independent of `modelopt.torch`; it requires only
Python 3.10+, PyYAML, and Rich. This lets it run on a Slurm login node while GPU jobs
use the full environment declared by the runner:

```bash
python3 -m venv .venv-orchestrator
source .venv-orchestrator/bin/activate
python -m pip install -r examples/puzzletron/requirements-orchestrator.txt
```

The login-node environment must expose `sbatch`, `squeue`, and `sacct`. It does
not need PyTorch, Hydra, ModelOpt installation, CUDA, or the worker container.

Each stage reads its AutoModel mesh from the experiment config
(`tp`, `cp`, `pp`, `ep`, `dp_shard`, `dp_replicate`). The orchestrator derives
`gpus_per_instance = PP × DP_REPLICATE × DP_SHARD × CP × TP` (EP overlays
`dp_shard`) and packs `instances` independent model copies onto nodes. For
example, sixteen one-GPU `vllm_stats` shards on an eight-GPU node type allocate
two nodes and dispatch sixteen shard commands.

Example single-stage dry-run:

```bash
PUZZLETRON_EXPERIMENT=examples/puzzletron/configs/families/nemotron3/nano_30b_a3b_bf16/runs/default.yaml
PUZZLETRON_RUNNER=examples/puzzletron/configs/orchestration/runner.slurm.example.yaml
PUZZLETRON_EXECUTION=examples/puzzletron/configs/orchestration/execution.example.yaml

python examples/puzzletron/orchestrate.py \
  --experiment "$PUZZLETRON_EXPERIMENT" \
  --runner "$PUZZLETRON_RUNNER" \
  --execution "$PUZZLETRON_EXECUTION" \
  --stage mip \
  --dry-run
```

Setup-v2-generated bundles encode evaluation, filtering, materialization,
AIPerf, and distillation in `post_mip.flows`. Run their generated experiment,
runner, and execution configs with `--stage full`; the orchestrator schedules
the dynamic `post.*` nodes as part of the campaign DAG.

The checked-in Nano experiment uses the legacy `zero_shot_evaluation`,
`aiperf`, and global distillation stages. Its online-solution path still needs
two explicit preparation steps because the orchestrator runs and aggregates
the shards but does not create the online evaluation plan or materialize the
selected finalists. Use site-specific runner and execution configs, then run
the prerequisite DAG through MIP by temporarily disabling the downstream
stages:

```bash
PUZZLETRON_EXPERIMENT=examples/puzzletron/configs/families/nemotron3/nano_30b_a3b_bf16/runs/default.yaml
PUZZLETRON_RUNNER=/path/to/runner.yaml
PUZZLETRON_EXECUTION=/path/to/execution.yaml
export PUZZLETRON_RUN_ROOT=/shared/puzzle_runs/my_campaign

python examples/puzzletron/orchestrate.py \
  --experiment "$PUZZLETRON_EXPERIMENT" \
  --runner "$PUZZLETRON_RUNNER" \
  --execution "$PUZZLETRON_EXECUTION" \
  --stage full \
  --override zero_shot_evaluation.enabled=false \
  --override aiperf.enabled=false \
  --override global_distillation_sanity.enabled=false \
  --override global_distillation.enabled=false \
  --override post_distillation_evaluation.enabled=false
```

Prepare the online evaluation plan, then run and aggregate its shards. The
profile IDs below match the Nano example; for another legacy experiment using
this path, pass every entry from its `zero_shot_evaluation.profile_ids` list.

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
  --stage zero_shot_evaluation
```

Materialize the evaluated finalists for the Nano example's configured AIPerf
profile before running AIPerf and the remaining enabled stages. This helper
loads ModelOpt and Safetensors, so run it in the full worker environment from
the installation steps above, not in the dependency-light controller
environment. If the runner uses a container, enter it with the same mounts
before activating the worker venv. For another legacy experiment using this
path, use its `aiperf.profile_id` value.

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

Return to the login node before launching the controller. The login node must
provide the scheduler commands listed above.

```bash
cd /path/to/modelopt
source .venv-orchestrator/bin/activate
PUZZLETRON_EXPERIMENT=examples/puzzletron/configs/families/nemotron3/nano_30b_a3b_bf16/runs/default.yaml
PUZZLETRON_RUNNER=/path/to/runner.yaml
PUZZLETRON_EXECUTION=/path/to/execution.yaml
export PUZZLETRON_RUN_ROOT=/shared/puzzle_runs/my_campaign

python examples/puzzletron/orchestrate.py \
  --experiment "$PUZZLETRON_EXPERIMENT" \
  --runner "$PUZZLETRON_RUNNER" \
  --execution "$PUZZLETRON_EXECUTION" \
  --stage full
```

For the legacy Nano experiment, the final `--stage full` resumes from the
verified completed stages. Do not use it as the initial command for legacy
configs with `mode: online_solutions`; setup-v2-generated bundles use their
dynamic `post.*` DAG instead.

The launch command is a blocking foreground controller: it submits every
dependency-ready branch concurrently, polls scheduler state, and exits when the
selected plan completes or fails. Progress is colorized automatically on a TTY.
Interactive terminals show a live stage table with status, nodes/tasks/GPUs,
elapsed time, and best-effort ETA when a stage exposes current/total progress.
Completed stages remain visible; dependency waits, failed stages, and descendants
blocked by failures are labeled explicitly. Redirected output falls back to
timestamped one-line progress updates.
Press `q` or Ctrl-C in an interactive terminal to choose between cancelling all
active jobs and quitting, leaving jobs running and detaching the controller, or
resuming the campaign. Non-interactive Ctrl-C and SIGTERM retain the safe
cancel-and-quit behavior. A detached controller preserves durable handles, so the
same command recovers the running jobs.
Use `--color always` when piping through `tee`, `--color never` for plain logs,
and `--poll-interval SECONDS` to change the default five-second scheduler poll.

Durable controller state is written under `${puzzle_dir}/orchestration/`. The
controller supports `single`, `sharded`, and `persistent_pool` strategies,
stdlib-first Slurm and SSH executors, attempt recovery, and semantic stage
validation through WorkAdapters. See
[`configs/orchestration/`](configs/orchestration/) for starter runner and
execution files.

## Campaign stages

Select one stage with the same v2 experiment, runner, and execution configs:

```bash
PUZZLETRON_BUNDLE=/path/to/generated/campaign/production

python examples/puzzletron/orchestrate.py \
  --experiment "$PUZZLETRON_BUNDLE/experiment.yaml" \
  --runner "$PUZZLETRON_BUNDLE/runner.yaml" \
  --execution "$PUZZLETRON_BUNDLE/execution.yaml" \
  --stage width_importance
```

The dependency-free [`StageSpec` registry](../../modelopt/torch/puzzletron/stages/graph.py) is the authoritative contract for every public stage's identity, dependencies, enablement, semantic config sections, and static completion artifacts. Add or change those properties there. Default execution strategies remain scheduler-specific and live in the [orchestration compiler](../../modelopt/torch/puzzletron/orchestration/compiler.py); handlers, scheduler adapters, mesh resolution, and heavyweight artifact validators remain separate runtime concerns.

| Stage | Purpose |
|---|---|
| `convert` | Convert the immutable Hugging Face teacher into the configured backend format. |
| `tokenize_data` | Build deterministic train and validation token caches. |
| `vllm_stats` | Measure exact runtime and memory costs for candidate subblocks. |
| `depth_importance` | Rank cumulative block or subblock removals. |
| `width_importance` | Collect activation-based rankings for every enabled width axis. |
| `sort` | Reorder the teacher so nested prefixes implement ranked width choices. |
| `sort_sanity` | Check that sorting preserves teacher outputs. |
| `width_sanity` | Compare ranked, original-order, and reverse slices on representative layers. |
| `slicing_sanity` | Verify dynamic slicing against physical materialization. |
| `bypass_sanity` | Overfit small local-distillation cases before production bypass. |
| `bypass` | Train nested replacement blocks across the configured search space. |
| `build_library` | Assemble sorted, bypassed, and no-op replacement candidates. |
| `replacement_scoring` | Score replacing one block or subblock at a time. |
| `mip` | Solve heterogeneous and homogeneous architecture searches under named constraints. |
| `zero_shot_evaluation` | Evaluate selected MIP recipes online without materializing every checkpoint. |
| `aiperf` | Materialize selected finalists and benchmark serving performance. |
| `global_distillation_sanity` | Overfit the selected global student as a correctness check. |
| `global_distillation` | Distill the selected architecture at the configured production scale. |
| `post_distillation_evaluation` | Evaluate the final distilled checkpoint. |

## Interpret width sanity results

Puzzletron separates implementation correctness from ranking quality:

- Sort and slicing equivalence failures are correctness errors and always fail
  their stages.
- A width-ranking miss means the activation-sorted candidate underperformed an
  original or reverse control. It is a quality warning by default.

To also fail a sanity stage on ranking-quality warnings, enable strict warning
handling:

```yaml
sanity:
  fail_on_warnings: true
```

See [Sorting, width ranking, and slicing sanity](docs/sanity_validation.md) for
the slicing mental model, measured metrics, comparison controls, tolerances,
worked example, and qualification guidance.

Independent DAG branches may run concurrently when they have disjoint writers.
Long-running stages should resume their durable checkpoints or immutable shards
rather than restarting completed work.

## Reports

After the selected plan completes cleanly, the v2 orchestrator generates the
final campaign report through the configured runner. Reporting is nonfatal to
the completed campaign, but a failed report attempt is recorded in the
controller result. The generator is read-only with respect to model artifacts
and includes valid partial results without marking their stage complete.

Regenerate without rerunning model work:

```bash
python examples/puzzletron/generate_campaign_progress_report.py \
  --puzzle-dir /shared/puzzle_runs/my_campaign \
  --model-name 'My model'
```

The output is
`<puzzle-dir>/artifacts/campaign_report/campaign_report.html`. It is a
self-contained file suitable for sharing. Section source and configuration
fingerprints are cached under
`<puzzle-dir>/artifacts/campaign_report/section_cache`, so unchanged sections
are reused and only affected sections rebuild. Use `--rebuild-section aiperf`
(repeatable) for selected sections, or `--no-cache` for an intentional full
rebuild.
