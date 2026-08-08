# Puzzletron

Puzzletron searches for smaller and faster variants of a pretrained model by
combining width and depth importance, physical slicing, measured serving costs,
mixed-integer search, evaluation, and optional knowledge distillation. Each
stage writes resumable artifacts and contributes to one self-contained HTML
campaign report.

## Table of Contents

- [End-to-end tested models](#end-to-end-tested-models)
- [Setup wizard](#setup-wizard)
- [Installation](#installation)
- [Run with an agent](#run-with-an-agent)
- [Configuration](#configuration)
- [Run the complete pipeline manually](#run-the-complete-pipeline-manually)
- [Run step by step](#run-step-by-step)
- [Online evaluation and downstream stages](#online-evaluation-and-downstream-stages)
- [Reports](#reports)

## End-to-end tested models

The configs below are the exact current-code entry points for the completed
campaigns. Each verified report is a self-contained HTML file that embeds all
sanity-check outputs, stage manifests, and evaluation results; they can be
100s of MB. Download them to disk and open locally rather than previewing in a
browser tab.

The most important section in every report is **Zero-shot Evaluation**, which
compares the pruned candidate solutions directly against the full teacher model
across multiple benchmarks. Use that section first to assess accuracy trade-offs
before inspecting the serving-performance or sanity-check sections.

| Model | Hugging Face model | Full experiment config | Verified report |
|---|---|---|---|
| Nemotron-3 Nano 30B-A3B | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` | [default.yaml](configs/families/nemotron3/nano_30b_a3b_bf16/runs/default.yaml) | [HTML report](reports/nemotron3_nano_30b_a3b.html) |
| Qwen3p5_9B | `Qwen/Qwen3.5-9B` | [default.yaml](configs/families/qwen3_5/qwen3p5_9b/runs/default.yaml) | [HTML report](reports/qwen3p5_9b.html) |

## Setup wizard

The setup wizard inspects a local checkpoint config or Hugging Face model config
and generates self-contained smoke and production experiment, runner, and
execution bundles. Its setup environment does not require PyTorch or model
weights:

```bash
python -m pip install -r examples/puzzletron/requirements-setup.txt
python examples/puzzletron/puzzletron_setup.py
```

Normal mode asks the model, data, pruning axes, MIP objectives/runs, and cluster
details while accepting defaults for lower-level tuning. Detailed mode also
exposes solver controls, extra constraints, custom post-MIP nodes, and resource
overrides:

```bash
python examples/puzzletron/puzzletron_setup.py --detailed
```

Every answer is saved atomically. Invocations without `--resume` start a campaign;
resuming an existing campaign requires its path:

```bash
python examples/puzzletron/puzzletron_setup.py --resume /path/to/campaign
```

The supported profiles cover Nemotron 3 and Qwen 3.5/3.6 dense, MoE, text, and
multimodal configurations. Unsupported configs exit with detected metadata and
point to `.agents/skills/running-puzzletron/SKILL.md` for descriptor onboarding.

Each campaign contains independent `smoke/` and `production/` bundles. Both are
validated and dry-run, but neither is submitted and production is not gated on
smoke. Slurm and SSH bare-metal runners are supported; use the bare-metal runner
with `localhost` for a single local host.

### Setup wizard v2

The setup wizard v2 offers three guided profiles:

- **Quick smoke** is the fastest way to verify that the campaign shape is valid.
- **Balanced pruning** is recommended for a first real campaign.
- **High-confidence search** spends more runtime on scoring and sanity checks.

The selected profile supplies nested pruning and MIP defaults from the detected
model family's `setup_v2_defaults.yaml`. A family file can refine those values
for an exact model geometry, so a small and large model in the same family do
not need to share scoring budgets. Unspecified model values inherit the family
profile, while an explicitly selected defaults file has the highest
default precedence. Setup then asks for the model and dataset, and requires
explicit acceptance or customization of infrastructure-specific worker and
cluster defaults:

```bash
python examples/puzzletron/puzzletron_setup_v2.py \
  --defaults examples/puzzletron/configs/setup/defaults.example.yaml
```

The example defaults use only repository-relative values. Copy the file and add
site-specific data, scheduler, and container settings before selecting it.
The defaults file is loaded only when passed explicitly and takes precedence
over the selected profile. To expose every per-section and nested setting, use
the advanced flow explicitly:

```bash
python examples/puzzletron/puzzletron_setup_v2.py --full
```

Press **Esc** to go back from any prompt. Selection prompts show a visible
**← Back** action, and text or numeric prompts accept `:back`.
Every accepted answer and the exact navigation frame are saved in
`answers_v2.yaml`, so an interrupted session can resume with:

```bash
python examples/puzzletron/puzzletron_setup_v2.py --resume /path/to/campaign
```

V2 supports reusable named parallel profiles, canonical per-stage execution
strategies, independent instance counts, scheduling-compatible effective batches,
multiple named vLLM workload/topology measurements, independent MIP goals with
internal constraints/variants/matrices, and editable post-MIP flow DAGs. The
recommended flow begins with online evaluation and uses LM loss for filtering;
it does not include Initial Filter. PTQ and downstream evaluation are shown as
reserved but unavailable.

The final review writes `resolved_defaults.yaml`, `README.md`, and validated
`smoke/` and `production/` bundles transactionally. The wizard never launches
the orchestrator.

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
            └── runs/default.yaml     # exact end-to-end experiment
```

Site-specific paths can be overridden without editing the checked-in config:

```bash
export PUZZLETRON_RUN_ROOT=/shared/puzzle_runs/my_campaign
```

Independent runs, variants, solution pools, resource constraints, and
homogeneous search are documented in [MIP runs](docs/mip_profiles.md). Configure
candidate evaluation, filtering, materialization, and distillation with
[post-MIP pipelines](docs/post_mip_pipeline.md).

## Run the complete pipeline manually

Choose one tested entry config:

```bash
export CONFIG=examples/puzzletron/configs/families/nemotron3/nano_30b_a3b_bf16/runs/default.yaml
source .venv/bin/activate
python examples/puzzletron/main.py \
  --config "$CONFIG" \
  --stage full \
  --gpus-per-node 8
```

`--stage full` executes every enabled stage in dependency order. Completed
stages with valid manifests and acceptance markers are skipped; use `--force`
only when intentionally invalidating and rerunning the selected work.

There is one current orchestration boundary to understand before using
`--stage full`. The canonical `zero_shot_evaluation` handler evaluates realized
checkpoints, while the Nano experiment uses `mode: online_solutions` to score
MIP architectures from one resident sorted teacher without materializing every
candidate. The canonical AIPerf handler also runs one topology, while the tested
Nano campaign uses a topology matrix. Until these paths are integrated into
`main.py`, run through `mip`, use the profile commands below, and then invoke
the canonical KD stages individually. A teacher-only canonical evaluation is
not completion of an online MIP profile.

On a scheduler, run the same command inside the site's container and launch
distributed stages with the topology declared by that stage's
`automodel.parallel` section. Do not assume one parallel recipe is valid for
all stages.

## Campaign orchestrator (v2)

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

Example dry-run:

```bash
python examples/puzzletron/orchestrate.py \
  --experiment examples/puzzletron/configs/families/nemotron3/nano_30b_a3b_bf16/runs/default.yaml \
  --runner examples/puzzletron/configs/orchestration/runner.slurm.example.yaml \
  --execution examples/puzzletron/configs/orchestration/execution.example.yaml \
  --dry-run
```

Submit on Slurm or bare metal:

```bash
python examples/puzzletron/orchestrate.py \
  --experiment "$CONFIG" \
  --runner "$RUNNER" \
  --execution "$EXECUTION" \
  --stage full
```

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

## Run step by step

Run one stage with the same entry config:

```bash
python examples/puzzletron/main.py \
  --config "$CONFIG" \
  --stage width_importance \
  --gpus-per-node 8
```

The authoritative dependencies and enabled-stage rules live in
[`stages/graph.py`](../../modelopt/torch/puzzletron/stages/graph.py).

| Stage | Purpose |
|---|---|
| `convert` | Convert the immutable Hugging Face teacher into the configured backend format. |
| `tokenize_data` | Build deterministic train and validation token caches. |
| `vllm_stats` | Measure exact runtime and memory costs for candidate subblocks. |
| `depth_importance` | Rank cumulative block or subblock removals. |
| `width_importance` | Collect activation-based rankings for every enabled width axis. |
| `sort` | Reorder the teacher so nested prefixes implement ranked width choices. |
| `sort_sanity` | Check that sorting preserves teacher outputs. |
| `width_sanity` | Compare ranked, random, and reverse slices on representative layers. |
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

Independent DAG branches may run concurrently when they have disjoint writers.
Long-running stages should resume their durable checkpoints or immutable shards
rather than restarting completed work.

## Online evaluation and downstream stages

Set the config and campaign directory explicitly so standalone tools and Hydra
resolve the same artifacts:

```bash
export CONFIG=examples/puzzletron/configs/families/nemotron3/nano_30b_a3b_bf16/runs/default.yaml
export PUZZLETRON_RUN_ROOT=/shared/puzzle_runs/nemotron3-nano
export PUZZLE_DIR="$PUZZLETRON_RUN_ROOT"
export PROFILE=runtime-075
```

After `mip`, prepare one deduplicated online-evaluation plan. Repeat
`--profile-id` for every configured profile; aliases ensure that an identical
architecture is evaluated once while remaining visible in every profile.

```bash
python examples/puzzletron/run_profile_online_evaluation.py \
  --puzzle-dir "$PUZZLE_DIR" --prepare \
  --profile-id params-075 \
  --profile-id runtime-075 \
  --profile-id memory-075 \
  --profile-id params-075-num-experts-only \
  --profile-id params-075-expert-dim-only \
  --profile-id params-075-num-experts-and-expert-dim
```

Run every width in the plan. One command is one resident model instance;
independent instances use distinct shard indices, and each receives the entire
stage-local AutoModel GPU mesh:

```bash
torchrun --standalone --nproc-per-node=8 \
  examples/puzzletron/run_profile_online_evaluation.py \
  --puzzle-dir "$PUZZLE_DIR" --config "$CONFIG" --run-shard \
  --width 2688 --shard-index 0 --shard-count 1 \
  --eval-samples 128 --block-size 8192 --micro-batch-size 4
```

After every width/shard pair finishes, merge the durable results:

```bash
python examples/puzzletron/run_profile_online_evaluation.py \
  --puzzle-dir "$PUZZLE_DIR" --merge \
  --eval-samples 128 --block-size 8192
```

Experiments using realized checkpoints instead call the canonical evaluator:

```bash
python examples/puzzletron/main.py \
  --config "$CONFIG" --stage zero_shot_evaluation --gpus-per-node 8
```

Materialize only the best online candidate needed by AIPerf and KD. This verifies
its physical parameter count and writes a registry containing the candidate and
teacher:

```bash
python examples/puzzletron/prepare_online_profile_finalists.py \
  --puzzle-dir "$PUZZLE_DIR" --config "$CONFIG" \
  --profile-id "$PROFILE" --count 1
```

The official AIPerf package installed above provides the default `aiperf`
executable; set `AIPERF_EXECUTABLE` only to select a different installation.

For one topology, call the canonical AIPerf stage:

```bash
python examples/puzzletron/main.py \
  --config "$CONFIG" --stage aiperf --gpus-per-node 8
```

For the tested all-eight-GPU topology matrix, launch one profile worker per node.
Under Slurm, each task must see all eight GPUs and derives its shard identity
from `SLURM_PROCID` and `SLURM_NTASKS`. Merge once after every worker exits:

```bash
srun --nodes="$NODES" --ntasks="$NODES" --ntasks-per-node=1 \
  python examples/puzzletron/run_profile_aiperf_worker.py \
  --puzzle-dir "$PUZZLE_DIR" --profile-id "$PROFILE" \
  --input-tokens 8192 --output-tokens 1024

python examples/puzzletron/run_profile_aiperf_worker.py \
  --puzzle-dir "$PUZZLE_DIR" --profile-id "$PROFILE" \
  --input-tokens 8192 --output-tokens 1024 --merge
```

Use the same selected registry for KD. Run the frozen-minibatch overfit gate
before production distillation; use `run_multinode_stage.sh` with these stage
names when the configured stage mesh spans multiple nodes:

```bash
python examples/puzzletron/main.py \
  --config "$CONFIG" --stage global_distillation_sanity --gpus-per-node 8

python examples/puzzletron/main.py \
  --config "$CONFIG" --stage global_distillation --gpus-per-node 8
```

Finally evaluate the consolidated global-KD checkpoint:

```bash
python examples/puzzletron/main.py \
  --config "$CONFIG" --stage post_distillation_evaluation --gpus-per-node 8
```

## Reports

`main.py` refreshes the report before and after each stage and records the
currently running stage. Standalone profile workers do not own the DAG, so
regenerate after online-evaluation merge, finalist realization, AIPerf merge,
or any other manual artifact publication. The generator is read-only with
respect to model artifacts and includes valid partial results without marking
their stage complete.

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
