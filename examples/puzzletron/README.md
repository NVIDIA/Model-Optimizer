# Puzzletron

Puzzletron finds smaller, faster variants of large language models by searching
over pruning configurations, evaluating them on real hardware, and optionally
fine-tuning the best results — all while keeping an auditable trail from the
original teacher checkpoint.

**New model / first time?** → [Use the AI skill](#using-the-ai-skill-recommended)  
**Have a config already?** → [Run the pipeline manually](#running-the-pipeline)  
**Reading results?** → [Understanding the report](#reading-the-html-report)

---

## Reproducible reference campaigns

These configs are experiment-specific entrypoints, not generic model defaults.
Their tutorials record the workload, search space, parallelism, artifact reuse,
selection policy, and downstream commands used by the corresponding report.

| Model / experiment | Exact config | Step-by-step tutorial | Current report |
|---|---|---|---|
| NVIDIA Nemotron-3 Nano 30B-A3B production | [`production.yaml`](configs/clean/families/nemotron3/nano_30b_a3b_bf16/production.yaml) | [Nano reproduction](docs/nemotron3_nano_30b_a3b_reproduction.md) | [Nano HTML](../../puzzle_runs/nemotron3-nano-30b-a3b-puzzletron-v1/artifacts/campaign_report/campaign_report.html) |
| Qwen3.5-9B sanity | [`sanity_reproduction.yaml`](configs/clean/families/qwen3_5/qwen3_5_9b/sanity_reproduction.yaml) | [Qwen sanity reproduction](docs/qwen3_5_9b_sanity_reproduction.md) | [Qwen HTML](../../../puzzle_runs/qwen3_5/qwen3_5_9b/sanity_check/artifacts/campaign_report/campaign_report.html) |

The HTML paths are local campaign outputs and are intentionally not committed.
Regenerate them from the preserved artifacts before sharing.

---

## Installation

Puzzletron uses one Python environment for **ModelOpt**, the patched **vLLM**
fork, **AutoModel**, and **AIPerf**. Install PyTorch first and build every CUDA
extension against that same installation; mixing PyTorch or CUDA builds can
cause import errors or incorrect GPU execution.

### 1. Start from the CUDA development image

Use this image for both local containers and cluster jobs:

```text
nvcr.io/nvidia/cuda:12.9.2-cudnn-devel-ubuntu24.04
```

For example, with Docker, mount a workspace that contains ModelOpt and will
hold the two sibling checkouts:

```bash
export PUZZLETRON_WORKSPACE=/absolute/path/to/workspace
docker run --gpus all --ipc=host --rm -it \
  -v "${PUZZLETRON_WORKSPACE}:/workspace" \
  -w /workspace \
  nvcr.io/nvidia/cuda:12.9.2-cudnn-devel-ubuntu24.04 bash
```

Inside the container, install Python 3.12 and the build tools used by the
editable packages and optional CUDA extensions:

```bash
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y \
  build-essential cmake git ninja-build \
  python3 python3-dev python3-pip python3-venv
```

Set the checkout paths, then clone the tracked Puzzletron branches:

```bash
export MODEL_OPT_ROOT=/workspace/modelopt
export VLLM_ROOT=/workspace/vllm
export AUTOMODEL_ROOT=/workspace/Automodel

git clone --branch feature/add_anymodel_to_vllm --single-branch \
  https://github.com/Separius/vllm.git "${VLLM_ROOT}"
git clone --branch puzzletron --single-branch \
  https://github.com/Separius/Automodel.git "${AUTOMODEL_ROOT}"
```

The remaining commands assume this sibling layout:

```text
/workspace/
├── modelopt/
├── vllm/
└── Automodel/
```

### 2. Create the virtual environment and install PyTorch

The patched vLLM checkout pins PyTorch 2.11.0 and uses CUDA 12.9 by default.
Install that combination before any package that compiles CUDA code:

```bash
python3 -m venv /workspace/.venv
source /workspace/.venv/bin/activate

python -m pip install --upgrade \
  pip "setuptools>=80,<81" "setuptools-scm>=8" setuptools-rust \
  wheel "packaging>=24.2" "cmake>=3.26.1" ninja jinja2

python -m pip install \
  torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 \
  --index-url https://download.pytorch.org/whl/cu129
```

### 3. Install Puzzletron's runtime packages

The default vLLM command reuses precompiled CUDA extensions while keeping the
patched Python source editable. A full vLLM CUDA/C++ source build requires the
separate vLLM source-build workflow.

```bash
VLLM_USE_PRECOMPILED=1 VLLM_PRECOMPILED_WHEEL_VARIANT=cu129 \
  python -m pip install --no-build-isolation -e "${VLLM_ROOT}"

python -m pip install -e "${AUTOMODEL_ROOT}"
python -m pip install aiperf
python -m pip install -e "${MODEL_OPT_ROOT}[hf,puzzletron]"
python -m pip install math-verify ray
```

Do not add `--no-deps` to these commands. The packages need their declared
Python dependencies, and their PyTorch requirements accept the already
installed vLLM-compatible version.

### 4. Install model-specific kernels

Run only the commands needed by the target model. Run more than one when the
architecture combines these layer types.

For mixture-of-experts (MoE) models:

```bash
python -m pip install --no-build-isolation \
  "git+https://github.com/fanshiqing/grouped_gemm@v1.1.4"
```

For Mamba models:

```bash
python -m pip install "mamba-ssm[causal-conv1d]" --no-build-isolation
```

For linear-attention models:

```bash
python -m pip install "flash-linear-attention[cuda]"
```

`--no-build-isolation` makes compiled extensions use the PyTorch 2.11.0+cu129
installation in the active venv. It is different from `--no-deps`: dependencies
are still installed.

### 5. Verify the environment

Run the checks inside the same container and venv used for Puzzletron jobs:

```bash
test "$(git -C "${VLLM_ROOT}" remote get-url origin)" = \
  "https://github.com/Separius/vllm.git"
test "$(git -C "${VLLM_ROOT}" branch --show-current)" = \
  "feature/add_anymodel_to_vllm"
test "$(git -C "${AUTOMODEL_ROOT}" remote get-url origin)" = \
  "https://github.com/Separius/Automodel.git"
test "$(git -C "${AUTOMODEL_ROOT}" branch --show-current)" = "puzzletron"

git -C "${MODEL_OPT_ROOT}" rev-parse HEAD
git -C "${VLLM_ROOT}" rev-parse HEAD
git -C "${AUTOMODEL_ROOT}" rev-parse HEAD
```

```bash
python - <<'PY'
import importlib.metadata as metadata

import aiperf
import modelopt
import nemo_automodel
import torch
import vllm

for package in ("torch", "vllm", "nemo-automodel", "aiperf", "nvidia-modelopt"):
    print(package, metadata.version(package))

print("torch CUDA", torch.version.cuda)
print("CUDA available", torch.cuda.is_available())
print("modelopt", modelopt.__file__)
print("vllm", vllm.__file__)

assert torch.__version__.startswith("2.11.0")
assert torch.version.cuda == "12.9"
assert torch.cuda.is_available()
PY

python -m pip check
```

Record the output and the three resolved source revisions before production
runs. Re-run the verification after pulling either tracked branch or rebuilding
vLLM or any optional CUDA extension.

---

## Using the AI Skill (Recommended)

The **running-puzzletron** skill handles the whole workflow: model inspection,
question intake, config generation, smoke validation, and production execution.

Invoke it in Claude Code:

```
Please use the puzzletron skill to prune meta-llama/Llama-3.1-8B-Instruct
```

or, with the explicit skill path:

```
@.claude/skills/running-puzzletron prune HuggingFace model <model-id>
```

### What the Skill Does

1. **Inspects your model** — reads the HuggingFace source files to enumerate
   every layer type before proposing anything.

2. **Asks two rounds of targeted questions** — first about your infrastructure
   and goals, then about the search configuration. See
   [Understanding the Skill's Questions](#understanding-the-skills-questions)
   for plain-language guidance on each question.

3. **Handles unsupported models** — if your model has no Puzzletron descriptor
   yet, the skill walks through adding one and validating the pruning axes.
   If it is already supported, it jumps directly to campaign configuration.

4. **Generates runnable config bundles** — a tiny smoke config to validate
   the setup end-to-end, and a production config for the real run.

5. **Executes and monitors** — runs the pipeline DAG in dependency order,
   handles Slurm or bare-metal scheduling, resumes interrupted stages, and
   keeps the HTML report current after each stage.

### Understanding the Skill's Questions

#### Round 1 — Infrastructure and Goals

The skill asks these once, grouped in one message. You do not need to know
all the answers up front; ask for help with any item.

| What the skill asks | Plain-language meaning | Example answer |
|---|---|---|
| Checkpoint path or URI | Where is the model? Local lustre path or HuggingFace model ID. | `meta-llama/Llama-3.1-8B-Instruct` or `/lustre/models/llama3` |
| Tokenizer / processor revision | Which version of the tokenizer to use. "Same as the model" is almost always correct. | `same as model` |
| Experiment ID | A short label for this run; used to name the artifact folder. | `llama3-8b-prune-v1` |
| Artifact root | Where outputs, checkpoints, and the HTML report should be saved. | `/lustre/puzzle_runs` |
| Resume existing artifacts? | If you have already run some stages, can they be reused? | `yes, resume what is already there` |
| Execution environment | Slurm cluster or bare-metal SSH machines? | `Slurm, NVIDIA DGX cluster` |
| Container or host Python? | Do you run inside an Enroot/Docker container, or directly on the host Python? | `Enroot container at /lustre/containers/nemo.sqsh` |
| Sibling checkout locations | Where are vLLM, AIPerf, and AutoModel installed on your cluster? | `all siblings are in /home/user/repos/` |
| Available nodes/GPUs/wall-time | How many resources do you have and for how long per job? | `4 × H100 nodes × 8 GPU each, 4-hour Slurm slots` |
| Storage capacity | How much disk space is free for artifacts? Large models can use 100s of GB. | `10 TB on /lustre/puzzle_runs` |
| Required final outcomes | How far do you want the pipeline to go? | `MIP search + zero-shot eval; skip distillation for now` |

#### Round 2 — Model-Specific Search Configuration

The skill asks these after inspecting your model's source code, so the
options presented are always valid for your specific architecture.

| What the skill asks | Plain-language meaning | Example answer |
|---|---|---|
| Data lanes (modalities) | Does this model handle text only, or also images/audio/video? | `text only` |
| Calibration dataset | Which dataset to use for importance estimation. | `wikitext-103, 512 samples` |
| Batch layout | How to pack sequences: fixed-length, padded, or packed (multiple docs per sequence). Packed is most GPU-efficient. | `packed` |
| Pruning axis ranges | For each prunable dimension (e.g. number of attention heads, FFN width), what is the smallest you are willing to go? | `num_heads: 8–32, ffn_dim: 2048–14336` |
| Depth removals | How many transformer layers to allow the search to remove. | `up to 4 layers` |
| Bypass | Whether to run the block-replacement search. This produces richer scoring data but adds significant compute. | `yes, block granularity` |
| vLLM stats | Whether to benchmark candidates in real vLLM for hardware-aware search. Needed for latency constraints. | `yes` |
| MIP constraints | The hardware budget for accepted solutions: max parameter count, latency per token, or GPU memory. | `≤ 6B params, ≤ 2 ms/token at BS=1` |
| Number of top solutions | How many MIP solutions to evaluate or distill. | `top 3 for eval, top 1 for distillation` |
| Parallelism | TP/PP/CP/EP/DP degrees for smoke and production runs. The skill validates these against your GPU count. | `TP=2, PP=1, DP=4` |

### What Happens for Unsupported Models

If your model has no Puzzletron descriptor:

1. The skill reads the HuggingFace layer implementations (attention, FFN, MoE,
   normalization, etc.) and proposes pruning axes with source evidence.
2. You review the proposed axes. The skill explains which dimensions are safe
   to prune and which have constraints (e.g., grouped-query heads require
   specific alignment; tied embeddings cannot be pruned independently).
3. You confirm or adjust, and the skill writes the model descriptor.
4. A small equivalence test confirms that dynamic slicing matches physical
   export — this is mandatory before any campaign compute.
5. From here, the workflow proceeds identically to a supported model.

If your model **is already registered** in Puzzletron, the skill skips the
inspection and descriptor steps and moves directly to Round 2.

---

## Running the Pipeline Manually

If you already have config files (generated by the skill or written by hand),
you can drive the pipeline without the AI.

### Setup: fill in local paths

```bash
cp examples/puzzletron/configs/clean/my_paths.example.yaml \
   examples/puzzletron/configs/clean/my_paths.yaml
# Edit my_paths.yaml — set checkpoint_root, dataset_root, artifact_root,
# container_image, account, etc. for your cluster.
```

### Run the full pipeline

```bash
python examples/puzzletron/main.py \
  --config examples/puzzletron/configs/clean/families/qwen3_5/qwen3_5_0_8b/smoke_test.yaml \
  --stage full \
  --gpus-per-node 8
```

`--stage full` runs every enabled stage in dependency order. The orchestrator
spawns an isolated worker process for each stage, so GPU memory and distributed
state cannot leak between stages.

### Run a single stage

```bash
python examples/puzzletron/main.py \
  --config path/to/experiment.yaml \
  --stage width_importance \
  --gpus-per-node 8
```

Stage names come from
[`modelopt/torch/puzzletron/stages/graph.py`](../../modelopt/torch/puzzletron/stages/graph.py).
Common values: `convert`, `tokenize`, `width_importance`, `sort`, `depth`,
`bypass`, `scoring`, `mip`, `evaluation`, `distillation`.

### Override a config value without editing the file

```bash
python examples/puzzletron/main.py \
  --config path/to/experiment.yaml \
  --stage bypass \
  --override bypass.granularity=subblock
```

### Force-rerun a completed stage

Normally the runner skips stages whose completion markers are valid. Pass
`--force` to redo one intentionally:

```bash
python examples/puzzletron/main.py \
  --config path/to/experiment.yaml \
  --stage sort \
  --force
```

### Verify your config before allocating GPUs

```bash
python - <<'PY'
from modelopt.torch.puzzletron.pipeline_config import pipeline_config_from_path
cfg = pipeline_config_from_path("path/to/experiment.yaml")
print(cfg["experiment"]["dir"])
PY
```

### Multi-node on Slurm

For distributed stages that span multiple nodes:

```bash
export PUZZLETRON_IMAGE="/path/to/container.sqsh"
export PUZZLETRON_CONTAINER_MOUNTS="/lustre:/lustre"
# optional: export PUZZLETRON_SETUP_ENV="/path/to/env-setup.sh"

sbatch --nodes=4 --gpus-per-node=8 \
  examples/puzzletron/run_multinode_stage.sh \
  replacement_scoring path/to/experiment.yaml
```

The script reads `PUZZLETRON_IMAGE` and `PUZZLETRON_CONTAINER_MOUNTS` from
the environment — keep concrete cluster paths in an ignored shell export file,
not in committed configs. Each model-loading stage owns its mesh under
`<stage>.automodel.parallel`; there is no shared AutoModel recipe YAML.

### Bare metal (no scheduler)

```bash
torchrun \
  --nnodes=4 \
  --nproc-per-node=8 \
  --rdzv-backend=c10d \
  --rdzv-endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
  --max-restarts=0 \
  examples/puzzletron/main.py \
  --config path/to/experiment.yaml \
  --stage scoring \
  --gpus-per-node 8
```

Require passwordless SSH, shared storage visible at identical paths on every
host, and per-host log files. Clean up all remote workers explicitly after a
failure.

---

## Understanding the Pipeline Stages

The pipeline is a dependency DAG — independent branches can (and should) run
concurrently when resources allow.

```
Convert Checkpoint
├── Tokenize Data
│   ├── Depth Importance Estimation ──────────────────────────┐
│   └── Width Importance Estimation                           │
│       └── Sort Checkpoint                                   │
│           ├── [Sanity checks: sort / width / slicing]       │
│           └── Bypass                                        │
│               └── Build Block Library                       │
│                   └── Replace-one Scoring ──────────────────┤
└── vLLM Stats ────────────────────────────────────────────────┘
                                                              │
                                           MIP Search ◄───────
                                           ├── AIPerf
                                           └── Zero-shot Evaluation
                                               └── Global Distillation
                                                   └── Post-KD Evaluation
```

| Stage | What it does | Typical resource |
|---|---|---|
| **Convert** | Converts the HF checkpoint into Puzzletron format | 1 node, 1 GPU |
| **Tokenize** | Runs calibration data through the tokenizer; produces sample hashes | 1 node, CPU |
| **Width Importance** | Measures per-head/neuron importance with activation statistics | 1 node, all GPUs |
| **Depth Importance** | Estimates the accuracy cost of removing each transformer layer | 1 node, all GPUs |
| **Sort** | Reorders parameters from most to least prunable | 1 node, 1 GPU |
| **Sanity checks** | Confirm that sorted ≈ original, physical ≈ dynamic (bugs are failures) | 1 node, 1 GPU |
| **vLLM Stats** | Benchmarks every candidate configuration in real vLLM for latency | 1–N nodes, all GPUs |
| **Bypass** | Trains local proxy losses per block/subblock — the main search data source | 1–N nodes, all GPUs |
| **Build Block Library** | Assembles cost/score records for all candidate configurations | 1 node, 1 GPU |
| **Replace-one Scoring** | Evaluates each candidate by swapping one block at a time | 1–N nodes, all GPUs |
| **MIP Search** | Solves a constrained optimization to find Pareto-optimal model sizes | 1 node, CPU |
| **AIPerf** | Sweeps concurrency and topology settings for top solutions | 1–N nodes, all GPUs |
| **Zero-shot Evaluation** | Evaluates top solutions on standard benchmarks | 1 node, all GPUs |
| **Global Distillation** | Fine-tunes the best pruned model to recover accuracy | 1–N nodes, all GPUs |
| **Post-KD Evaluation** | Evaluates the distilled model | 1 node, all GPUs |

---

## Reading the HTML Report

The report at
`<artifact-root>/<experiment-id>/artifacts/campaign_report/campaign_report.html`
is updated automatically after each stage. Open it in any browser.

Regenerate it from existing artifacts without rerunning anything:

```bash
python examples/puzzletron/generate_campaign_progress_report.py \
  --puzzle-dir puzzle_runs/my-experiment \
  --model-name "My Model"
```

Normal generation reuses versioned section snapshots under
`artifacts/campaign_report/section_cache/` and rebuilds only sections whose inputs changed. Force a
section and its declared dependents with `--rebuild-section replacement`, or ignore all snapshots
with `--no-cache`. The selected snapshot identities, cache hits/misses, and extraction timings are
recorded in `artifacts/campaign_report/report_manifest.json`.

The cache directory and manifest are disposable accelerators. Removing them is safe; the next run
is a cold rebuild from canonical campaign artifacts. The generated `campaign_report.html` remains
self-contained, and it is the only file required when sharing the report.

### What Each Section Shows

| Section | What to look for |
|---|---|
| **Experiment header** | Experiment ID, model name, config hash, key resolved settings. Check this first to confirm the right checkpoint and dataset were used. |
| **Pipeline DAG** | Visual graph of all stages: green = complete, grey = pending, strikethrough = disabled. Use this to see what is left. |
| **Artifact provenance** | File paths, hashes, and the stage that produced each artifact. Useful for auditing resume correctness. |
| **Sorting** | Original vs. sorted vs. reverse-sorted model output comparison. Sorted and original should match; large differences indicate a bug. |
| **Width ranking** | Whether the importance ordering actually helps at reduced widths. A flat curve means the ranking is not informative. |
| **Slicing sanity** | Dynamic pruning vs. physical export comparison. Any mismatch is a bug, not a tolerance question. |
| **Bypass** | Scatter plots of local loss vs. parameter ratio per block. A good bypass run shows a clear downward trend. Hover over a point to see its exact configuration. Click a block to view all candidates for that block. |
| **vLLM runtime** | Per-candidate latency from real vLLM. Used as the MIP latency constraint. |
| **Scoring** | Replace-one accuracy drop per candidate. Lower = better (less accuracy lost). |
| **MIP results** | Pareto front of solutions. Each point is a valid model config that satisfies your hardware constraints. The table lists parameter count, latency, and accuracy for each. |
| **Zero-shot evaluation** | Benchmark scores for each MIP solution and the teacher. Check the accuracy gap vs. teacher. |

See [MIP profiles](docs/mip_profiles.md) for named workloads, combined
constraints, restricted search spaces, and homogeneous top-k solutions.
| **AIPerf** | Throughput (tokens/sec) and latency curves across concurrency values and topology configurations. |
| **Distillation** | Training loss curves. `main_ce`, `mtp_ce`, `main_kd`, `mtp_kd` are tracked separately. A successful run shows all terms decreasing and overfit on the sanity check. |
| **Warnings** | Appear in yellow beside affected table cells or stage nodes. A warning does not block a stage, but must be understood before trusting the result. |

---

## Troubleshooting

**Stage skipped unexpectedly** — The runner skips stages with valid completion
markers. This is expected resume behavior. Use `--force` to explicitly redo
a completed stage.

**OOM during scoring or bypass** — Reduce `local_batch_size` or enable the
chunked flash KD path (`scoring.use_flash_kd: true`). For large-vocabulary
models (>100k tokens), full-vocab loss tensors are the usual culprit.

**Dynamic ≠ physical outputs in slicing sanity** — This is always a code bug.
Debug normalization inputs, residual paths, rotary embeddings, grouped
projections, or tied weights. Do not adjust the physical path to match dynamic.

**Multi-node job hangs at NCCL init** — Verify `MASTER_ADDR` is reachable on
all nodes and `MASTER_PORT` is not blocked by a firewall. Check per-node logs
(`node0.log`, `node1.log`) for the first divergence.

**Resume after scheduler timeout** — Resubmit the same command. Bypass and KD
write transactional checkpoints; scoring and vLLM stats use shard manifests.
The stage detects and skips already-complete work automatically.

**Report missing a section** — The generator only emits sections for which
artifacts exist. If bypass artifacts are missing, rerun the bypass stage.
