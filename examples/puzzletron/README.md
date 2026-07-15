# Puzzletron

Puzzletron finds smaller, faster variants of large language models by searching
over pruning configurations, evaluating them on real hardware, and optionally
fine-tuning the best results — all while keeping an auditable trail from the
original teacher checkpoint.

**New model / first time?** → [Use the AI skill](#using-the-ai-skill-recommended)  
**Have a config already?** → [Run the pipeline manually](#running-the-pipeline)  
**Reading results?** → [Understanding the report](#reading-the-html-report)

---

## Installation

Puzzletron needs four sibling packages: **ModelOpt** (this repo), a patched
**vLLM fork**, **AIPerf**, and **AutoModel**. All four must be in the same
Python environment with a compatible PyTorch/CUDA build.

> The vLLM fork, AIPerf, and AutoModel will be published to GitHub. Until
> then install them from your local cluster clones.

```bash
# Activate (or create) your environment
source /path/to/.venv/bin/activate
# or: python -m venv .venv && source .venv/bin/activate

# Install all four sibling packages in editable mode
python -m pip install -e /path/to/modelopt       # this repo
python -m pip install -e /path/to/vllm           # patched vLLM fork
python -m pip install -e /path/to/aiperf         # AIPerf benchmarker
python -m pip install -e /path/to/Automodel      # AutoModel backend

# Extra deps for the examples
python -m pip install -r examples/puzzletron/requirements.txt

# Make the repo importable
export PYTHONPATH="/path/to/modelopt:${PYTHONPATH:-}"
```

After any compiled sibling rebuild (vLLM, custom kernels), re-run that
package's `pip install -e .` and verify a GPU forward passes cleanly:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

Record exact package versions and `torch.version.cuda` before a production run.

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
  scoring path/to/experiment.yaml path/to/recipe.yaml
```

The script reads `PUZZLETRON_IMAGE` and `PUZZLETRON_CONTAINER_MOUNTS` from
the environment — keep concrete cluster paths in an ignored shell export file,
not in committed configs.

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
python - <<'PY'
from modelopt.torch.puzzletron.diagnostics.campaign_progress_report import (
    generate_campaign_progress_report,
)
result = generate_campaign_progress_report(
    "puzzle_runs/my-experiment",
    model_name="My Model",
)
print(result)
PY
```

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
