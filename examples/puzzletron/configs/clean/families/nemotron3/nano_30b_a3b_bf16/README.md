# Reproduce the Nemotron-3 Nano 30B-A3B pruning campaign

This tutorial reproduces `nemotron3-nano-30b-a3b-puzzletron-v1`. It is a live
runbook: commands move into the verified sequence only after their stage output
and regenerated campaign report have both passed. The production config is the
source of truth for all search values and budgets; the smoke config inherits it
and only reduces workload sizes.

## 1. Fixed inputs

Run from the ModelOpt repository root. Do not replace any of these identities
when reproducing this campaign.

```bash
export ROOT=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_llm/users/ssameni/engineering/modelopt_qwen
export IMAGE=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_llm/users/ssameni/enroot/pytorch_25p05.sqsh
export VLLM_IMAGE=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_llm/users/ssameni/enroot/cuda_12p9p2_cudnn_devel_ubuntu24p04.sqsh
export SETUP_ENV=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_llm/users/ssameni/setup-envs.sh
export MODEL_REVISION=cbd3fa9f933d55ef16a84236559f4ee2a0526848
export MODEL_SNAPSHOT=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_llm/users/ssameni/hf/hub/models--nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/snapshots/${MODEL_REVISION}
export DATASET=${ROOT}/../Puzzle-KD-Nemotron-Post-Training-Dataset-v2
export CONFIG=examples/puzzletron/configs/clean/families/nemotron3/nano_30b_a3b_bf16/production.yaml
export SMOKE_CONFIG=examples/puzzletron/configs/clean/families/nemotron3/nano_30b_a3b_bf16/smoke.yaml
export RUN_ROOT=puzzle_runs/nemotron3-nano-30b-a3b-puzzletron-v1
export SMOKE_ROOT=${RUN_ROOT}/smoke
export VLLM_REVISION=a056958c78226dcc5476ad5083a26155dd8863c5
export AUTOMODEL_REVISION=b22cd029d806197e249f2cc4a42c5de91713b772
```

Record the exact repository, config, setup, and package identities before doing
work. The commands importing Python packages must run on a compute node.

```bash
git rev-parse HEAD
git status --short
sha256sum "$CONFIG" "$SMOKE_CONFIG" "$SETUP_ENV"
```

The Nano architecture has `moe_latent_size=None`. Its routed expert residual
width is therefore sliced on every expert's `up_proj` input and `down_proj`
output. Do not enable `moe_latent_dim`. Nemotron Super is different: it has a
latent MoE and slices the residual width through `fc1_latent_proj` and
`fc2_latent_proj`.

## 2. Compute-node invariant

The login node is only for scheduler queries and read-only Bash, `sed`, `rg`, or
Python-standard-library inspection. Every test, import, config resolution,
report, tokenization, conversion, and model command runs through `srun` in the
specified image. At the beginning of every compute task, source the cache setup
script and then the repository virtual environment, in that order.

For an edit-test-debug loop, acquire one reusable eight-GPU interactive node.
Set `ACCOUNT` to the Slurm account associated with the original campaign.

```bash
cd "$ROOT"
salloc -p interactive -t 4:00:00 -A "$ACCOUNT" --nodes=1 --gpus-per-node=8
export JOBID=$SLURM_JOB_ID
```

Relay a CPU-only or focused test task to that same node. Omitting GPU flags from
the step is intentional; the allocation is retained for the next GPU command.

```bash
srun --jobid="$JOBID" --overlap --nodes=1 --ntasks=1 \
  --container-image="$IMAGE" --container-mounts=/lustre:/lustre \
  --container-workdir="$ROOT" /bin/bash -lc '
    set -Eeuo pipefail
    source /lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_llm/users/ssameni/setup-envs.sh
    source ./.venv_new/bin/activate
    export PYTHONPATH="$PWD:${PYTHONPATH:-}"
    export PYTHONUNBUFFERED=1
    python --version
  '
```

Relay an eight-GPU stage with this shape:

```bash
srun --jobid="$JOBID" --overlap --nodes=1 --ntasks=1 --gpus-per-task=8 \
  --container-image="$IMAGE" --container-mounts=/lustre:/lustre \
  --container-workdir="$ROOT" /bin/bash -lc '
    set -Eeuo pipefail
    source /lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_llm/users/ssameni/setup-envs.sh
    source ./.venv_new/bin/activate
    export PYTHONPATH="$PWD:${PYTHONPATH:-}"
    export PYTHONUNBUFFERED=1
    python examples/puzzletron/main.py \
      --config examples/puzzletron/configs/clean/families/nemotron3/nano_30b_a3b_bf16/smoke.yaml \
      --stage STAGE
  '
```

The verified one-model topology is PP=2, EP=4, TP=CP=DP=1. It uses all eight
GPUs and peaked at about 10.5 GiB per rank during smoke importance estimation.
Only switch to PP=4, EP=2 after reproducing an out-of-memory failure. Each model
replica loads on exactly one node.

Width importance is one coordinated scoring experiment. On one node it uses the
verified PP=2/EP=4 replica. To scale it beyond one node, place one PP=2/EP=4
replica on each node and increase DP to the node count so all replicas contribute
to the same aggregated importance pass; do not shard axes into unrelated runs.
Depth importance is different: each no-op replacement can be evaluated by an
independent one-node model instance, so distribute distinct sublayer cases across
nodes and merge the validated results.

### Clean vLLM environment

The vLLM statistics retry introduced `.venv_new` and a persistent squashfs
of the CUDA 12.9 development image from `examples/puzzletron/README.md`.
Historical preparation stages used `.venv` and `pytorch_25p05.sqsh`; every
remaining campaign task, including reports, bypass, scoring, MIP, KD, and
AIPerf, uses `.venv_new` with the CUDA 12.9 image. The
source image is `nvcr.io/nvidia/cuda:12.9.2-cudnn-devel-ubuntu24.04`, while
`VLLM_IMAGE` above prevents every compute node from importing its 10.7 GiB
filesystem again. The exact sibling
checkouts are `../vllm_new` at `${VLLM_REVISION}` and `../Automodel_new` at
`${AUTOMODEL_REVISION}`. Install ModelOpt with both required extras:

```bash
python -m pip install -e ".[hf,puzzletron]"
```

The precompiled editable vLLM wheel did not install `_C` or `_moe_C`. Configure
the source tree once for H100 (`sm_90a`), then incrementally build and install
both components in the same compute image and environment:

```json
{
  "version": 6,
  "cmakeMinimumRequired": {"major": 3, "minor": 26, "patch": 1},
  "configurePresets": [
    {
      "name": "release",
      "binaryDir": "${sourceDir}/cmake-build-release",
      "cacheVariables": {
        "CMAKE_CUDA_COMPILER": "/usr/local/cuda/bin/nvcc",
        "CMAKE_BUILD_TYPE": "Release",
        "VLLM_PYTHON_EXECUTABLE": "/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_llm/users/ssameni/engineering/modelopt_qwen/.venv_new/bin/python",
        "CMAKE_INSTALL_PREFIX": "${sourceDir}",
        "CMAKE_CUDA_FLAGS": "",
        "NVCC_THREADS": "4",
        "CMAKE_JOB_POOLS": "compile=32"
      },
      "generator": "Ninja"
    }
  ],
  "buildPresets": [
    {"name": "release", "configurePreset": "release", "jobs": 32}
  ]
}
```

Save that content as `../vllm_new/CMakeUserPresets.json`, then run:

```bash
cd ../vllm_new
cmake --preset release
cmake --build --preset release --target _C _moe_C -j 16
cmake --install cmake-build-release --component _C
cmake --install cmake-build-release --component _moe_C
cd ../modelopt_qwen

python - <<'PY'
import importlib
import torch

core = importlib.import_module("vllm._C")
moe = importlib.import_module("vllm._moe_C")
assert hasattr(torch.ops._moe_C, "grouped_topk")
assert hasattr(torch.ops._moe_C, "moe_align_block_size")
print(core.__file__, moe.__file__, torch.__version__, torch.version.cuda)
PY
python -m pip check
```

Run these commands through `srun` with `VLLM_IMAGE`, sourcing `setup-envs.sh`
and then `.venv_new` at the start of the task. The verified environment is
Python 3.12.11, PyTorch 2.11.0+cu129, vLLM `${VLLM_REVISION}`, and AutoModel
`${AUTOMODEL_REVISION}`. `VLLM_USE_FUSED_MOE_GROUPED_TOPK` is an
experiment-local vLLM setting; do not export it globally.

The Nano production config also sets
`vllm_stats.runtime_stats.ignore_negatives: true`. This is a campaign-local
aggregation policy: it preserves and warns on the three measured 2432-width
MoE prefill marginals below zero instead of clamping them. The shared base
config remains strict (`false`). The verified canonical artifact contains 957
records across hidden widths 2688, 2560, and 2432.

Before enabling the production 8192/1024 measurement, run the exact focused
real-model gate:

```bash
sbatch \
  puzzle_runs/nemotron3-nano-30b-a3b-puzzletron-v1/commands/continue_vllm_new_probe.sbatch
```

It uses four visible H100s with TP2/PP2, a 256-token prefill, 32 generated
tokens, two warmups, three measured iterations, repeat count two, and only the
teacher/64-expert MoE comparison plus the three architecture anchors. The ten
benchmark specs cache independently under `smoke_vllm_new_probe/runtime_cache`,
so a preemption-safe retry resumes completed measurements. Treat the gate as
passed only after all ten specs produce the final statistics and manifest.

## 3. Validate the configs

Run the immutable campaign assertions on the compute node before every new
campaign or after any config edit:

```bash
srun --jobid="$JOBID" --overlap --nodes=1 --ntasks=1 \
  --container-image="$IMAGE" --container-mounts=/lustre:/lustre \
  --container-workdir="$ROOT" /bin/bash -lc '
    set -Eeuo pipefail
    source /lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_llm/users/ssameni/setup-envs.sh
    source ./.venv_new/bin/activate
    export PYTHONPATH="$PWD:${PYTHONPATH:-}"
    python puzzle_runs/nemotron3-nano-30b-a3b-puzzletron-v1/commands/validate_configs.py \
      --smoke examples/puzzletron/configs/clean/families/nemotron3/nano_30b_a3b_bf16/smoke.yaml \
      --production examples/puzzletron/configs/clean/families/nemotron3/nano_30b_a3b_bf16/production.yaml \
      --output puzzle_runs/nemotron3-nano-30b-a3b-puzzletron-v1/configs/resolved_configs.yaml
  '
```

The production search domain is:

- hidden width: `2688, 2560, 2432`;
- Mamba head dimension: `64, 56, 48`;
- Mamba heads: `64, 56, 48`;
- routed expert intermediate: `1856, 1600, 1344, 1088, 832`;
- shared expert intermediate: `3712, 3072, 2560, 2048`;
- routed experts: `128, 112, 96, 80, 64`;
- experts per token: `6, 4, 2`;
- KV heads: `2, 1`;
- query heads per KV group: `16, 12, 8, 4`.

## 4. Verified smoke sequence

Run one stage at a time in this order. Conversion and tokenization are CPU-bound
but still run on the compute node in the exact environment. Width and depth
importance use the eight-GPU relay above.

```text
convert
tokenize_data
width_importance
depth_importance
sort
sort_sanity
width_sanity
slicing_sanity
bypass_sanity
bypass
build_library
replacement_scoring
vllm_stats
mip
zero_shot_evaluation
global_distillation_sanity
global_distillation
post_distillation_evaluation
aiperf
```

The first five smoke stages through `sort` are verified for this campaign.
Production `convert`, `tokenize_data`, `width_importance`, and `sort` are also
complete and report-verified.
Later commands remain in dependency order but are not declared result-verified
in this document until their report gates pass.

For example, the verified width-importance invocation is the eight-GPU template
from section 2 with `STAGE` replaced by `width_importance`. The depth command is
identical with `depth_importance`. Do not use the sorted teacher as the depth
source: `depth_importance.source_checkpoint_dir` must resolve to `teacher_dir`.

## 5. Report and artifact gate after every stage

A successful process exit alone is not completion. Regenerate the report on the
compute node, assert that the stage is completed, and assert at least one
stage-defining artifact. For smoke width importance:

```bash
srun --jobid="$JOBID" --overlap --nodes=1 --ntasks=1 \
  --container-image="$IMAGE" --container-mounts=/lustre:/lustre \
  --container-workdir="$ROOT" /bin/bash -lc '
    set -Eeuo pipefail
    source /lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_llm/users/ssameni/setup-envs.sh
    source ./.venv_new/bin/activate
    export PYTHONPATH="$PWD:${PYTHONPATH:-}"
    export REPORT_RUN_ROOT="$PWD/puzzle_runs/nemotron3-nano-30b-a3b-puzzletron-v1/smoke"
    puzzle_runs/nemotron3-nano-30b-a3b-puzzletron-v1/commands/report_and_verify.sh \
      --label "smoke width importance" \
      --expect-stage width_importance=completed \
      --expect-artifact pruning/pruning_scores
  '
```

Use the same gate with the following defining artifacts:

| Stage | Artifact relative to the run root |
|---|---|
| `convert` | `ckpts/teacher/config.json` |
| `tokenize_data` | `dataset_cache` |
| `width_importance` | `pruning/pruning_scores` |
| `depth_importance` | `depth/iterative/trajectory.json` |
| `sort` | `ckpts/sorted_teacher/config.json` |
| `sort_sanity` | `artifacts/sort_sanity/summary.json` |
| `width_sanity` | `artifacts/width_sanity/summary.json` |
| `slicing_sanity` | `artifacts/slicing_sanity/summary.json` |
| `bypass_sanity` | `artifacts/bypass_sanity/summary.json` |
| `bypass` | `artifacts/bypass/local_kd_loss_history.json` |
| `build_library` | `candidate_library.json` |
| `replacement_scoring` | `artifacts/replacement_scoring/summary.json` |
| `vllm_stats` | `subblock_stats.json` (957 verified records) |
| `mip` | `mip/profiles` |

Copy the verification output into the campaign `evidence/` directory and add a
row to `TASKS.md`. Inspect the report rather than merely checking that the HTML
file exists.

## 6. Failed-step cleanup and resume rule

When a smoke stage fails, retain its log under `evidence/`, diagnose the first
rank failure, and remove only that stage's incomplete artifact before retrying.
For a failed sort, the exact cleanup target is:

```bash
rm -rf puzzle_runs/nemotron3-nano-30b-a3b-puzzletron-v1/smoke/ckpts/sorted_teacher
```

Never delete a completed parent stage. After the corrected retry starts and
shows stable model loading and forward progress, stop the short validation run,
remove its partial stage artifact, and relaunch the production-sized work. Use
resume only when the stage explicitly validates durable shards or checkpoints.

## 7. Verified production preparation and importance launches

Use the one-node relay from section 2 with `production.yaml`. These two commands
and their report gates are result-verified:

```bash
python examples/puzzletron/main.py \
  --config examples/puzzletron/configs/clean/families/nemotron3/nano_30b_a3b_bf16/production.yaml \
  --stage convert

python examples/puzzletron/main.py \
  --config examples/puzzletron/configs/clean/families/nemotron3/nano_30b_a3b_bf16/production.yaml \
  --stage tokenize_data
```

The resulting immutable caches are:

- `dataset_cache/train_8192x8192.tokens` with exactly 8192 samples;
- `dataset_cache/validation_128x8192.tokens` with exactly 128 samples.

Production depth importance runs directly from `teacher_dir`; sorting is not a
dependency. The following one-node PP2/EP4 launch is forward-progress-verified
on the full validation cache and writes resumable candidate JSON files:

```bash
python examples/puzzletron/main.py \
  --config examples/puzzletron/configs/clean/families/nemotron3/nano_30b_a3b_bf16/production.yaml \
  --stage depth_importance
```

Width importance is a single coordinated scoring experiment. A one-node launch
uses the same PP2/EP4 command shape with `--stage width_importance`. If it is
scaled beyond one node, add data-parallel replicas of that complete eight-GPU
model instance; never split axes into unrelated importance runs. Depth differs:
independent model instances may score disjoint no-op candidates, followed by one
validated ranking merge per removal iteration.

For the production eight-node run, use the dedicated DP8 recipe and launcher:

```bash
puzzle_runs/nemotron3-nano-30b-a3b-puzzletron-v1/commands/run_width_dp8_batch16.sh
```

The script issues `srun` directly on `batch`, sources `setup-envs.sh` before the
virtual environment on every compute-node task, and launches eight local ranks
per node. Its effective HSDP mesh is `(PP, DP_REPLICATE, DP_SHARD, CP, TP) =
(2, 8, 4, 1, 1)`; EP4 overlays the four-rank shard axis, leaving logical DP8.
The global microbatch is 16,
so every logical DP lane receives two 8192-token samples and the iterative hooks
perform 512 updates: `8192 / 16 = 512`. AutoModel overlays EP on its configured
DP shard mesh, so PP pipeline batch alignment must divide the configured DP size
by EP size before deriving the per-lane microbatch. The focused regression test
for that invariant is:

```bash
pytest -q \
  tests/unit/torch/puzzletron/test_automodel_config.py::test_pipeline_batch_alignment_splits_global_batch_by_dp_after_ep_overlay
```

This production width run is result-verified. Slurm job `14037066` completed
all 512 updates. The artifact verifier found 150 module records, 6,315 finite
score tensors, and 16,822,360 score elements across the six configured passes.
Each pass has two disjoint PP-local rank files; attention, Mamba, and MoE layer
partitions together cover all 52 layers. The retained evidence is
`evidence/verify_production_width_importance.txt`.

`--worker-stage` performs distributed worker execution but deliberately does not
write the canonical content-addressed resume marker. After the distributed
artifact is complete, run the canonical stage once on a one-node PP2/EP4 model;
it detects all completed scoring passes without recomputing them and writes
`manifests/completions/width_importance.json`:

```bash
torchrun --standalone --nproc-per-node=8 \
  examples/puzzletron/main.py \
  --config examples/puzzletron/configs/clean/families/nemotron3/nano_30b_a3b_bf16/production.yaml \
  --stage width_importance
```

Production sorting then uses one eight-GPU model instance and the same PP2/EP4
topology:

```bash
torchrun --standalone --nproc-per-node=8 \
  examples/puzzletron/main.py \
  --config examples/puzzletron/configs/clean/families/nemotron3/nano_30b_a3b_bf16/production.yaml \
  --stage sort
```

All 13 sorted checkpoint shards completed. If a distributed sort was launched
with `--worker-stage`, repeat the canonical `--stage sort` command after the
durable checkpoint exists; it reuses the checkpoint and writes
`manifests/completions/sort.json`. The combined width/sort report gate is saved
in `evidence/report_production_width_and_sort.txt` and verified a 96,828-byte
cumulative report, both completion markers, the activation-pass manifest, and
the sorted checkpoint config.

Run the full production equivalence gate with both the activation order and its
reverse control enabled:

```bash
puzzle_runs/nemotron3-nano-30b-a3b-puzzletron-v1/commands/run_sort_sanity_production.sh
```

This command uses one interactive eight-GPU PP2/EP4 instance and the production
config's `include_reverse: true`, `eval_samples: 128`, and `block_size: 8192`.
The verified results are:

| checkpoint | LM loss | delta vs teacher |
|---|---:|---:|
| teacher | 0.90014515 | 0 |
| activation-sorted | 0.90017814 | +3.29893e-05 |
| reverse-sorted | 0.90001506 | -1.30090e-04 |

Both directions passed the absolute `1e-3` gate. The defining artifact is
`artifacts/sort_sanity/summary.json`; the cumulative report gate is
`evidence/report_production_sort_sanity.txt` and verified a 101,392-byte report.

Production depth uses persistent RPC evaluation so independent candidates are
evaluated concurrently without reloading a model for every request. The exact
eight-instance batch launcher is:

```bash
puzzle_runs/nemotron3-nano-30b-a3b-puzzletron-v1/commands/run_depth_rpc_pool.sh
```

It launches eight one-node PP2/EP4 model instances. Each request evaluates one
cumulative removal set, candidate results are cached by scenario identity, and
the coordinator performs one validated ranking merge before advancing to the
next removal. This is distinct from width importance: depth is multiple model
instances scoring different candidates, while multi-node width is one
coordinated importance experiment using DP replicas.

The coupled-sort characterization tests used for this model are:

```bash
pytest \
  tests/unit/torch/puzzletron/test_sorted_teacher.py::test_sort_state_dict_applies_each_expert_channel_order_before_expert_order \
  tests/unit/torch/puzzletron/test_sorted_teacher.py::test_sort_state_dict_composes_mamba_channels_with_groupwise_head_order \
  -q
```

They verify that each original Nano expert receives its own channel permutation
before expert identities/router rows are reordered. They also verify that Mamba
heads stay within the model's eight groups and that per-head channel order is
composed with the group-preserving head permutation.

## 8. Production experiment contract

The production importance run uses 8192 samples of length 8192 and evaluation
uses 128 samples of length 8192. Bypass sanity remains 128 steps per mode. The
accepted production bypass budget is 268,435,456 tokens, with microbatch one,
GA4, 32,768 tokens per optimizer update, and 8,192 updates. Global KD uses
128 steps at about 0.5M tokens per step; overfit sanity uses the small batch
setting of 16. Width/slicing diagnosis is exactly three
layers times two target configurations for **each pruning axis**, not six cases
shared across axes. Depth importance enumerates zero through five removed
sublayers.

For the ninth `hidden_width` diagnosis axis, the two requested ratios are
exactly `7/8` and `3/4` of the 2688-wide teacher. Physical slicing requires the
campaign's 128-channel alignment, so the realized diagnostic widths are 2304
and 1920. Do not substitute the MIP search widths for these two sanity targets.

Nano's merged experts require an EP mesh even when a diagnostic model uses less
than one node. Each axis worker therefore uses four GPUs with PP2/EP2 and
DP_SHARD=2 (EP overlays that shard dimension, leaving logical DP1). Two workers
fit on each eight-GPU node. The width-diagnostic worker injects this stage-local
mesh and uses the grouped launchers:

```text
width_sanity.automodel.parallel = {tp: 1, cp: 1, pp: 2, ep: 2, dp_shard: 2, dp_replicate: 1}
puzzle_runs/nemotron3-nano-30b-a3b-puzzletron-v1/commands/run_axis_diagnostic_group.sh
puzzle_runs/nemotron3-nano-30b-a3b-puzzletron-v1/commands/run_axis_diagnostics_batch.sh
```

The verified campaign launch reuses one interactive node for axes 0-1 and maps
axes 2-8 across four batch nodes; each full node runs two workers except the
single hidden-width worker. It preserves three layers x two targets independently
for every sortable width axis. `moe_top_k` is a discrete routing choice, not a
sortable width: retain it for replacement scoring and MIP, but exclude it from
width importance, width sanity, and slicing sanity. Finalize after the eight
applicable worker manifests succeed; the finalizer writes both `artifacts/width_sanity`
and `artifacts/slicing_sanity`, which must each pass the report gate.

Nano width equivalence needs two details that are easy to miss. For the native
no-cache Mamba path, zero-masking the teacher-sized fused kernel is not
numerically equivalent to physical pruning in BF16. Intercept the target
mixer's fused call and compact its projected `[gate, x, B, C, dt]` tensor,
convolution weight/bias, `A`, `D`, `dt_bias`, RMSNorm weight, and output
projection columns with the same group-preserving indices used by
`slice_mamba2_state_dict`; pass the target head dimension and retain the
teacher epsilon. This produces bitwise-equal fused outputs for both head and
head-dimension reductions. The older epsilon/output compensation remains only
for static-shape fallback implementations that cannot call the compact fused
kernel. For routed experts, apply the correction bias and native group
selection to the compact kept-expert vector; only then map compact ids back to
the resident sorted teacher. Masking a 128-expert vector after grouping is not
equivalent to physically constructing, for example, a 112-expert gate because
the per-group width changes.

The focused regression and exact targeted reruns are:

```bash
pytest -o addopts= \
  tests/unit/torch/puzzletron/test_automodel_solution_scoring.py \
  tests/unit/torch/puzzletron/test_automodel_mamba_hook.py \
  tests/unit/torch/puzzletron/test_materialize.py -q
python puzzle_runs/nemotron3-nano-30b-a3b-puzzletron-v1/commands/diagnose_mamba_runtime_physical.py
sbatch puzzle_runs/nemotron3-nano-30b-a3b-puzzletron-v1/commands/rerun_width_mamba_axes.sbatch
sbatch puzzle_runs/nemotron3-nano-30b-a3b-puzzletron-v1/commands/rerun_width_moe_experts.sbatch
```

Before rerunning, quarantine the failed
`artifacts/activation_diagnostic_axis_{mamba_heads,mamba_head_dim,moe_experts}`
directories. Do not remove any successful
`.axis_workers/*/manifests/width_sanity.json`.

The first vLLM attempt was invalid because its editable extension was linked
against a different CUDA runtime. The clean `.venv_new` retry above has both
native extensions built against PyTorch 2.11/CUDA 12.9 and passed the focused
real-model fused-MoE probe. Its first full artifact was still invalid because
the Nemotron-H runtime descriptor silently capped routed experts, expert
widths, shared-expert widths, and top-k. Delete that artifact and its runtime
cache; production measurements now use exact physical dimensions by default.
The `runtime-075` profile may use measured statistics only after the exact
rerun passes its report gate. Every enabled profile must contain exactly
`(5 + 1) * 3 = 18` solutions: six depth-removal choices crossed with three
hidden widths. Evaluate every solution with LM loss. Select the top three
independently within each profile; profile identity is retained even if two
profiles select the same structural model.

### Future homogeneous and restricted-search comparisons

The following two comparison modes are specified for reproducibility but are
not implemented or enabled in this campaign yet. They diagnose the value of
per-layer mix-and-match decisions; they do not replace the three production
MIP profiles.

**Homogeneous Cartesian search.** For each constraint profile and each fixed
embedding width, enumerate the Cartesian product of the legal values for every
structural axis. One tuple is a homogeneous candidate: its selected attention,
Mamba, routed-expert, shared-expert, expert-count, and routing values are
applied uniformly to every compatible active sublayer. No-op placeholders are
never changed, and values for one layer family do not apply to another family.
Hold depth at the teacher value for this comparison. For each candidate:

1. calculate exact parameters and, when available, measured runtime and memory;
2. discard the candidate if it violates the active constraint;
3. sum the same layer-local importance terms used by MIP for the uniform
   choices; and
4. retain the feasible candidate with the minimum summed score for that
   embedding width and profile.

Record the full tuple, constraint measurements, summed score, and deterministic
tie-break fields. Report an explicit `infeasible` result when no homogeneous
tuple meets the constraint; do not relax the constraint. To isolate the value
of heterogeneous layer choices, compare this result with a matched MIP run
that uses the same fixed embedding width and teacher depth.

**Single-axis restricted MIP.** Rerun the 75% parameter search once per MoE
axis, with teacher embedding width, teacher depth, and every non-target axis
fixed at its teacher value. Only the named axis may vary independently across
eligible layers. The intended comparisons are:

- routed-expert count only;
- shared-expert intermediate width only;
- routed-expert intermediate width only, realized per expert for Nano's
  non-latent MoE; and
- latent projection width only on architectures that actually have latent
  MoE.

Nano has `moe_latent_size: null`, so the latent-only run is `not_applicable`,
not an infeasible optimization. The other three runs must report either the
minimum-score feasible solution or `infeasible` under the unchanged 75%
parameter constraint. Compare them with a matched mix-and-match MIP in which
all three Nano MoE axes may vary while embedding width and depth remain fixed.
This matched comparison—not the unrestricted production MIP—is what measures
the benefit of combining axes and choosing different values per layer.

The clean N=2 probe completed all ten layouts but its attention prefill slope
was noise-dominated and negative. Reproduce the one-variable N=4/N=8 gate with:

```bash
sbatch puzzle_runs/nemotron3-nano-30b-a3b-puzzletron-v1/commands/run_vllm_attention_n4_probe.sbatch
```

It uses TP2/PP2 on GPUs exposed by Slurm, the persistent CUDA 12.9 squashfs,
`.venv_new`, N=4 versus 2N=8 candidate layouts, three warmups, ten measured
iterations, prefill 256, output 32, and batch/concurrency one. The result is
written to
`smoke_vllm_new_probe_n4_attention/attention_n4_result.json`. A non-positive
prefill marginal is a failed numerical gate; debug the prefill harness instead
of disabling validation. For reference, the initial three-iteration N=4 run
completed both layouts and yielded `-0.024242 ms`, much closer to zero than the
N=2 result but still correctly rejected.

The ten-iteration gate passed with prefill `0.146240 ms`, decode `2.968960 ms`,
total `3.115200 ms`, and decode/token `0.095773 ms` per attention layer. The
verified result is
`smoke_vllm_new_probe_n4_attention/attention_n4_result.json`. Launch the full
Cartesian production sweep with:

```bash
sbatch puzzle_runs/nemotron3-nano-30b-a3b-puzzletron-v1/commands/run_vllm_stats_production_new_env.sbatch
```

This requests four batch nodes and uses eight independent one-GPU workers on
each node, with explicit disjoint `CUDA_VISIBLE_DEVICES` groups. It measures
N=4 versus 2N=8 at ISL 8192, OSL 1024, and max concurrency 4. Sparse sampling
is disabled: the intended library is exactly 8 attention choices + 9 Mamba
choices + 300 MoE Cartesian choices = 317 structural candidate options. The
runtime log reports 319 unique subblock objects because it also retains the
attention and FFN companion no-ops at zero cost. For each hidden-size anchor
(`2688`, `2560`, and `2432`), it benchmarks 636 layouts: 634 candidate N/2N
layouts plus one N/2N attention-scaffold control pair used to subtract the
required scaffold overhead from cacheless Mamba/MoE candidates. The complete
runtime table is therefore 317 candidates evaluated through 636 layouts per
anchor, or 1,908 layout-anchor measurements—not 951 independent candidate
choices. The 32 Slurm tasks are also the 32 runtime shards; each receives 19 or
20 layouts per anchor and owns exactly one GPU. The launcher asserts both one
visible GPU per task and exactly 317 distinct active candidates before each
shard starts. Runtime caches are immutable, so resubmitting the same launcher
resumes only missing layouts or hidden anchors after the cluster's four-hour
limit.

Every worker sources `setup-envs.sh` first, then gives compiler-generated state
its own local directory under
`/tmp/puzzletron-vllm-${SLURM_JOB_ID}-${SLURM_PROCID}`. This isolates Triton,
TileLang, vLLM, FlashInfer, TorchInductor, CUDA, PyTorch-kernel, temporary, and
XDG caches while leaving the shared Hugging Face model cache from
`setup-envs.sh` intact. Per-task isolation is required for the 32-worker sweep:
sharing compiler caches caused simultaneous missing-file races during kernel
generation even though the model configurations themselves were valid.

Nemotron-H's synthetic config derives `hybrid_override_pattern` from the exact
layout passed to vLLM. Attention candidates use `****` versus `********`.
Cacheless families retain one fixed attention cache anchor, so Mamba uses
`*MMMM` versus `*MMMMMMMM` and MoE uses `*EEEE` versus `*EEEEEEEE`. The anchor,
embedding, final norm, and LM head appear identically on both sides and cancel
in `(latency_2N - latency_N) / N`; embeddings and the LM head do not themselves
replace an attention KV-cache layer. Do not remove the anchor without a
separate cache-planner validation.

The rejected proxy run completed the 2688 and 2560 anchors first and later used
endpoint-only repairs for noisy 2432 prefill slopes. Those artifacts are
historical evidence only and must not be merged into the exact-dimension run.
If the exact run exposes a numerical outlier, remeasure only its contributing
N/2N endpoints under the new exact cache identity. The old repair command was:

```bash
sbatch puzzle_runs/nemotron3-nano-30b-a3b-puzzletron-v1/commands/remeasure_vllm_2432_outlier.sbatch
```

Do not execute that old command unchanged: it targets the rejected proxy cache.
An exact-run repair must archive only the affected exact cache entries and then
rebuild the 32-shard aggregate. The final report gate must verify the stable
model identity and all 317 active candidates at each of widths 2688, 2560, and
2432; companion no-op sentinels are zero-cost internals, not Nano FFN results.

The cumulative report reads the canonical root `subblock_stats.json` directly;
it does not require `scenarios/width-*/depth-00/replacement_library.json` to
exist first. Stage-driven report refreshes resolve `display_name`, then
`model_info.hf_repo`, before falling back to a local model source path, so an
immutable Hugging Face snapshot path cannot replace the human model title.

When several one-GPU vLLM engines share a node, each subprocess must receive a
fresh `VLLM_PORT`; clearing only torch-elastic `MASTER_PORT` is insufficient
because vLLM otherwise starts every independent MP engine at its own default
port. The runtime harness assigns this ephemeral base port outside the cache
identity, so a rendezvous retry never invalidates a completed latency record.

After the eight sortable width-axis manifests are successful, run the bypass
sanity gate with:

```bash
width_finalize=$(sbatch --parsable \
  --dependency=afterok:<MAMBA_JOB>:<ROUTED_EXPERT_JOB> \
  puzzle_runs/nemotron3-nano-30b-a3b-puzzletron-v1/commands/finalize_width_sanity.sbatch)
bypass_sanity=$(sbatch --parsable \
  --dependency=afterok:${width_finalize} \
  puzzle_runs/nemotron3-nano-30b-a3b-puzzletron-v1/commands/run_bypass_sanity_production.sbatch)
```

Every remaining compute task sources `setup-envs.sh` before activating
`./.venv_new` in the CUDA 12.9 image. The
finalizer aggregates exactly the eight sortable axes and report-verifies both
`width_sanity` and `slicing_sanity`. Bypass sanity uses one PP2/EP4 model
instance for 128 fixed-smallest steps and 128 diverse-resampled steps. Its
explicit verifier requires contiguous finite histories, at least 5% loss
reduction for the comparable fixed structure, and multiple sampled structures
for the diverse mode.

The first 128-step production run was mechanically valid but rejected because
its loss did not improve. Preserve its small diagnostics under
`evidence/rejected_bypass_14056809`, remove its checkpoint and canonical bypass
artifacts. The attempted 2 Gi-token DP4 replacement was also stopped: although
all 32 GPUs were 95--100% active, cross-node synchronization reduced aggregate
throughput to about 7k tokens/s, below the previous one-node run's 17.6k
tokens/s. Reproduce the approved middle-budget rerun with:

```bash
first=$(sbatch --parsable \
  puzzle_runs/nemotron3-nano-30b-a3b-puzzletron-v1/commands/run_bypass_256m_1node.sbatch)
second=$(sbatch --parsable --dependency=afterany:${first} \
  puzzle_runs/nemotron3-nano-30b-a3b-puzzletron-v1/commands/run_bypass_256m_1node.sbatch)
third=$(sbatch --parsable --dependency=afterany:${second} \
  puzzle_runs/nemotron3-nano-30b-a3b-puzzletron-v1/commands/run_bypass_256m_1node.sbatch)
```

The already verified `train_262144x8192.tokens` cache is reused; the training
budget consumes 32,768 sequences from it. The one-node mesh is PP2/EP4 with
logical DP1. Microbatch one and GA4 give `1 x 8192 x 4 = 32768` tokens per
optimizer update and exactly 8,192 updates. This trades some throughput for
more update opportunities without paying the inefficient four-node reduction.
`bypass_256m.yaml` overrides the production default with AdamW learning rate
`1.0e-4`; do not inherit the earlier `1.0e-6` value. The production launch
verified four backward passes before every optimizer sync, ~2.1 seconds/update,
95--97% GPU utilization, and 59.6--67.6 GiB used per 81.6-GiB GPU. Its measured
training ETA is about 4.8 hours before resume and final-checkpoint overhead.

Each slot writes a validated distributed checkpoint after 12,000 seconds,
leaving time for publication before the four-hour limit, and
`find_last_ckpt_for_resume=true` restores counters, optimizer, scheduler, RNG,
and sampler state. `afterany` is allowed only because each continuation rejects
an incomplete checkpoint transaction. A continuation exits quickly if the
staging completion marker is already valid. The successful slot regenerates
and parses the cumulative report and requires
`artifacts/bypass/local_kd_loss_history.json`.

For a fresh production bypass, `bypass.step_num`, `bypass.iter_num`, and
`bypass.token_count` are runtime resume counters and must begin at `1`, `0`,
and `0`, respectively. Do not initialize them to the target totals. When
resuming a compatible production checkpoint, Puzzletron restores the counters
from that checkpoint's `args.json`.

Evaluate every mixed and homogeneous MIP recipe online without realizing a
checkpoint. Rank `runtime-075` by zero-shot LM loss, then materialize only its
best candidate. AIPerf compares that candidate with the teacher as a serving sanity check using ISL
8192, OSL 1024, concurrencies `1, 4, 16, 64`, all eight GPUs, at least three
parallel settings, and EP greater than one. The exact topology matrix is in
`production.yaml`. The absolute KD winner is the same lowest-LM-loss candidate;
the teacher is excluded. Run its 16-sample overfit and, if healthy, its
128-step full KD followed by the same 128-sample LM-loss evaluation. No
downstream task suite is required.

The exact sharding/launch commands for production replacement scoring, the
six MIP profiles, AIPerf, and the selected KD run will be added here after their
smoke and production gates pass. This prevents an untested launcher from being
presented as an exact-reproduction command.

## 9. Known non-blocking messages

The current environment may print duplicate torchao/CUTLASS/vLLM registration
messages and Transformer Engine/Apex fallback warnings. They are known for this
image and virtual environment and are not failures. CUDA OOM, a missing artifact,
a failed report assertion, a rank exception, or a numerical sanity violation is
a real failure and must be diagnosed before continuing.
