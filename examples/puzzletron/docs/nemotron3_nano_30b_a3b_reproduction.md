# Nemotron-3 Nano 30B-A3B Puzzletron Reproduction

This is the living, exact-run tutorial for pruning
`nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` at revision
`cbd3fa9f933d55ef16a84236559f4ee2a0526848`.

The canonical configuration is
`examples/puzzletron/configs/clean/families/nemotron3/nano_30b_a3b_bf16/production.yaml`.
Treat that file and the completed stage manifests under the campaign root as the source of truth;
do not reconstruct the configuration from report HTML.

## 1. Fixed environment

Run scheduler commands from the login node. Run every Python import, test, report generator, and
model command on a compute node with:

```bash
source /lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_llm/users/ssameni/setup-envs.sh
source .venv_new/bin/activate
```

Use this image and mount in every `srun`/`sbatch` task:

```text
/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_llm/users/ssameni/enroot/cuda_12p9p2_cudnn_devel_ubuntu24p04.sqsh
--container-mounts=/lustre:/lustre
```

Campaign root:

```text
puzzle_runs/nemotron3-nano-30b-a3b-puzzletron-v1
```

The train cache contains `8192 * 8192` tokens and the validation cache contains
`128 * 8192` tokens:

```text
dataset_cache/train_8192x8192.tokens
dataset_cache/validation_128x8192.tokens
```

## 2. Exact search space

- Hidden width: `2688`, `2560`, `2432`.
- Mamba head dimension: `64`, `56`, `48`.
- Mamba heads: `64`, `56`, `48`; preserve the descriptor's eight Mamba groups.
- Per-expert intermediate size: `1856`, `1600`, `1344`, `1088`, `832`.
- Shared-expert intermediate size: `3712`, `3072`, `2560`, `2048`.
- Routed experts: `128`, `112`, `96`, `80`, `64`.
- Experts per token: `6`, `4`, `2`.
- KV heads: `2`, `1`.
- Query heads per KV group: `16`, `12`, `8`, `4`.

Nano has no latent-MoE dimension. Sort channels inside each known expert first, then sort expert
identity. For Mamba, sort channels inside heads while retaining legal group structure, then sort
heads. Routing top-k is a discrete replacement/MIP choice, not a sortable width-sanity axis.

## 3. Stage order and invariants

Run stages in this order:

1. Convert the pinned HF checkpoint.
2. Materialize the fixed train and validation token caches.
3. Collect width importance once with one logical model instance; use DP replication, not multiple
   independently loaded width models.
4. Collect iterative subblock depth importance through five removals with RPC model instances.
5. Sort and run forward/reverse physical slicing sanity.
6. Run three-layer by two-target sanity for every sortable axis, plus hidden widths `7/8` and
   `3/4` of the teacher width.
7. Run bypass sanity, then the production nested-subblock bypass.
8. Build the three hidden-width replacement libraries from the sorted parent plus nested-bypass
   overlay.
9. Score every replace-one-subblock candidate.
10. Solve the three main 73--75% MIP profiles plus the three restricted 73--75%-parameter profiles, then
    evaluate every mixed and homogeneous recipe online by LM loss.
11. Materialize only the lowest-LM-loss `runtime-075` candidate. Run the AIPerf sanity matrix on
    that candidate plus the teacher. Run KD overfit and then full KD on the same candidate; the
    teacher is never eligible for KD selection.
12. Run post-KD LM-loss evaluation; downstream benchmarks are not required.

For model loading, keep each model instance within one node. The normal full-model topology is
PP=2, EP=4. Replacement scoring is verified with 32 independent EP=2/FSDP model instances across
8 nodes, one instance per GPU pair. Always pass the intended GPU pair through
`CUDA_VISIBLE_DEVICES`.

## 4. Bypass and library contract

The accepted bypass checkpoint is an elastic nested-subblock checkpoint trained over the complete
width/config search space. A replacement candidate must be evaluated as follows:

1. Load the sorted teacher for its hidden width.
2. Overlay the corresponding nested-bypass weights.
3. Replace exactly the requested semantic subblock.
4. Apply that candidate's physical config change.
5. Score 128 validation sequences of length 8192.
6. Restore the resident sorted teacher and verify restoration before taking another request.

The verified library cardinality at each hidden width is:

```text
23 Mamba layers * 9 candidates
+ 23 MoE layers * 300 candidates
+ 6 attention layers * 8 candidates
= 7155 entries
```

Each library contains 52 teacher entries and 7103 alternatives. The three width libraries are
written under `scenarios/width-{2688,2560,2432}/depth-00/`.

Prepare or validate them with:

```bash
sbatch puzzle_runs/nemotron3-nano-30b-a3b-puzzletron-v1/commands/prepare_replacement_rpc_inputs.sbatch
```

## 5. Replace-one-subblock RPC scoring

The disposable smoke covers one representative candidate per axis and uses the immutable ID files
written by input preparation:

```bash
sbatch puzzle_runs/nemotron3-nano-30b-a3b-puzzletron-v1/commands/replacement_rpc_smoke.sbatch
```

After smoke passes and its disposable artifacts are removed, launch production:

```bash
sbatch puzzle_runs/nemotron3-nano-30b-a3b-puzzletron-v1/commands/replacement_rpc_production.sbatch
```

Production uses 8 batch nodes, 8 GPUs per node, `--exclusive`, and 32 EP=2 model instances. The
partition has a hard four-hour limit, so queue continuation jobs with `afterany`; the campaign
journal makes requests and results durable and idempotent. The production script exits immediately
when `evidence/replacement_rpc_production_verification.json` already exists.

The campaign identity retains micro-batch 1 for compatibility with the initial durable journal.
Continuation workers use runtime micro-batch 4, which was verified on the same 128x8192 candidate:

- evaluator time: 74.5 s at micro-batch 1, 41.8 s at micro-batch 4;
- maximum micro-batch-4 VRAM: 75,335 MiB of 81,559 MiB;
- largest average metric difference: `3.7e-8`;
- largest per-sample metric difference: `1.6e-7`.

Runtime micro-batch size is stored in result provenance. Do not try micro-batch 8 without a new
isolated benchmark; micro-batch 4 leaves only about 6.2 GiB at the observed peak.

## 6. MIP and downstream selection

The root `mip.workloads` section defines named workload points. Profiles reference those names with
`at`, and every constraint accepts percentages or exact numbers plus list/range forms. The canonical
main profiles are `params-075`, `runtime-075`, and `memory-075`. Their inclusive retained interval
is 73--75% of the teacher; runtime and memory use the named `serving-8k` workload. Three additional
parameter-limited profiles still search all six depth choices and all three embedding widths while
holding every other width axis at the teacher and searching routed-expert count only, per-expert
intermediate width only, or both. Every profile sets `num_homogeneous_solutions: 5`, so the best
five feasible homogeneous comparisons are retained alongside the mixed winner for each requested
depth/embedding scenario. Even the narrower 73--75% band admitted about 921 unique homogeneous
architectures in a single unrestricted Nano probe, so retaining all of them would still create
thousands of unnecessary 128-by-8192-token evaluations.

Typed depth restrictions are valid only because depth importance uses subblock granularity. A
profile may constrain total removed subblocks or per-kind counts such as attention, Mamba, and MoE.
Lists are Cartesian-producted. See `examples/puzzletron/docs/mip_profiles.md` for the full profile
language, homogeneous enumeration, and restricted per-axis comparisons.

MIP runs with `skip_realize_model: true`: it writes architecture recipes, not checkpoints. Evaluate
every recipe on 128 sequences of length 8192 by loading the width-specific sorted teacher once,
overlaying accepted nested-bypass weights only on changed layers, applying all chosen subblock and
hidden-width changes in one dynamic architecture context, and restoring the resident parent after
each candidate. This is the zero-shot ranking used below; do not materialize every MIP recipe.

Rank `runtime-075` by finite zero-shot LM loss and retain only its best candidate. Materialize that
candidate for AIPerf and KD—it must not be the teacher or a candidate from another profile. Run the
16-sample overfit first. If healthy, run 128 full
KD steps using 8192 samples of length 8192 and global batch 64, which is 524,288 tokens per step.

AIPerf compares the best realized candidate with the original teacher at ISL 8192, OSL 1024,
concurrency `1`, `4`, `16`, and `64`, and at least three all-eight-GPU parallelization settings
including EP>1. This is a serving sanity check rather than a finalist tournament.

Run the exact solve-only MIP/profile/report gate with:

```bash
sbatch puzzle_runs/nemotron3-nano-30b-a3b-puzzletron-v1/commands/refresh_parameter_stats_and_run_mip.sbatch
```

The job verifies the three physical width parameter inventories, solves 18 scenarios for each of
the six profiles, prepares the deduplicated online plan, and regenerates the report. The plan and
worker manifests record one resident sorted-teacher load, nested-bypass checkpoint roles, and no
materialized per-solution checkpoints. Smoke one full 128x8192 architecture next:

```bash
sbatch puzzle_runs/nemotron3-nano-30b-a3b-puzzletron-v1/commands/run_online_eval_smoke.sbatch
```

Preserve its log as evidence, remove the disposable `artifacts/zero_shot_evaluation/online_plan/raw`
directory, then run all three widths as 24 shards with at most eight exclusive batch nodes active:

```bash
EVAL=$(sbatch --parsable puzzle_runs/nemotron3-nano-30b-a3b-puzzletron-v1/commands/run_online_eval_array.sbatch)
sbatch --dependency=afterok:${EVAL} puzzle_runs/nemotron3-nano-30b-a3b-puzzletron-v1/commands/finalize_online_eval.sbatch
```

After the merge/report gate, materialize and register the best `runtime-075` candidate:

```bash
sbatch puzzle_runs/nemotron3-nano-30b-a3b-puzzletron-v1/commands/materialize_runtime_finalists.sbatch
```

The resulting `mip/profiles/runtime-075/selected_solutions.json` contains teacher plus the best
candidate and records `absolute_best_solution_id`. AIPerf and the overfit may then run in parallel;
the full KD depends on a successful overfit:

```bash
AIPERF=$(sbatch --parsable puzzle_runs/nemotron3-nano-30b-a3b-puzzletron-v1/commands/run_runtime_aiperf.sbatch)
sbatch --dependency=afterok:${AIPERF} puzzle_runs/nemotron3-nano-30b-a3b-puzzletron-v1/commands/finalize_runtime_aiperf.sbatch
OVERFIT=$(sbatch --parsable puzzle_runs/nemotron3-nano-30b-a3b-puzzletron-v1/commands/run_runtime_kd_overfit.sbatch)
sbatch --dependency=afterok:${OVERFIT} puzzle_runs/nemotron3-nano-30b-a3b-puzzletron-v1/commands/run_runtime_global_kd.sbatch
```

## 7. Reports and verification

The stable Nano report is:

```text
puzzle_runs/nemotron3-nano-30b-a3b-puzzletron-v1/artifacts/campaign_report/campaign_report.html
```

Regenerate the Nano report, or both Nano and the exact Qwen reference report, on a compute node:

```bash
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
python examples/puzzletron/generate_campaign_progress_report.py \
  --puzzle-dir puzzle_runs/nemotron3-nano-30b-a3b-puzzletron-v1 \
  --model-name nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
python puzzle_runs/nemotron3-nano-30b-a3b-puzzletron-v1/commands/regenerate_requested_reports.py nano
python puzzle_runs/nemotron3-nano-30b-a3b-puzzletron-v1/commands/regenerate_requested_reports.py both
python puzzle_runs/nemotron3-nano-30b-a3b-puzzletron-v1/commands/verify_requested_reports.py
NODE=/home/ssameni/.cache/pre-commit/repo724x1gs2/node_env-default/bin/node
"$NODE" --max-old-space-size=6144 \
  puzzle_runs/nemotron3-nano-30b-a3b-puzzletron-v1/commands/verify_vllm_sweep_runtime.mjs \
  puzzle_runs/nemotron3-nano-30b-a3b-puzzletron-v1/artifacts/campaign_report/campaign_report.html \
  ../puzzle_runs/qwen3_5/qwen3_5_9b/sanity_check/artifacts/campaign_report/campaign_report.html
```

The first command is incremental by default. For a targeted rebuild or a diagnostic cold build,
use:

```bash
python examples/puzzletron/generate_campaign_progress_report.py \
  --puzzle-dir puzzle_runs/nemotron3-nano-30b-a3b-puzzletron-v1 \
  --model-name nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
  --rebuild-section replacement
python examples/puzzletron/generate_campaign_progress_report.py \
  --puzzle-dir puzzle_runs/nemotron3-nano-30b-a3b-puzzletron-v1 \
  --model-name nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
  --no-cache
sed -n '1,160p' \
  puzzle_runs/nemotron3-nano-30b-a3b-puzzletron-v1/artifacts/campaign_report/report_manifest.json
```

`section_cache/` and `report_manifest.json` are disposable accelerators, not campaign evidence.
They can be removed to force a correct cold reconstruction. Only `campaign_report.html` is needed
for sharing. A completed replacement-scoring snapshot is keyed by its completion receipt, summary,
and library identities, so warm downstream report updates do not reopen the 21,309 raw score files.

The bypass explorer has a sublayer selector, hidden-width selector, dynamic per-axis selectors,
visible selected-config/parameter summaries, full configuration hover text, and a fixed 0-to-1
active/teacher parameter color scale. It also averages the visible DP lanes per optimizer step and
draws a trailing 16-step moving average. Attention controls expose KV heads and query heads per KV
head, avoiding impossible cross-products between independently selected raw head counts. The
verifier checks exact observation counts, catalog
referential integrity, compact payload fields, stable model identity, and recomputed parameter
ratios. The vLLM and replacement-score explorers use the same independent per-axis selection
model: the chosen x-axis remains unsliced, every other axis with at least two real values has an
`ALL`-capable selector, and no combined configuration or sublayer-family selector remains. Partial
durable replacement scores are visible while the replacement stage remains pending.
The aggregated vLLM chart has one clickable legend entry per sublayer family while retaining
hidden-width colors and family marker shapes. MIP output is split into mixed and homogeneous
tables. Homogeneous rows show their constant per-family assignments, while numeric cost columns
show both the absolute value and percentage of the original teacher; internal presence counters,
the zero sliced-teacher baseline, and the redundant parameter-ratio column are omitted.
The standalone report parses its embedded campaign payload once and shares the resulting object
across all explorers; do not reintroduce per-section parsing for large campaign reports.
The runtime verifier must report `selectors=8, points=951` for Nano and
`selectors=1, points=16` for the Qwen reference report.
With **Connect matching configurations** enabled, it must additionally report 317 connected Nano
traces and 4 connected Qwen traces while preserving 951 and 16 total points respectively.
The overview runtime gate must report three clickable family traces for both reports. If the
container does not expose the host Node executable, add
`/home/ssameni/.cache/pre-commit:/home/ssameni/.cache/pre-commit` to `--container-mounts` for this
verification-only command.

Do not mark an artifact-producing stage complete until its manifest, numerical outputs, and the
regenerated cumulative report have all been verified.
