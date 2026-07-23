---
name: qad
description: >-
  Recover a measured BF16-to-PTQ accuracy gap on Slurm with ModelOpt
  Quantization-Aware Distillation (QAD) through Megatron Bridge. Use when PTQ
  has already been benchmarked against the matching BF16 model and the
  quantized checkpoint needs short, evidence-driven distillation; also use for
  QAD topology, data preparation, Slurm launch, resume, checkpoint export, or
  recovery decisions.
---

# ModelOpt Quantization-Aware Distillation

Use QAD only to recover a material, apples-to-apples PTQ benchmark gap. Do not
train merely because a quantized checkpoint exists.

## 1. Read the supported workflow

Read these sources before constructing commands:

- `examples/megatron_bridge/README.md`, especially Post-Training Quantization,
  Data Preparation, Quantization Aware Distillation, export, and Slurm usage
- `examples/dataset/MEGATRON_DATA_PREP.md`, especially token-budgeted blends
- `examples/megatron_bridge/{quantize.py,distill.py}` via `--help`

Treat them as the source of truth for mutable flags, containers, model support,
and checkpoint formats. Do not copy their full command lines into new wrappers.
The deprecated `examples/llm_qad` flow is not the default.

## 2. Establish the gap

When QAD follows `evaluation` or `quant-recipe-search`, reuse its complete PTQ
config or recipe to reproduce PTQ with `examples/megatron_bridge/quantize.py`;
do not use the usual HF checkpoint as the QAD student. Preserve the quantization
format, layer selection, calibration dataset, sample count, sequence length,
seed, and topology; do not substitute a cheaper or otherwise different PTQ run.

Use the `evaluation` and `compare-results` skills to run and validate the same
benchmark configuration on the BF16 source and the export of the exact
quantized Megatron checkpoint that will initialize QAD. If the existing PTQ
result came from another checkpoint or its lineage is unclear, export and
benchmark the intended Megatron checkpoint first. Record its path or identifier,
metric, direction, uncertainty, and acceptable delta. Stop if the gap is not
material or the runs are not comparable.

## 3. Set up Slurm execution

Read `skills/common/environment-setup.md` and detect whether the Slurm target is
local or remote. This version of the skill supports Slurm only; stop if the
target is not Slurm.

- On local Slurm, follow `skills/common/workspace-management.md` and
  `skills/common/slurm-setup.md` directly. Do not add SSH or sync steps.
- On remote Slurm, also follow `skills/common/remote-execution.md`: establish
  its persistent SSH session, create matching local and remote session/model
  workspaces, and sync the source plus the two-script Slurm wrapper/inner runner.

Use the common account, partition, registry-auth, submission, and monitoring
procedures in either case.

Keep BF16, the exact benchmarked PTQ Megatron checkpoint and its export, QAD
Megatron, exported checkpoints, data, logs, and benchmark results in the same
session/model workspace. QAD must start from the checkpoint produced by
`examples/megatron_bridge/quantize.py` because it carries the ModelOpt state; an
exported HF PTQ checkpoint is not a substitute.

## 4. Choose topology explicitly

Inspect model parameter count and architecture, sequence length, GPU count/type
and memory, then choose the smallest topology that fits both student and teacher:

1. Choose node count from model-state plus activation memory; do not start with
   an oversized multi-node topology.
2. Use TP only when model-state/GEMM size warrants it, and preserve attention
   head and hidden-size divisibility.
3. Use CP to distribute the 32K context when activation/attention memory is the
   constraint. Prefer CP before TP for a small dense model at long context.
4. Keep EP=1 for dense models. For MoE, choose EP from expert count, memory, and
   available ranks; verify expert and data-parallel divisibility.
5. Add PP only when layer/model state still does not fit.

Verify `DP = world_size / (TP * PP * CP)` is integral and
`GBS % (MBS * DP) == 0`. Record nodes, GPUs/node, TP/PP/CP/EP/DP, MBS, GPU
memory, and why each non-one dimension is needed.

Example only: for the requested one-node, eight-H100 Qwen3-0.6B validation at
32K, TP=1, PP=1, CP=4, EP=1, DP=2, MBS=1 is a reasonable starting point. This
is validation evidence, not a default topology; derive a fresh topology for
every target model and cluster.

## 5. Prepare only the next data tranche

Copy `assets/nemotron-cascade-2-blend.yaml`, then set its tokenizer, output
directory, and token budget. Run:

```bash
python -m modelopt.torch.utils.plugins.prepare_megatron_data_blend \
  --config <copied-blend.yaml>
```

The utility streams `nvidia/Nemotron-Cascade-2-SFT-Data`, applies a deterministic
approximate buffer shuffle, and stops at the token budget. Never snapshot or
materialize the full dataset. For the initial step-150 gate, the default
2.6-billion-token budget covers
`150 * 512 * 32768` tokens plus a small margin. Use a much smaller explicit
budget for workflow validation. At very small budgets, use a representative
subset of source configs rather than giving a rare config too few documents
for Megatron's train/validation split.

Pass the generated `data_blend.txt` entries as `distill.py --data_paths`.
The supported real-data `GPTDatasetConfig` shuffles documents and concatenates
them into fixed-length 32K samples, so sequences are packed instead of padded.
Do not use mock data as training evidence.

## 6. Retain PTQ lineage, then run staged QAD

Use the exact PTQ Megatron checkpoint established in step 2, or produce and
benchmark it before QAD. Follow the current Megatron Bridge README for PTQ, QAD,
resume, and export commands. Apply these QAD defaults:

| Setting | Default |
| --- | --- |
| Sequence length | 32768 |
| Peak / minimum LR | `1e-5` / `1e-6` |
| LR schedule | cosine |
| Training cap | 1000 iterations |
| Global batch size | 512 |
| Dataset | `nvidia/Nemotron-Cascade-2-SFT-Data` |
| Initial eval/save/exit | iteration 150 |
| Slurm duration exit | 220 minutes for a 4-hour allocation |

Keep `train_iters=1000` from the first run so the cosine schedule has a stable
horizon. Set `eval_interval=150`, `exit_interval=150`, and
`exit_duration_in_mins=220` for the initial stage. The duration exit is a
checkpointing safety margin, not the training target.

Use one result-bearing Slurm job per stage and put Pyxis container flags on the
final `srun`. Fold startup validation into that job; do not submit separate GPU
preflight or smoke jobs.

## 7. Benchmark before continuing

At the initial exit:

1. Confirm finite QAD/KD loss, sensible learning rate, non-pathological gradient
   norm, and an iteration-150 checkpoint.
2. Export that QAD checkpoint with the current quantized Megatron exporter.
3. Run the identical benchmark on BF16, the step-2 PTQ export, and QAD-150.
4. Calculate the original gap and recovery. For higher-is-better metrics:
   `gap = BF16 - PTQ`, `recovered = QAD - PTQ`, and
   `recovery_fraction = recovered / gap`.

Continue only if benchmark recovery is positive beyond run noise and loss is
stable. Choose the next absolute checkpoint from the evidence (for example 300,
500, 750, or 1000), resume the same output directory with `train_iters=1000`,
and set both `eval_interval` and `exit_interval` to that target. Stop on
recovered gap, plateau, regression, divergence, or iteration 1000. Never raise
the 1000-step cap without explicit user direction.

## 8. Report evidence

Report exact source revision, commands, Slurm job/account/partition, container,
paths, topology derivation, dataset configs/seed/token budget/materialized
counts, PTQ format, checkpoint iterations, loss/LR/grad trend, scheduler state,
and BF16/PTQ/QAD benchmark scores. Support success with both scheduler and log
evidence; identify the first real error when a run fails.
