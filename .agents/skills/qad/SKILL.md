---
name: qad
description: >-
  Run explicitly requested ModelOpt Quantization-Aware Distillation (QAD) on
  Slurm through Megatron Bridge to recover a measured BF16-to-PTQ accuracy gap.
  Use only when the user explicitly asks for QAD, including its topology, data
  preparation, Slurm launch, resume, checkpoint export, or recovery decisions.
---

# ModelOpt Quantization-Aware Distillation

QAD is expensive. Use it only to recover a material, apples-to-apples PTQ
benchmark gap and only when the user explicitly requests QAD for the target
model or run. Do not infer permission from a quantized checkpoint, evaluation
gap, or recipe-search result.

## 1. Read the supported workflow

Read these sources before constructing commands:

- `examples/megatron_bridge/README.md`, especially Post-Training Quantization,
  Data Preparation, Quantization Aware Distillation, export, and Slurm usage
- `examples/dataset/MEGATRON_DATA_PREP.md`, especially token-budgeted blends
- `examples/megatron_bridge/{quantize.py,distill.py}` via `--help`

Treat them as the source of truth for mutable flags, containers, model support,
and checkpoint formats. Do not copy their full command lines into new wrappers.
The deprecated `examples/llm_qad` flow is not the default.

## 2. Reuse or establish the gap

When QAD follows `evaluation`, `compare-results`, or `quant-recipe-search`,
reuse the validated BF16/PTQ scores and exact benchmark configuration; do not
rerun valid baselines. Record their run IDs, metric, direction, uncertainty, and
acceptable delta. Use `evaluation` and `compare-results` only for missing,
unvalidated, or non-comparable results. Stop if the gap is not material.

Reuse the preceding PTQ config or recipe to reproduce PTQ with
`examples/megatron_bridge/quantize.py`; do not use the usual HF checkpoint as
the QAD student. Preserve the quantization format, layer selection, calibration
dataset, sample count, sequence length, and seed. Choose Megatron execution
topology separately without changing quantization semantics. If any quantization
setting must change, treat it as a new PTQ candidate and evaluate it before QAD.

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

Keep the reused BF16/PTQ run references and configs with the reproduced PTQ
Megatron checkpoint, QAD checkpoints, data, logs, exports, and new benchmark
results in the same session/model workspace. QAD must start from the checkpoint
produced by `examples/megatron_bridge/quantize.py` because it carries the
ModelOpt state; an exported HF PTQ checkpoint is not a substitute.

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
   available ranks, and inspect the current `distill.py` value for expert tensor
   parallelism (ETP); do not assume ETP equals TP.
5. Add PP only when layer/model state still does not fit.

For MoE, Megatron folds two overlapping meshes onto the same ranks within each
PP stage; EP/ETP do not multiply the dense TP/CP mesh:

- Attention DP: `DP = world_size / (TP * PP * CP)`
- Expert DP: `EDP = world_size / (ETP * EP * PP)`

Require both divisions to be integral, `num_experts % EP == 0`, and
`GBS % (MBS * DP) == 0`. Record nodes, GPUs/node, TP/PP/CP/EP/ETP/DP/EDP, MBS,
GPU memory, and why each non-one dimension is needed.

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
It deterministically holds out 1% of those shuffled documents for validation
with `split="99,1,0"`; bounded materialization uses shuffle seed 42, while the
training seed controls ordering within the Megatron splits. Do not download a
duplicate validation copy. With the defaults, two GBS-512 32K validation
batches consume 33.6M tokens from the nominal 26M-token holdout, so about 1.3
passes per validation event is accepted. Recalculate this ratio when changing
GBS, sequence length, or `eval_iters`, and increase the token budget rather than
allowing substantially more repetition. Treat this loss as a training-health
signal; the independent benchmark remains the recovery gate. Do not use mock
data as training evidence.

## 6. Reproduce PTQ, then run staged QAD

Produce the required Megatron checkpoint with the PTQ config or recipe preserved
in step 2. Reuse a valid preceding PTQ evaluation; run a new PTQ evaluation only
when the baseline is missing, invalid, or non-comparable, or a quantization
setting changed. Follow the current Megatron Bridge README for PTQ, QAD, resume,
and export commands. Apply these QAD defaults:

| Setting | Default |
| --- | --- |
| Sequence length | 32768 |
| Peak / minimum LR | `1e-5` / `1e-6` |
| LR schedule | cosine |
| Training cap | 1000 iterations |
| Global batch size | 512 |
| Dataset | `nvidia/Nemotron-Cascade-2-SFT-Data` |
| Validation | every 25 iterations; deterministic 1% holdout; 2 batches |
| Checkpoint interval | 50 iterations |
| Initial benchmark/exit | iteration 150 |
| Slurm duration exit | 220 minutes for a 4-hour allocation |

Keep `train_iters=1000` from the first run so the cosine schedule has a stable
horizon. Set `save_interval=50`, `eval_interval=25`, `eval_iters=2`,
`exit_interval=150`, and `exit_duration_in_mins=220` for the initial stage. This
preserves checkpoints at iterations 50, 100, and 150. The duration exit is a
checkpointing safety margin, not the training target.

Use one result-bearing Slurm job per stage and put Pyxis container flags on the
final `srun`. Fold startup validation into that job; do not submit separate GPU
preflight or smoke jobs.

## 7. Monitor loss, then benchmark

Monitor the running log instead of waiting for the stage to finish. Record total
or logits-distillation loss every log interval. The expected signal is a
decreasing smoothed trend, not a decrease at every noisy step:

1. From iteration 100 onward, compare the median loss in each 50-step window
   with the preceding window.
2. Treat non-finite loss, NaN iterations, repeated skipped iterations, or a
   sharp sustained increase as immediate failure.
3. Flag one flat or rising window. If a second consecutive window also fails to
   decrease, stop at the checkpoint ending that second window. Preserve the
   latest known-good checkpoint and diagnose before resuming.

At the initial exit:

1. Confirm the smoothed QAD/KD loss decreased, learning rate is sensible,
   gradient norm is non-pathological, and checkpoints 50, 100, and 150 exist.
2. Export that QAD checkpoint with the current quantized Megatron exporter.
3. Evaluate QAD-150 with the exact step-2 benchmark configuration. Reuse
   validated, comparable BF16/PTQ results; run only missing, invalid, or
   non-comparable baselines.
4. Calculate the original gap and recovery. For higher-is-better metrics:
   `gap = BF16 - PTQ`, `recovered = QAD - PTQ`, and
   `recovery_fraction = recovered / gap`.

Continue only if benchmark recovery is positive beyond run noise and loss is
stable. Choose the next absolute checkpoint from the evidence (for example 300,
500, 750, or 1000), resume the same output directory with `train_iters=1000`,
retain `eval_interval=25`, `eval_iters=2`, and `save_interval=50`, and set only
`exit_interval` to that target. Stop on recovered gap, plateau, regression,
divergence, or iteration 1000. Never raise the 1000-step cap without explicit
user direction.

## 8. Report evidence

Report exact source revision, commands, Slurm job/account/partition, container,
paths, topology derivation, dataset configs/seed/token budget/materialized
counts, PTQ format, reused evaluation run IDs, checkpoint iterations,
loss/LR/grad trend, scheduler state, and BF16/PTQ/QAD benchmark scores. State
which scores were reused versus newly evaluated. Support success with both
scheduler and log evidence; identify the first real error when a run fails.
