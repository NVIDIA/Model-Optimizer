---
name: qad
description: >-
  Run explicitly requested ModelOpt Quantization-Aware Distillation (QAD) on
  Slurm through Megatron Bridge to recover a measured BF16-to-PTQ accuracy gap.
  Use only when the user explicitly asks for QAD, including its topology, data
  preparation, Slurm launch, resume, checkpoint export, or recovery decisions.
---

# ModelOpt Quantization-Aware Distillation

QAD is expensive. Run it only when the user explicitly authorizes QAD for the
target model or run. A Day-0, PTQ, evaluation, comparison, or recipe-search
request alone is not authorization to start QAD.

## Follow the supported workflow

Before constructing commands, read:

- `examples/megatron_bridge/README.md`, especially PTQ, data preparation, QAD,
  export, and Slurm usage
- `examples/megatron_bridge/{quantize.py,distill.py}` via `--help`
- `skills/common/{environment-setup,workspace-management,slurm-setup}.md`; also
  `skills/common/remote-execution.md` for remote Slurm

Treat the example README and `--help` output as authoritative for mutable flags,
commands, containers, and checkpoint formats. This skill supports Slurm only.

## Execute in this order

1. **Confirm the gap.** Reuse only validated, comparable BF16/PTQ results and
   the exact benchmark configuration from preceding evaluation or recipe
   search; run missing, invalid, or non-comparable baselines. Set the recovery
   target to the user's acceptable BF16 delta, or the benchmark-noise envelope
   if none is supplied. Stop if the initial gap already meets that target.
2. **Reproduce PTQ and verify compatibility.** In the target runtime, require
   `AutoBridge.can_handle()` for the target model and PTQ through `quantize.py`
   to succeed while preserving the exact preceding PTQ config or recipe:
   format, layer selection, calibration data/count, sequence length, and seed.
   A changed quantization setting is a new PTQ candidate and must be evaluated
   before QAD. In the master-rank `.quant_summary.txt`, require finite positive
   `amax` for enabled static quantizers; accept `dynamic`/format-defined `None`
   only when the recipe intends it. Treat the summary as rank-local under model
   parallelism.
3. **Choose topology explicitly.** Derive the smallest fitting node count and
   TP/PP/CP/EP/ETP from student and teacher architecture, 32K activation memory,
   and available GPU memory. Prefer CP before TP for small long-context models;
   keep EP=1 for dense models. For MoE require:

   - `DP = world_size / (TP * PP * CP)`
   - `EDP = world_size / (ETP * EP * PP)`
   - integral DP/EDP, `num_experts % EP == 0`, and
     `GBS % (MBS * DP) == 0`

4. **Prepare the full capped dataset once.** Copy
   `examples/megatron_bridge/data/nemotron-cascade-2-blend.yaml`, set the target
   tokenizer and workspace path, and materialize it before training. Use the
   generated random sample for packed 32K sequences and its 1% validation
   holdout; do not download separate validation data or use mock data as
   evidence.
5. **Run staged QAD.** Use one result-bearing Slurm job per stage and fold startup
   validation into it; do not submit separate GPU preflight jobs. Use the
   defaults below and evaluate checkpoint 150 before deciding whether to
   continue.

## Default training policy

| Setting | Default |
| --- | --- |
| Sequence length | 32768 |
| Peak / minimum LR | `1e-5` / `1e-6` |
| LR schedule | cosine |
| Training cap | 1000 iterations |
| Global batch size | 512 |
| Dataset | `nvidia/Nemotron-Cascade-2-SFT-Data` |
| Materialized token budget | 17.3B, prepared once before training |
| Validation | every 25 iterations; deterministic 1% holdout; 2 batches |
| Checkpoint interval | 50 iterations |
| Initial benchmark/exit | iteration 150 |
| Slurm duration exit | 220 minutes for a 4-hour allocation |

Keep `train_iters=1000` from the first launch. A duration exit before checkpoint
150 is incomplete: resume the same run and do not label or benchmark an earlier
checkpoint as QAD-150.

Monitor smoothed QAD/KD loss, learning rate, and gradient norm. Stop on non-finite
loss, repeated skipped iterations, or a sustained spike. Starting at iteration
100, stop and diagnose if two consecutive 50-step windows fail to lower median
loss.

Export and evaluate checkpoint 150 with the exact BF16/PTQ benchmark
configuration. Stop if it meets the recovery target. Otherwise continue only
when partial recovery is positive beyond run noise and training is healthy.
Choose later absolute checkpoints from that evidence.

Resume the same Megatron output directory without changing prepared data, data
paths/cache, seed, checkpoint lineage, optimizer, scheduler, iteration, or
consumed-sample state. Among Megatron training arguments, change only the
absolute `exit_interval`; keep topology unchanged, never restart from PTQ, and
never reset training progress. Stop when the recovery target is met, recovery
plateaus or regresses, training diverges, or iteration 1000 is reached.

Report exact revisions and commands; Slurm job/container/topology; sampled data
configuration and counts; PTQ recipe; checkpoint, loss, optimizer, and scheduler
state; and comparable BF16/PTQ/QAD results.
