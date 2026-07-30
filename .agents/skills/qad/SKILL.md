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
5. **Run and monitor QAD.** Run one QAD training job at a time and fold startup
   validation into it; do not submit separate GPU preflight jobs or split at
   recovery iterations. Let training continue while evaluating saved
   checkpoints, and cancel it when a stop condition below is met.

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
| Training validation | every 25 iterations; deterministic 1% holdout; 2 batches |
| Checkpoint interval | 50 iterations |
| Loss logging | every 10 iterations |
| Recovery benchmark | 150, then every 100 iterations while training runs |
| Slurm duration exit | 220 minutes for a 4-hour allocation |

Keep `train_iters=1000` from the first launch and do not set `exit_interval`.
Benchmark saved checkpoints 150, 250, 350, and so on while the training job
continues. After a representative allocation, estimate jobs to the next
checkpoint as `ceil(remaining_iters / observed_iters_per_allocation)`, using
end-to-end progress that includes startup, validation, and saves. Before then,
treat the early seconds-per-iteration estimate as a lower bound and provision
one additional allocation. Resume after a duration exit under the rule below.

Monitor smoothed QAD/KD loss, learning rate, and gradient norm. Cancel immediately
on non-finite loss, repeated skipped iterations, or a sustained spike; preserve
the latest complete checkpoint and do not wait for another save. With
`log_interval=10`, require the loss aggregate reported at iteration 50
(iterations 41–50) to be lower than the one at iteration 10 (iterations 1–10);
otherwise diagnose and cancel under the evidence-driven rule below.

At each recovery checkpoint, first export and evaluate only the one to three
benchmarks with the largest measured PTQ drops, using their exact BF16/PTQ
configurations. At checkpoint 150, new recovery means QAD-150 over PTQ; later it
means improvement over the prior evaluated QAD checkpoint. In all cases, judge
BF16-gap recovery against the fixed BF16/PTQ baselines. When the targeted set
shows new recovery beyond run noise, run the remaining original PTQ benchmark
suite at that same checkpoint; otherwise do not. Declare the recovery target met
only from a full-suite result at that checkpoint. Wait until a recovery
checkpoint is fully committed before export; never read one still being written.

For normal duration resumes, use the same Megatron output directory without
changing prepared data, data paths/cache, seed, topology, checkpoint lineage,
optimizer, scheduler, iteration, or consumed-sample state; never restart from
PTQ or reset training progress. If emergency cancellation occurs before the
first complete QAD checkpoint, report that no resumable QAD state exists and
diagnose before relaunching from PTQ.

Cancel the training job when the recovery target is met. Otherwise continue
while training is healthy and recovery has not regressed beyond run noise. At
each recovery checkpoint, compare the median of the five latest 10-step loss
aggregates with the preceding five. If benchmark change is flat within run
noise, continue only when the latest loss median is lower; otherwise cancel.
Cancel on recovery regression beyond run noise. For an evidence-driven
cancellation, wait only for an in-progress scheduled save to commit; do not
continue to create a future checkpoint. Never train past iteration 1000.

Report exact revisions and commands; Slurm job/container/topology; sampled data
configuration and counts; PTQ recipe; checkpoint, loss, optimizer, and scheduler
state; and comparable BF16/PTQ/QAD results.
