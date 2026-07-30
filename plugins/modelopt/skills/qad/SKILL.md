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
- the common skill's `environment-setup.md`, `workspace-management.md`, and
  `slurm-setup.md`; also its `remote-execution.md` for remote Slurm

Treat the example README and `--help` output as authoritative for mutable flags,
commands, containers, and checkpoint formats. This skill supports Slurm only.

## Execute in this order

1. **Confirm the gap.** Reuse only validated, comparable BF16/PTQ results and
   the exact benchmark configuration from preceding evaluation or recipe
   search; run missing, invalid, or non-comparable baselines. Confirm the target
   benchmarks and their context-length needs. Stop if the PTQ gap to BF16 is
   already below 1%.
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
   TP/PP/CP/EP from student and teacher architecture, the chosen sequence length,
   and available GPU memory. Prefer CP before TP for small long-context models;
   keep EP=1 for dense models and ETP=1 because the current `distill.py`
   workflow does not support expert tensor parallelism. For MoE require:

   - `DP = world_size / (TP * PP * CP)`
   - `EDP = world_size / (EP * PP)`
   - integral DP/EDP, `num_experts % EP == 0`, and
     `GBS % (MBS * DP) == 0`

4. **Prepare the full capped dataset once.** Use suitable user-provided data, or
   copy `examples/megatron_bridge/data/nemotron-cascade-2-blend.yaml` as the
   default. Set the target tokenizer and workspace path, then materialize the
   randomly sampled subset before training. Pack the chosen sequence length;
   Megatron's `99,1,0` split creates the 1% validation holdout from the same
   data.
5. **Run and monitor QAD.** Run one QAD training job at a time and fold startup
   validation into it; do not submit separate GPU preflight jobs or split at
   recovery iterations. Let training continue while evaluating saved
   checkpoints, and cancel it when a stop condition below is met.

## Default training policy

| Setting | Default |
| --- | --- |
| Sequence length | 32768; adjust for target benchmarks |
| Peak / minimum LR | `1e-5` / `1e-6` |
| LR schedule | cosine |
| Training cap | 1000 iterations |
| Global batch size | 512 |
| Dataset | `nvidia/Nemotron-Cascade-2-SFT-Data` by default |
| Materialized token budget | 17.3B at 32K; cover the full cap at the chosen length |
| Training validation | every 25 iterations; deterministic 1% holdout; 2 batches |
| Checkpoint interval | 50 iterations |
| Loss logging | every 10 iterations |
| Recovery benchmark | 150, then every 100 iterations while training runs |
| Slurm duration exit | 220 minutes for a 4-hour allocation |

## Run policy

- Keep `train_iters=1000` and leave `exit_interval` unset.
- From initial step timing, submit only enough sequential jobs to reach
  checkpoint 150; never submit through iteration 1000 upfront. At each recovery
  checkpoint, submit to the next only after its targeted evaluation and any
  triggered full suite, and only if the BF16 gap remains at least 1% and
  recovery has neither plateaued nor regressed.
- Give all training jobs the same run-specific job name and
  `--dependency=singleton`; record job IDs and, on any stop, cancel pending jobs
  before the active job.
- Cancel on non-finite loss, repeated skipped iterations, or a sustained spike.
  At iteration 50, require the loss aggregate to be lower than at iteration 10.
- At each recovery checkpoint, first evaluate the one to three benchmarks with
  the largest PTQ drops. Run the remaining original PTQ suite at that checkpoint
  only after recovery beyond run noise.
- Cancel when the full-suite gap to BF16 is below 1%, benchmark recovery
  regresses beyond run noise, or benchmark recovery and loss both plateau.
- After a duration exit, resume the latest QAD checkpoint in the same output
  directory with unchanged prepared data paths/cache, seed, topology, optimizer,
  scheduler, iteration, and consumed-sample state; do not restart from PTQ.
- Report the PTQ recipe, data sample, Slurm topology, loss/state, checkpoints,
  and comparable BF16/PTQ/QAD results.
