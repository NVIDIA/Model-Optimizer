---
name: qad
description: >-
  Run explicitly requested ModelOpt Quantization-Aware Distillation (QAD) through
  Megatron Bridge (recommended) or Hugging Face Transformers to
  recover a measured BF16-to-PTQ accuracy gap.
  Use only when the user explicitly asks for QAD, including its topology, data
  preparation, launch, resume, checkpoint export, or recovery decisions.
---

# ModelOpt Quantization-Aware Distillation

QAD is expensive. Run it only when the user explicitly authorizes QAD for the
target model or run. A Day-0, PTQ, evaluation, comparison, or recipe-search
request alone is not authorization to start QAD.

## Confirm the gap

For both frameworks, reuse only validated, comparable BF16/PTQ results and
the exact benchmark configuration from preceding evaluation or recipe
search; run missing, invalid, or non-comparable baselines. Confirm the target
benchmarks and their context-length needs. Stop if the PTQ gap to BF16 is
already below 1%.

## Follow the supported workflow

Read `docs/source/guides/quantization_aware_training_and_distillation.rst`
for QAD background and framework selection.
Recommend Megatron Bridge for scalable training and its model-specific tutorials.
Choose the applicable tutorial from `examples/megatron_bridge/tutorials/README.md`.
Honor an explicit HF choice.

Use the common skill's `environment-setup.md` and `workspace-management.md`;
for Slurm, also read `slurm-setup.md` and, when remote, `remote-execution.md`.

Treat the example README and `--help` output as authoritative for mutable flags,
commands, containers, and checkpoint formats. Both frameworks can run on local
GPUs or Slurm; use the user's requested environment.

## Megatron Bridge execution and training policy

For Megatron Bridge, before constructing commands, read:

- `examples/megatron_bridge/README.md`, especially PTQ, data preparation, QAD,
  export, and Slurm usage
- `examples/megatron_bridge/{quantize.py,distill.py}` via `--help`

### Execute in this order

1. **Reproduce PTQ and verify compatibility.** In the target runtime, require
   `AutoBridge.can_handle()` for the target model and PTQ through `quantize.py`
   to succeed while preserving the exact preceding PTQ config or recipe:
   format, layer selection, calibration data/count, sequence length, and seed.
   A changed quantization setting is a new PTQ candidate and must be evaluated
   before QAD. In the master-rank `.quant_summary.txt`, require finite positive
   `amax` for enabled static quantizers; accept `dynamic`/format-defined `None`
   only when the recipe intends it. Treat the summary as rank-local under model
   parallelism.
2. **Choose topology explicitly.** Derive the smallest fitting node count and
   TP/PP/CP/EP from student and teacher architecture, the chosen sequence length,
   and available GPU memory. Prefer CP before TP for small long-context models;
   keep EP=1 for dense models and ETP=1 because the current `distill.py`
   workflow does not support expert tensor parallelism. For MoE require:

   - `DP = world_size / (TP * PP * CP)`
   - `EDP = world_size / (EP * PP)`
   - integral DP/EDP, `num_experts % EP == 0`, and
     `GBS % (MBS * DP) == 0`

   Static-block NVFP4 recipes require TP=1; use PP for memory. With PP, balance
   stages with `--num_layers_in_{first,last}_pipeline_stage` (the first and last
   stages also hold the embedding and output layer) and recompute only where
   memory needs it (`--recompute_num_layers_per_stage`). Time a few iterations
   of each candidate layout in the first job and keep the fastest.

3. **Prepare the full capped dataset once.** Use suitable user-provided data, or
   copy `examples/megatron_bridge/data/nemotron-cascade-2-blend.yaml` as the
   default. Set the target tokenizer and workspace path, then materialize the
   randomly sampled subset before training. Pack the chosen sequence length;
   Megatron's `99,1,0` split creates the 1% validation holdout from the same
   data. Keep every source mixed across training, never ordered as a
   curriculum, so early stops see the whole blend.
4. **Run and monitor QAD.** Run one QAD training job at a time and fold startup
   validation into it; do not submit separate GPU preflight jobs or split at
   recovery iterations. Let training continue while evaluating saved
   checkpoints, and cancel it when a stop condition below is met.

### Default training policy

| Setting | Default |
| --- | --- |
| Sequence length | 32768; adjust for target benchmarks |
| Peak / minimum LR | `1e-5` / `1e-6` |
| LR schedule | cosine |
| LR horizon (`train_iters`) | 1000 iterations |
| Global batch size | 512 |
| Dataset | `nvidia/Nemotron-Cascade-2-SFT-Data` by default |
| Materialized token budget | 17.3B at 32K; cover `train_iters` at the chosen length |
| Training validation | every 25 iterations; deterministic 1% holdout; 2 batches |
| Checkpoint interval | 50 iterations |
| Loss logging | every 10 iterations |
| Recovery benchmark | from the KD-loss knee, then every 50-100 iterations |
| Slurm duration exit | 220 minutes for a 4-hour allocation |

### Run policy

- `train_iters` sets the cosine LR horizon and is a loop knob: with 1000, the
  LR stays near its peak through iteration 150. Stop runs by the decisions
  below, not by `train_iters`, and leave `exit_interval` unset.
- On Slurm, use initial step timing to submit only enough sequential jobs to reach
  a few checkpoints past the expected KD-loss knee; never submit through
  `train_iters` upfront. At each recovery checkpoint, submit to the next only
  after its targeted evaluation and any triggered full suite, and only while
  the decision is to continue.
- On Slurm, give all training jobs the same run-specific job name and
  `--dependency=singleton`; record job IDs and, on any stop, cancel pending jobs
  before the active job.
- Cancel on non-finite loss, repeated skipped iterations, or a sustained spike.
  At iteration 50, require the loss aggregate to be lower than at iteration 10.
- At each recovery checkpoint, first evaluate the one to three benchmarks with
  the largest PTQ drops or that failed in earlier loops. Run the remaining
  original PTQ suite at that checkpoint only after recovery beyond run noise.
- Start benchmark evaluation at the knee of the KD loss curve, where its steep
  drop ends, rounded to the next checkpoint, then evaluate every checkpoint
  while the scores still change. Before the knee, use loss only to catch
  drift.
- Run noise is about 2x the standard error of a benchmark's gap to BF16, pooled
  over repeated evaluations; a single evaluation is often noisier than 0.5
  points. Repeat a benchmark only when its gap is within run noise of 1%, and
  cap the repeats.
- Export each checkpoint to evaluate and pass the ptq skill's
  `references/checkpoint-validation.md` gate on it before evaluation.
- At each evaluated checkpoint, decide:
  - **Pass:** every benchmark is within 1% of BF16. Keep the checkpoint and
    stop the run.
  - **Miss:** the scores converge short of 1%, two evaluated checkpoints past
    the knee bring no gain beyond run noise, or two consecutive checkpoints
    decline. Stop and change one QAD knob in the [recipe loop](#recipe-loop).
  - **Otherwise continue** to the next checkpoint.
- Pick the checkpoint by benchmarks, never by KD loss, which keeps falling after
  the benchmarks peak. Between near ties, the benchmark most sensitive to
  quantization decides.
- After a duration exit, resume the latest QAD checkpoint in the same output
  directory with unchanged prepared data paths/cache, seed, topology, optimizer,
  scheduler, iteration, and consumed-sample state; do not restart from PTQ.
- Report the PTQ recipe, data sample, training topology, loss/state, checkpoints,
  and comparable BF16/PTQ/QAD results.

### Recipe loop

Each loop runs QAD on the best PTQ recipe for one layout and changes one thing
from the last loop. Record each loop's recipe, calibration, QAD settings,
checkpoints, and scores.

1. **Start from the best PTQ recipe for the target.** Take it from the
   preceding PTQ evaluation or recipe search, and run QAD on it with the
   default policy.
2. **On a pass, compress more by one module group** from the passing recipe,
   in order of bytes saved; never add two groups, or the MTP head, in one loop.
   Find the best PTQ recipe for the new layout, starting from the passing
   recipe's calibration, and evaluate it before its QAD. When GPUs are idle,
   run the next 2-3 single-group candidates in parallel. A group can be a slice:
   its middle layers with the first and last few kept, or some of its
   projections (such as GDN `in_proj` only, or attention `linear_qkv` only).
3. **On a miss, change one QAD knob.** Run every new layout first with the
   default policy as its control. Read each missed benchmark against PTQ:
   worse than PTQ is drift (lower the peak LR or add matching data); better
   than PTQ but short is under-recovery (keep the LR high longer with a larger
   `train_iters`, or train further). Knobs, in order: peak LR, `train_iters`,
   data blend, then teacher-generated data.
4. **Compare a knob with its control at the same checkpoints** past the knee,
   on benchmark scores pooled over fresh runs, never on KD loss. A rejection
   holds only for that PTQ base: rerun the control when the base changes.
5. **Compress less** only after the knobs miss clearly on a layout, or at
   once when QAD stays flat against PTQ or far from passing: shrink the last
   added group to a slice before dropping it.

## Hugging Face execution and training policy

Before constructing commands, read `examples/llm_qat/README.md`, its
`ARGUMENTS.md`, selected training/dataset/Accelerate configs, and
`{quantize.py,train.py,export.py} --help`.
Follow its quantize → distill → export flow: the PTQ checkpoint is the student;
the original unquantized model is the teacher. Adapt the QAD training and
Accelerate configs to the model and budget.
Resolve tutorial-relative commands and config paths from `examples/llm_qat/`.
Use the HF tutorial's evaluation and checkpoint resume guidance.

Reuse the Bridge procedure's PTQ-recipe preservation and one-training-job-at-a-time
guidance. Apply the [Run policy](#run-policy)'s loss-stability and
benchmark-recovery stop conditions and report the same fields.
Use HF evaluation/checkpoint intervals; Bridge-specific topology, data defaults,
iteration thresholds, and Slurm job chaining are not HF requirements.
