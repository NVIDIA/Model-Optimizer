---
name: qad
description: >-
  Run explicitly requested ModelOpt Quantization-Aware Distillation (QAD) through
  Megatron Bridge (recommended) or Hugging Face Transformers to recover a
  measured BF16-to-PTQ accuracy gap, including the PTQ -> QAD recipe loop
  (compress more on a pass, change one QAD knob or compress less on a miss),
  topology, data preparation, launch, per-checkpoint evaluation and stop
  decisions, resume, and export. Use only when the user explicitly asks for QAD.
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

## Goal

A quantized checkpoint whose every target benchmark is within 1 percentage
point of BF16 (`BF16 - quantized < 1 pp`), at the most compression that
achieves it. Evaluate BF16, PTQ, and QAD checkpoints through the
**evaluation** skill with identical configs apart from the checkpoint: its task
recipes set the repeat counts (SciCode: 8+ separate submissions), and its run
validation decides which runs count. Add precision by submitting a benchmark
more times, in parallel, and pooling the runs, not by raising `num_repeats`.
AA-LCR and Terminal-Bench are slow: run them on a PTQ or QAD checkpoint only
after it passes every other benchmark.

## Flow

```text
Conservative recipe
  │
  ↓
PTQ calibration sweep + evaluate ←───────── next recipe ──────────────────┐
  ├─ all pass ──→ compress more ──────────────────────────────────────────┤
  │ miss: QAD the best calibrated base                                    │
  ↓                                                                       │
Profile layout → smoke test → train to the horizon ←───────────────┐      │
  │                                                                │      │
  ↓                                                                │      │
Each checkpoint: failing benchmarks first → gate_qad.py            │      │
  ├─ CONTINUE ──────→ keep training                                │      │
  ├─ NEEDS_REPEATS ─→ resubmit the blocking benchmark, pool        │      │
  ├─ SCREEN_PASS ───→ run the rest (AA-LCR, TB last)               │      │
  ├─ PASS ──────────→ confirmation run → pass: compress more ─────────────┤
  └─ MISS (horizon) ─→ still improving: longer horizon ────────────┤      │
                       near miss or knob untested: one knob ───────┘      │
                       clear miss in 2 settings: compress less ───────────┘

Finish: the most compressed recipe that passes, or when the budget runs out.
```

## Recipe loop

A loop is one PTQ recipe plus one QAD configuration. Change exactly one thing
per loop, and record every loop (recipe, QAD settings, job IDs, checkpoints,
per-benchmark deltas, decision) including failed and unevaluated ones.

1. **Start from the conservative recipe** (e.g. the Day-0 recipe: NVFP4 MLP,
   FP8 attention, FP8 KV).
2. **Sweep its PTQ calibration before any QAD.** A PTQ base costs one short
   job and a screen, and calibration has recovered more than QAD loops. Run a
   small factorial early: four-over-six scales (`nvfp4_four_over_six`) ×
   activation headroom (`nvfp4_act_headroom`) × weight calibration (`max`,
   `local_hessian`, `gptq`). If a base passes, skip QAD for it and go to step
   4; otherwise QAD the best base. When a better base appears, re-base running
   loops on it instead of finishing them.
3. **QAD it** with the default policy below, evaluating every checkpoint.
4. **On a pass, compress more:** quantize one more module group to the target
   format (e.g. full-attention projections, then linear-attention/GDN
   projections). Prefer the middle layers first, keeping the first and last
   ~5 layers at the previous precision, then extend to all layers. Calibrate
   and evaluate each new recipe before its QAD. Start the next tier as soon
   as the current one is near passing, in parallel.
5. **On a miss still closing the gap** (`improving`), keep going: rerun the
   same recipe with a longer horizon (e.g. 150 to 250), however low PTQ
   started; it does not count as a knob's miss.
6. **On any other miss, change one QAD knob** on the same recipe, such as the
   LR, the data blend, or the horizon. A near miss always gets another knob.
   Reject a knob only after it misses in at least 2 settings (different base
   or LR), with the loss curves checked.
7. **Compress less only after clear misses:** return to the last passing
   recipe, or, if none passed, keep more modules (first/last layers, attention)
   at higher precision than the starting recipe.
8. **Finish** with the most compressed recipe that passes, or when the user's
   budget (loops or GPU-hours) runs out; report the closest candidate and its
   blocking benchmark if none passes.

## Megatron Bridge execution and training policy

For Megatron Bridge, before constructing commands, read:

- `examples/megatron_bridge/README.md`, especially PTQ, data preparation, QAD,
  export, and Slurm usage
- `examples/megatron_bridge/{quantize.py,distill.py}` via `--help`

### Execute each QAD run in this order

1. **Reproduce PTQ and verify compatibility.** In the target runtime, require
   `AutoBridge.can_handle()` for the target model and PTQ through `quantize.py`
   to succeed with the loop's exact recipe: format, layer selection,
   calibration data/count, sequence length, and seed. A changed quantization
   setting is a new PTQ candidate and must be evaluated before QAD. In the
   master-rank
   `.quant_summary.txt`, require finite positive `amax` for enabled static
   quantizers; accept `dynamic`/format-defined `None` only when the recipe
   intends it. Treat the summary as rank-local under model parallelism.
2. **Choose the topology.** Read `references/topology.md`: TP=1 for static
   NVFP4 weight scales, the smallest fitting PP/DP, and a balanced pipeline
   layout from a short profile. Prefer CP for small long-context models; keep
   EP=1 for dense models and ETP=1 (`distill.py` does not support expert tensor
   parallelism). For MoE require:

   - `DP = world_size / (TP * PP * CP)`
   - `EDP = world_size / (EP * PP)`
   - integral DP/EDP, `num_experts % EP == 0`, and
     `GBS % (MBS * DP) == 0`

3. **Prepare the full capped dataset once.** Use suitable user-provided data, or
   copy `examples/megatron_bridge/data/nemotron-cascade-2-blend.yaml` as the
   default. Set the target tokenizer and workspace path, then materialize the
   randomly sampled subset before training. Pack the chosen sequence length;
   Megatron's `99,1,0` split creates the 1% validation holdout from the same
   data.
4. **Smoke-test the run.** Whenever the layout or sequence length changes,
   run a short job in a held allocation that trains a few iterations, runs
   validation, saves a checkpoint, and resumes from it. Before every launch,
   diff the arguments against the reference loop and check each change's side
   effects (e.g. `--kd_loss_alpha` below 1 adds the LM loss, which also trains
   the MTP head unless `--freeze_mtp` is set).
5. **Run, evaluate, and decide.** Keep one active training job per run;
   independent runs may go in parallel. Submit each checkpoint's evaluation
   as soon as it is saved, in parallel with training and other evaluations,
   failing benchmarks first (largest PTQ drops), at repeat counts fixed for
   the whole campaign. Then decide with:

   ```bash
   python "$SKILL_DIR/../day0-release/scripts/gate_qad.py" --scores <qad-scores.json>
   ```

   Gate only validated runs, giving every score the SE of its delta from the
   run SDs (see `--help`) and the run's `--train_iters` as `horizon`. A task
   is a near miss when its gap is at least 1 pp but less than 1 pp + 2 SE,
   and a clear miss above that. `CONTINUE`: keep training; no miss stops a run
   before its horizon. `NEEDS_REPEATS`: the best near miss so far, within
   1.5 SE of the threshold; resubmit its blocking benchmark, pool, and gate
   again, one benchmark at a time up to its `max_runs` cap (about twice the
   recipe's). `SCREEN_PASS`: run the remaining benchmarks on that checkpoint.
   `MISS`: the run ended without a pass; `improving` means it was still
   closing the gap, so train the same recipe longer, else take the next loop. For
   `CONTINUE`/`MISS` only, two adjacent checkpoints near the end of the anneal
   may be pooled as one. `PASS`: run a confirmation run (fresh runs of every
   benchmark at full repeat counts, not reusing screen runs) and accept only
   if it passes the gate too, since the best of many screened checkpoints is
   optimistic. Validate the accepted checkpoint with the ptq skill's
   `references/checkpoint-validation.md` serving canary.

### Default training policy

| Setting | Default |
| --- | --- |
| Sequence length | 16384 (8K-32K is reasonable; not a loop knob) |
| Peak / minimum LR | `1e-5` / `1e-6` |
| LR schedule | cosine over `--train_iters`, 50 warmup iterations |
| KD loss | logit KL, `--logit_kl_top_k 4096` |
| Horizon (`--train_iters`) | 150, the planned stop; longer when a run is still improving at it, shorter if runs keep degrading |
| Global batch size | 512 |
| Dataset | `nvidia/Nemotron-Cascade-2-SFT-Data` by default |
| Materialized token budget | cover the horizon at the chosen length (150 x 512 x 16K = 1.3B tokens) |
| Training validation | every 25 iterations; deterministic 1% holdout; 2 batches |
| Checkpoint interval | 50 iterations |
| Recovery benchmark | every checkpoint: failing benchmarks first, the rest on `SCREEN_PASS` |
| Loss logging | every 10 iterations |
| Slurm duration exit | 220 minutes for a 4-hour allocation |

QAD typically peaks within the first few hundred iterations and then declines
on benchmarks while the KD loss keeps falling. Select checkpoints on benchmark
scores only: validation KD loss and drift probes have not tracked them, so use
them as health checks. Do not add image-text KD data at high weight (e.g. 20)
to recover a vision benchmark; in trials it ended below the PTQ base.

### Run policy

- Set `--train_iters` to the horizon so the LR has annealed when the run
  stops, and leave `exit_interval` unset. Checkpoints are
  large (about 14 bytes per parameter with the optimizer state), so keep only
  the latest ones needed to resume (`--checkpoint_keep_last 2`) plus the best
  checkpoint so far (`cp -al` it out of the rotation); delete a checkpoint, and
  its evaluation export, once the gate has passed it over.
- On Slurm, use initial step timing to submit enough sequential jobs to reach the
  horizon; cancel the rest on `PASS`.
- On Slurm, give all training jobs of a run the same job name and
  `--dependency=singleton`; record job IDs and, on any stop, cancel pending
  jobs before the active job.
- Cancel on non-finite loss, repeated skipped iterations, or a sustained spike.
  At iteration 50, require the loss aggregate to be lower than at iteration 10.
- After a duration exit, resume the latest QAD checkpoint in the same output
  directory with unchanged prepared data paths/cache, seed, topology, optimizer,
  scheduler, iteration, and consumed-sample state; do not restart from PTQ.
  With MLflow, pass `--mlflow_resume` so all segments log to one run.
- Report each loop's PTQ recipe, data sample, training topology, loss/state,
  checkpoints, and comparable BF16/PTQ/QAD results.

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
guidance. Follow the same recipe loop and gate every evaluated checkpoint with
`gate_qad.py`; apply the [Run policy](#run-policy)'s loss-stability conditions
and report the same fields.
Use HF evaluation/checkpoint intervals; Bridge-specific topology, data defaults,
iteration thresholds, and Slurm job chaining are not HF requirements.
