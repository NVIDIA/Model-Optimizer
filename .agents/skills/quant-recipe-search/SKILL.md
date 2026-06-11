---
name: quant-recipe-search
description: >-
  Use when the user asks to find, search for, or optimize the best quantization
  recipe for a model, including direct requests like "find the best quantization
  recipe and generate a PTQ checkpoint." Guides the multi-candidate loop:
  choose compute-vs-memory success metrics, select ModelOpt recipe baselines,
  design AutoQuant/manual recipe deltas, interpret sensitivity, and decide next
  candidates. Do NOT use for a single known PTQ recipe run (use ptq), serving
  (use deployment), creating/running evals (use evaluation or launching-evals),
  monitoring jobs (use monitor), MLflow browsing (use accessing-mlflow), or
  comparing completed baseline-vs-candidate scores only (use compare-results).
---

# Quant Recipe Search

Use this skill when quantization is an iterative recipe search, not a one-off
PTQ run. The skill owns strategy: define success, choose the search space,
sequence candidates, and decide the next iteration. It delegates checkpoint
generation, serving, evaluation, monitoring, and metric comparison to the
existing execution skills.

Treat a direct request such as "find the best quantization recipe and generate a
PTQ checkpoint for this model" as enough to start. Recover local state first,
then ask only for missing decisions that change the search.

## Skill Boundaries

- Use `ptq` to produce and validate checkpoints.
- Use `deployment` to serve checkpoints and debug serving-specific flags.
- Use `evaluation` to create NEL configs and submit evals.
- Use `launching-evals` to run, resume, debug, and analyze NEL runs.
- Use `monitor` for active job tracking.
- Use `accessing-mlflow` for MLflow artifact lookup.
- Use `compare-results` for validated baseline-vs-candidate deltas and score-field comparability.

Do not duplicate those workflows here. This skill should leave the user with a clear recipe portfolio, success metric, experiment sequence, and next decision.

## Problem

The task is to find the best recipe for a user-defined target, not merely to
produce a quantized checkpoint. A generated PTQ checkpoint is only a candidate.
It becomes a recommended recipe only after evaluation and comparison against the
matching baseline.

Required inputs before planning candidates:

- **Optimization goal:** compute/throughput, memory/latency, or a custom metric.
- **Primary quantization family:** for example NVFP4, W4A16 NVFP4, FP8/W8A8,
  INT4/AWQ, or a custom mixed set.
- **Benchmark set or baseline results:** the user-defined acceptance surface.

If any of these are missing, ask for them. Do not silently default to FP8/W8A8
or call a checkpoint "best" before evaluation.

Default success rule: maximize the chosen performance objective while keeping
each benchmark within 1 percentage point of the matching BF16/FP16 baseline.
Near-threshold or noisy regressions require reruns before making a decision.

## Search Space

Keep the search space explicit. A candidate recipe is a tuple across these axes:

- **Numeric format:** FP8/W8A8, NVFP4/W4A4, W4A16 NVFP4, INT4/AWQ, or mixed
  formats such as NVFP4+FP8.
- **Calibration/search algorithm:** max calibration, MSE calibration, GPTQ,
  AWQ, AutoQuant scoring, and calibration dataset or sample-count variants.
- **Selection method:** manual/heuristic rules, sensitivity-guided manual
  recipes, AutoQuant selection, or a hybrid of AutoQuant plus manual overrides.
- **Module family:** attention, MLP, MoE experts, routers/gates, embeddings,
  `lm_head`, adapters, vision encoders, and model-specific modules.
- **Runtime fusion constraints:** modules fused by the inference library must
  use compatible quantization. Examples: vLLM Qwen `linear_attn.in_proj_qkvz`
  and fused MoE expert projections such as gate/up (`w1`/`w3`).
- **Calibration budget:** dataset mix, sample count, sequence length, and batch
  settings.

Do not collapse the search to one dimension such as numeric format only. Read
`references/recipe_iteration.md` when choosing concrete axes or candidates.

## Design Workflow

1. **Recover state**
   - Read result tables, recipe logs, AutoQuant states, sensitivity reports, and
     experiment notes before proposing new work.
   - Ask `monitor`, `launching-evals`, or `compare-results` to recover active
     job state and completed metrics when needed.

2. **Define the target**
   - Confirm the optimization goal, primary quantization family, benchmark set,
     accuracy-loss threshold, calibration budget, and cost metric.
   - Include quantization metadata such as scale storage in active-cost or size
     estimates.

3. **Pick baselines and first candidates**
   - Always include BF16/FP16 and a near-lossless FP8/W8A8 baseline unless FP8
     itself is the target.
   - For ModelOpt work, start from `modelopt_recipes`: model-specific recipes
     first, then general PTQ presets or recipe fragments.
   - Add an AutoQuant candidate in the requested primary family when AutoQuant
     is available. Expect AutoQuant to find a better trade-off than a first
     manual recipe, but validate that assumption with the same evals.
   - Add at least one manual or sensitivity-guided candidate so AutoQuant can be
     compared against controlled ablations and there is a fallback if AutoQuant
     misses the best frontier or hits runtime constraints.

4. **Generate candidates**
   - Delegate checkpoint generation and PTQ validation to `ptq`.
   - Change one major axis at a time: format, calibration algorithm, module
     selection, granularity, exclusions, or calibration data.
   - Use AutoQuant for broad candidate generation and sensitivity reports; use
     manual recipes for controlled module-family ablations and overrides.

5. **Gate before scaling**
   - Validate checkpoint coverage and metadata.
   - Reject or rewrite recipes that mix quantization algorithms inside a fused
     runtime group.
   - If the checkpoint is valid but serving fails due to runtime support, do not
     reject the recipe immediately. Delegate to `deployment` / `debug` for small
     patches or flags, then rerun a pipe-clean check.

## Iteration Loop

1. Run cheap screen evals for every candidate that passes the gates.
2. Compare accuracy, verbosity/token usage, and active cost against baselines.
3. Rerun noisy or near-threshold results before labeling a regression.
4. Decide the next candidate:
   - Accuracy drop: protect or ablate sensitive module families, try MSE/GPTQ,
     or use AutoQuant sensitivity to choose overrides.
   - Poor performance/cost: quantize the next high-cost active family, adjust
     active-cost objective, or try a more aggressive format.
   - AutoQuant underperforms manual recipes: inspect sensitivity reports,
     achieved bits, excluded modules, and runtime-fusion constraints; keep the
     manual recipe in the portfolio instead of forcing the AutoQuant result.
   - Runtime incompatibility: rewrite around fused groups or isolate deployment
     support from checkpoint quality.
   - Repeated AutoQuant recipes: inspect achieved bits and recipe hashes, then
     adjust constraints before launching a larger sweep.
5. Promote only when `compare-results` shows the candidate is comparable to the
   baseline and satisfies the user-defined goal.

Maintain a recipe portfolio table with recipe name, objective, active-cost
estimate, calibration notes, checkpoint path, eval/log references, accuracy,
verbosity, and decision.

## References

- For recipe design, search-space details, sensitivity, and active-cost
  accounting, read `references/recipe_iteration.md`.
- For a concrete prior case study, read `references/qwen36_case_study.md` only
  when Qwen3.5/Qwen3.6 details are relevant.
