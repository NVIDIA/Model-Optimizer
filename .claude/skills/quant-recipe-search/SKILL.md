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

Use this skill to steer quantization as a portfolio loop. It is an orchestration layer: it decides what to try next and which existing skill should do the execution.

Users may ask for the outcome directly, for example: "find the best quantization recipe and generate a PTQ checkpoint for this model." Treat that as enough to start. Recover what can be inferred from the workspace, ask only for missing constraints that materially affect the search, then delegate checkpoint generation to `ptq`.

A "best recipe" request is not complete after checkpoint generation. It requires evaluation and baseline comparison before recommending or promoting a checkpoint. If benchmark tasks or baseline results are missing, ask for them or delegate to `evaluation` to create the matching baseline/candidate runs.

## Skill Boundaries

- Use `ptq` to produce and validate checkpoints.
- Use `deployment` to serve checkpoints and debug serving-specific flags.
- Use `evaluation` to create NEL configs and submit evals.
- Use `launching-evals` to run, resume, debug, and analyze NEL runs.
- Use `monitor` for active job tracking.
- Use `accessing-mlflow` for MLflow artifact lookup.
- Use `compare-results` for validated baseline-vs-candidate deltas and score-field comparability.

Do not duplicate those workflows here. This skill should leave the user with a clear recipe portfolio, success metric, experiment sequence, and next decision.

## Core Loop

1. **Recover state**
   - Read existing result tables, recipe logs, state files, and experiment notes before proposing new candidates.
   - Ask `monitor` / `launching-evals` to check active jobs when needed.
   - Use `compare-results` when existing baseline/candidate evals need a formal comparability check.

2. **Define the target**
   - Ask the user what makes the compression successful before choosing recipes.
   - If the user did not provide an optimization objective, stop and ask them to choose before planning candidates. Do not infer or default silently.
   - Offer these default objective choices:
     - **Compute / throughput:** typical data-center target. Prefer recipes with activation quantization such as NVFP4 or FP8 when the downstream stack can use fast kernels.
     - **Memory / latency:** typical edge target. Minimize activated memory per forward pass to reduce latency; prefer weight-only or W4A16-style recipes when they preserve accuracy.
     - **Custom:** user-provided objective, such as checkpoint size, throughput at a fixed batch size, decode latency, prefill latency, or a product-specific memory budget.
   - Ask for the primary quantization format or search family before choosing recipes.
   - If the user did not provide a primary quantization format, stop and ask them to choose. Do not silently choose FP8/W8A8 because it is likely lossless.
   - Offer common format choices:
     - **NVFP4 / NVFP4_MSE:** low-bit Blackwell-oriented search family.
     - **W4A16 NVFP4:** weight-only NVFP4 family, often useful for memory/latency targets.
     - **FP8 / W8A8:** near-lossless baseline or primary target if the user explicitly chooses FP8.
     - **INT4 / AWQ:** weight-only INT4 family for low-batch memory/latency use cases.
     - **Custom / mixed:** user-provided format set, such as `nvfp4,fp8`, `w4a16_nvfp4+fp8_attn`, or model-specific recipe constraints.
   - Default acceptance goal: find the recipe with the best performance for the chosen objective while keeping each benchmark's accuracy loss under 1 percentage point versus the matching baseline.
   - Treat near-threshold or noisy benchmark deltas as inconclusive until reruns confirm whether the drop is a real regression.
   - Record recipe-selection criteria: optimization objective, primary quantization format/search family, target active bytes/token, acceptable accuracy loss, calibration budget, and any user-provided throughput/latency goal.
   - Include quantization metadata such as scale storage in size estimates.
   - Keep accuracy and verbosity/token usage as separate first-class metrics.

3. **Generate candidates**
   - Start with baselines: BF16/FP16, all-FP8 or W8A8, and one conservative low-bit recipe.
   - Treat all-FP8/W8A8 as a near-lossless baseline unless the user selected FP8/W8A8 as the primary format. Do not end the search at FP8 just because it has the smallest accuracy drop.
   - If using ModelOpt, start from `modelopt_recipes` rather than inventing patterns from scratch. Prefer model-specific recipes first, then general PTQ presets/fragments.
   - Let the ModelOpt `ptq` skill own checkpoint generation and PTQ validation; use this skill to choose the objective, sequence recipes, compare results, and decide the next iteration.
   - Generate PTQ checkpoints only as candidates. Do not call a candidate "best" until it has passed the evaluation and comparison stages below.
   - Use AutoQuant or sensitivity tooling for broad search and module ranking.
   - Use manual recipes for controlled ablations by module family.
   - Change one major recipe axis at a time: weight format, activation format, quantization granularity, calibration method, excluded modules, or module family.

4. **Gate before scaling**
   - Ask `ptq` to validate checkpoint coverage and metadata after generation.
   - Ask `deployment` or `evaluation` for runtime sanity only when execution behavior is needed to qualify the recipe.
   - Promote only candidates that pass checkpoint validation and the required delegated sanity checks.

5. **Evaluate in stages**
   - For any request that says "best", "search", or "optimize", run or recover evaluations for the baseline and every candidate that reaches this stage. Do not stop at PTQ checkpoint generation unless the user explicitly asks to pause before eval.
   - Pick cheap screen benchmarks that expose likely failure modes for the model/domain.
   - If the benchmark set is missing, ask the user which benchmark suite defines success; use the `<1pp` default acceptance goal only after a benchmark set exists.
   - Ask `evaluation` to create or modify configs and submit runs.
   - Ask `launching-evals` / `monitor` to track, resume, and debug runs.
   - Ask `compare-results` to validate comparability and compute deltas.
   - Treat sampling, parser, token-cap, and runtime/backend changes as non-recipe variables unless the user explicitly asks to study them.

6. **Refresh tables**
   - Maintain a recipe portfolio table, not a replacement for evaluator artifacts.
   - Include recipe name, objective, active-cost estimate, calibration notes, checkpoint reference, comparison reference, accuracy summary, verbosity summary, and decision.
   - Link to `compare-results` / `launching-evals` artifacts for exact metric extraction and provenance.

7. **Decide next iteration**
   - Promote a recipe only after `compare-results` shows the candidate is comparable to the baseline and satisfies the accuracy-loss constraint for the chosen benchmark set.
   - If accuracy drops, inspect the most sensitive module families first.
   - If verbosity grows, compare parser settings, token caps, failed samples, backend changes, and sampling config before blaming quantization numerics.
   - If AutoQuant produces identical recipes for multiple budgets, inspect recipe hashes and achieved bits; adjust constraints/objective before launching a bigger sweep.

## Practical Defaults

- Ask whether the primary success metric is compute/throughput, memory/latency, or a custom objective. Do not assume, and do not proceed to candidate planning until the objective is explicit.
- Ask for the primary quantization format/search family, such as NVFP4, W4A16 NVFP4, FP8/W8A8, INT4/AWQ, or a custom mixed set. Do not assume, and do not silently select FP8 as the final recipe.
- Default to a `<1pp` per-benchmark accuracy-loss constraint versus the matching baseline unless the user gives another threshold.
- A generated PTQ checkpoint is a candidate artifact, not the final answer to a "best recipe" request. Evaluation and comparison are required before final selection.
- Prefer active runtime cost over checkpoint size when optimizing routed or sparsely activated models.
- Always compare against BF16/FP16 and a near-lossless FP8/W8A8 baseline.
- Treat benchmark variance as real: run repeat sweeps for close decisions.
- Do not mix parser/no-parser, FP8-KV/BF16-KV, runtime/backend, or sampling changes into recipe conclusions unless they are explicitly part of the experiment.
- Use delegated pipe-clean checks for new checkpoint families before full evals.

## References

- For recipe design, sensitivity, and active-cost accounting, read `references/recipe_iteration.md`.
- For a concrete prior case study, read `references/qwen36_case_study.md` only when Qwen3.5/Qwen3.6 details are relevant.
