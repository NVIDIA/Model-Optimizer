---
name: quant-recipe-loop
description: Use when building or iterating quantization recipes for LLM/VLM checkpoints, including AutoQuant/PTQ, manual mixed-precision recipe creation, sensitivity analysis, active-memory accounting, sanity checks, and accuracy/verbosity evaluation tables.
---

# Quant Recipe Loop

Use this skill to run quantization as an iteration loop: define the objective, generate candidate recipes, sanity-gate the checkpoint, evaluate accuracy and verbosity, update the results table, and choose the next candidate.

## Core Loop

1. **Recover state**
   - Read existing result tables, run logs, state files, and experiment notes before launching anything.
   - Check active jobs and avoid duplicates.
   - Resume interrupted evaluator jobs from their output folder when possible; restart only when the config/checkpoint is wrong.

2. **Define the target**
   - Ask the user what makes the compression successful before choosing formats or recipes.
   - If the user does not specify, use one of two default objectives:
     - **Compute / throughput:** typical data-center target. Optimize large-batch throughput by using activation quantization such as NVFP4 or FP8 where the serving stack has fast kernels.
     - **Memory / latency:** typical edge target. Minimize activated memory per forward pass to reduce latency; prefer weight-only or W4A16-style recipes when they preserve accuracy.
   - State the deployment format: native framework, compressed-tensors, TensorRT-LLM, etc.
   - Track the real deployment objective: active bytes/token, latency, memory, checkpoint size, or a weighted combination.
   - Include quantization metadata such as scale storage in size estimates.
   - Keep accuracy and verbosity/token usage as separate first-class metrics.

3. **Generate candidates**
   - Start with baselines: BF16/FP16, all-FP8 or W8A8, and one conservative low-bit recipe.
   - If using ModelOpt, start from `modelopt_recipes` rather than inventing patterns from scratch. Prefer model-specific recipes first, then general PTQ presets/fragments.
   - Let the ModelOpt `ptq` skill own checkpoint generation and PTQ validation; use this skill to choose the objective, sequence recipes, compare results, and decide the next iteration.
   - Use AutoQuant or sensitivity tooling for broad search and module ranking.
   - Use manual recipes for controlled ablations by module family.
   - Change one major axis at a time: weight format, activation format, calibration method, excluded modules, backend, or serving flags.

4. **Sanity-gate before full eval**
   - Verify checkpoint export/load.
   - Run short generation probes and inspect the actual output.
   - Check server logs to confirm the intended quantization config, kernel/backend, KV-cache dtype, parser/tool settings, and CUDA graph/eager mode.
   - Fail fast on corrupt output, wrong backend, missing scales, bad tensor naming, or mismatched conversion metadata.

5. **Evaluate in stages**
   - Run cheap screen benchmarks first, chosen to expose likely failure modes for the model/domain.
   - Promote only promising recipes to the full benchmark table.
   - Use consistent sampling, parser, tool-call, token-cap, and backend settings within a comparison table.

6. **Refresh tables**
   - Maintain an accuracy table and a verbosity table.
   - Store output path, evaluator config, server/client logs, checkpoint path, conversion path, git branch/commit, image, and launch command for each row.
   - Mark partial, resumed, failed, and generation-only results explicitly.

7. **Decide next iteration**
   - If accuracy drops, inspect the most sensitive module families first.
   - If verbosity grows, compare parser settings, token caps, failed samples, backend changes, and sampling config before blaming quantization numerics.
   - If AutoQuant produces identical recipes for multiple budgets, inspect recipe hashes and achieved bits; adjust constraints/objective before launching a bigger sweep.

## Practical Defaults

- Ask whether the primary success metric is compute/throughput or memory/latency. Do not assume.
- Prefer active runtime cost over checkpoint size when optimizing routed or sparsely activated models.
- Always compare against BF16/FP16 and a near-lossless FP8/W8A8 baseline.
- Treat benchmark variance as real: run repeat sweeps for close decisions.
- Do not mix parser/no-parser, FP8-KV/BF16-KV, backend, or sampling changes in a single row unless the row name says so.
- Keep a small pipe-clean config for every new task/backend/checkpoint family before launching full runs.

## References

- For recipe design, sensitivity, and active-cost accounting, read `references/recipe_iteration.md`.
- For metric extraction, verbosity, table hygiene, and evaluator resume rules, read `references/eval_tracking.md`.
- For a concrete prior case study, read `references/qwen36_case_study.md` only when Qwen3.5/Qwen3.6 details are relevant.
