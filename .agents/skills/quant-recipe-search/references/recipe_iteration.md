# Recipe Iteration Reference

## Problem

Quantization recipe search is a constrained optimization loop:

- Improve the user's chosen objective: compute/throughput, memory/latency, or a
  custom score.
- Keep every benchmark within the accepted accuracy-loss threshold. Default:
  less than 1 percentage point versus the matching BF16/FP16 baseline.
- Keep accuracy, verbosity/token usage, runtime behavior, and active cost
  separate until the final decision.
- Treat generated checkpoints as candidates. Evaluation and comparison decide
  whether a candidate is useful.

Ask for missing objective, primary quantization family, and benchmark set before
planning candidates.

## Search Space

Write the search space before launching PTQ. A recipe is a combination of these
axes, not just a numeric format.

| Axis | Choices To Try | Notes |
| --- | --- | --- |
| Numeric format | FP8/W8A8, NVFP4/W4A4, W4A16 NVFP4, INT4/AWQ, mixed formats | Keep FP8/W8A8 as a near-lossless baseline unless it is the user's target. |
| Calibration/search algorithm | Max, MSE, GPTQ, AWQ, AutoQuant scoring, calibration data variants | Algorithm choice is independent from numeric format. |
| Selection method | Manual/heuristic, sensitivity-guided manual, AutoQuant, hybrid | Record how each candidate was selected. |
| Module family | Attention, MLP, MoE experts, routers/gates, embeddings, `lm_head`, adapters, vision encoders | Change one major family at a time for ablations. |
| Runtime constraints | Fused attention groups, fused MoE expert projections, backend-supported formats | Do not mix incompatible quantization inside a fused runtime group. |
| Calibration budget | Dataset mix, sample count, sequence length, batch size | Vary deliberately and record the budget. |

### Objective Axis

Use the user's objective to decide which candidates are worth testing.

- **Compute / throughput:** typical data-center target. Favor activation
  quantization such as NVFP4 or FP8 when the runtime has fast kernels.
- **Memory / latency:** typical edge or memory-pressure target. Favor W4A16 or
  weight-only recipes when they preserve accuracy. Prefer active bytes per
  forward/decode path over total checkpoint size for routed or sparse models.
- **Custom:** use the user-provided score, for example checkpoint size, latency
  at a fixed concurrency, or product-specific memory budget.

If the user needs multiple objectives, maintain separate tables or define an
explicit weighted score.

### Numeric Format Axis

Common starting families:

- FP8/W8A8: near-lossless baseline or explicit primary target.
- NVFP4/W4A4: low-bit candidate family when activation quantization is part of
  the target.
- W4A16 NVFP4: weight-only NVFP4 family for accuracy-preserving memory/latency
  searches.
- INT4/AWQ: weight-only low-bit family for low-batch memory/latency targets.
- Mixed formats: examples include NVFP4+FP8, W4A16 NVFP4 with FP8 attention, or
  model-specific recipe fragments.

KV-cache dtype, parser settings, token caps, and backend flags are runtime/eval
controls unless the user explicitly makes them part of the recipe objective.

### Calibration Algorithm Axis

Try calibration/search algorithms as independent recipe variants:

- Max calibration: fast baseline for many FP8/NVFP4 formats.
- MSE calibration: try when max calibration loses accuracy for low-bit weights
  or sensitive layers.
- GPTQ: try for weight-quantized candidates where correction cost is acceptable.
- AWQ: try for INT4 or other weight-only candidates.
- AutoQuant scoring: use KL-divergence or gradient-based scoring when available
  to rank layers/modules and produce sensitivity reports.

### Selection Method Axis

Choose which modules get which format by one of these methods:

- Manual/heuristic: use prior experience, module-family cost, and controlled
  ablations.
- Sensitivity-guided manual: generate or recover an AutoQuant sensitivity report,
  then protect sensitive families or quantize low-sensitivity high-cost families.
- AutoQuant: search per-layer/per-module selections under constraints such as
  target bits, active-cost objective, or allowed formats. When AutoQuant is
  available, include at least one AutoQuant-generated candidate in the portfolio
  so its trade-off can be compared against manual recipes.
- Hybrid: start from AutoQuant, then override known runtime constraints or
  known-sensitive fused groups manually.

## Design Workflow

1. Recover existing evidence:
   - Result tables, checkpoints, recipe logs, AutoQuant states, sensitivity
     reports, and active jobs.
   - Use `monitor`, `launching-evals`, and `compare-results` for execution
     state and metric provenance.

2. Define the target:
   - Objective, primary quantization family, benchmark set, accuracy-loss
     threshold, cost metric, and calibration budget.
   - Include scale storage and other quantization metadata in cost estimates.

3. Pick baselines:
   - BF16/FP16 baseline.
   - FP8/W8A8 near-lossless baseline, unless FP8 is the final target.
   - Existing production recipe if one exists.

4. Pick first candidates:
   - Start from `modelopt_recipes` when ModelOpt is available.
   - Prefer model-specific recipes, then general PTQ presets, then recipe
     fragments.
   - Add one AutoQuant candidate in the requested primary family when AutoQuant
     is available. Treat it as the expected best-search path, but validate it.
   - Add at least one manual or sensitivity-guided candidate for comparison and
     as a fallback if AutoQuant misses the benchmark frontier or produces a
     runtime-incompatible recipe.

5. Generate and validate:
   - Delegate checkpoint generation and validation to `ptq`.
   - Check checkpoint coverage, quantization metadata, and expected module
     coverage.
   - Pipe-clean serving only after checkpoint validation passes.

6. Scale evaluation:
   - Run cheap screen benchmarks first.
   - Expand only candidates that pass screen evals and runtime gates.

## Runtime Fusion Rules

Search by module family, but respect modules fused by the target runtime.

- vLLM Qwen linear attention can fuse `linear_attn.in_proj_qkv` and
  `linear_attn.in_proj_z` into `linear_attn.in_proj_qkvz`; do not mix formats or
  algorithms across those shards unless the runtime supports it.
- Fused MoE kernels can couple expert projections such as gate/up (`w1`/`w3`, or
  equivalent names); treat each fused expert group as one recipe unit unless
  deployment confirms mixed formats are supported.
- If a checkpoint is valid but deployment fails due to missing support, classify
  it as checkpoint-quality, recipe/runtime compatibility, or deployment
  implementation. For deployment implementation, try small patches or flags via
  `deployment` / `debug` before rejecting the recipe.

## Iteration Loop

Use this loop after each candidate:

1. Update the portfolio table with recipe axes, active cost, checkpoint path,
   eval logs, accuracy, verbosity, and decision.
2. Compare against BF16/FP16 and FP8/W8A8 baselines.
3. If accuracy drops:
   - Protect sensitive module families.
   - Try MSE, GPTQ, or AWQ variants.
   - Use AutoQuant sensitivity to choose manual overrides.
4. If performance or active cost is insufficient:
   - Quantize the next high-cost active family.
   - Try a more aggressive format.
   - Revisit the active-cost objective or AutoQuant constraints.
5. If verbosity changes:
   - Inspect output samples and generation stats.
   - Verify parser, token cap, sampling, backend, and KV-cache settings did not
     change.
6. If results are close or noisy:
   - Rerun before labeling a benchmark regression.
7. If AutoQuant gives repeated recipes:
   - Check achieved bits and recipe hashes.
   - Adjust objective, allowed formats, or constraints before larger sweeps.
8. If AutoQuant underperforms manual recipes:
   - Compare the AutoQuant sensitivity report against manual ablation results.
   - Check whether AutoQuant protected high-active-cost modules, excluded the
     wrong families, optimized checkpoint size instead of active cost, or hit
     runtime-fusion constraints.
   - Keep the manual recipe in the table and use AutoQuant sensitivity to design
     the next hybrid/manual candidate.

Promote a recipe only when validated comparison shows it satisfies the user's
objective and benchmark threshold.

## Delegating To Existing Skills

Do not reimplement workflows that existing skills own:

| Need | Use |
| --- | --- |
| Generate/check a quantized checkpoint | `ptq` |
| Serve a checkpoint or test backend flags | `deployment` |
| Create or submit NEL configs | `evaluation` |
| Resume/debug/analyze live eval runs | `launching-evals` |
| Track active Slurm/NEL jobs | `monitor` |
| Fetch MLflow artifacts | `accessing-mlflow` |
| Compute baseline-vs-candidate deltas | `compare-results` |

Before launching PTQ in a ModelOpt repo, read the current PTQ skill from
`.agents/skills/ptq/SKILL.md`; recipe paths and validation gates can change.

## ModelOpt Starting Points

When ModelOpt is available, start from `modelopt_recipes`:

1. Check model-specific recipes first, for example
   `modelopt_recipes/huggingface/<model_family>/ptq/`.
2. Check general PTQ recipes and presets.
3. Use recipe fragments to build controlled manual variants.
4. Summarize include/exclude coverage before calibration. If a pattern misses the
   intended layer family, fix the recipe before launching.

Useful starting candidates:

- Compute/throughput: FP8/W8A8, NVFP4/W4A4, mixed NVFP4+FP8 with activation
  quantization.
- Memory/latency: W4A16 NVFP4, weight-only NVFP4, or W4A16 mixed with FP8 for
  sensitive modules.
- MoE: experts-only or MLP-only recipes, then expand based on sensitivity and
  active-routing cost.

## Candidate Record

For every candidate, record:

- Objective and acceptance threshold.
- Numeric formats and module-family coverage.
- Calibration/search algorithm and calibration data budget.
- Selection method: manual, sensitivity-guided, AutoQuant, or hybrid.
- Whether the candidate came from AutoQuant, manual ablation, or a hybrid
  override, so AutoQuant and manual trade-offs can be compared directly.
- Runtime fusion assumptions.
- Active bytes/token estimate including scales.
- Checkpoint path and eval/log paths.
- Accuracy and verbosity metrics.
- Decision and next action.
