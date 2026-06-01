# Recipe Iteration Reference

## Candidate Types

- **Baselines:** BF16/FP16, all-FP8/W8A8, and existing production recipe.
- **Auto search:** AutoQuant/PTQ search over formats and excluded modules.
- **Manual ablation:** explicit module-family recipes, changing one major choice at a time.
- **Calibration variants:** MSE, GPTQ, KL-div, gradient/search-based, calibration dataset changes.
- **Serving variants:** CT/native format, kernel/backend, KV-cache dtype, CUDA graph/eager, parser/tool-call settings.

## Target Selection

Ask the user what success means before designing recipes.

### Compute / Throughput

Use this default for data-center serving when the workload is compute bound, especially at large batch size.

- Favor activation quantization with fast kernels, such as NVFP4 or FP8.
- Measure throughput and latency at the target batch/concurrency, not just checkpoint size.
- Keep memory accounting, but do not let weight-only memory savings override a throughput regression.
- Sanity-check backend selection because a recipe only helps compute if the serving stack uses the intended kernels.

### Memory / Latency

Use this default for edge or memory-pressure scenarios where the goal is minimizing activated memory per forward pass.

- Favor W4A16 or weight-only recipes when they preserve accuracy.
- `w4a16_nvfp4` is a strong default candidate for preserving accuracy while reducing active weight traffic.
- FP8 is usually close to lossless and can still help prefill or sensitive modules.
- Weight-only NVFP4 can give extra headroom to quantize more layers than W4A4 activation-quantized recipes.
- Compare active bytes per forward/decode path with scale storage included.

If the user needs both, keep two tables or explicitly define a weighted score.

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

This skill should provide the experiment strategy and next-candidate decision,
then delegate execution to the appropriate skill.

## Working With The ModelOpt PTQ Skill

This skill should orchestrate the recipe loop; the ModelOpt `ptq` skill should produce and validate checkpoints.

Use the PTQ skill for:

- Environment/workspace setup.
- Model support checks.
- Running `hf_ptq.py` or launcher jobs.
- Using `--qformat` or `--recipe`.
- Post-quantization checkpoint validation.

Use this skill for:

- Choosing the success metric.
- Selecting baseline recipes and manual deltas.
- Deciding which evals to run next.
- Maintaining the high-level recipe portfolio table.
- Interpreting sensitivity and benchmark tradeoffs.

Before launching PTQ in a ModelOpt repo, read the current PTQ skill from `.claude/skills/ptq/SKILL.md`; if possible, compare it with latest `origin/main` because recipe paths and validation gates may change.

## ModelOpt Recipe Baselines

When ModelOpt is available, start from `modelopt_recipes`:

1. Check model-specific recipes first, for example `modelopt_recipes/huggingface/<model_family>/ptq/` or `modelopt_recipes/models/<model>/`.
2. Check general recipes such as `modelopt_recipes/general/ptq/`.
3. Check current preset sources under `modelopt_recipes/configs/ptq/presets/model/`, especially when the workflow uses `--qformat`.
4. Use recipe fragments under `modelopt_recipes/configs/ptq/units/` to build controlled manual variants.
5. Summarize include/exclude pattern coverage before calibration. If a pattern misses the intended layer family, fix the recipe before launching.

Useful starting points by target:

- Compute/throughput: FP8/W8A8, NVFP4/W4A4, mixed NVFP4+FP8 recipes with activation quantization.
- Memory/latency: `w4a16_nvfp4`, weight-only NVFP4, or W4A16 mixed with FP8 for sensitive modules.
- MoE: experts-only or MLP-only recipes, then expand based on sensitivity and active-routing cost.

## AutoQuant Checklist

- Save the AutoQuant state/checkpoint so sensitivity data can be recovered later.
- Record requested target bits and achieved effective bits separately.
- Hash or diff the resulting quantized-layer recipe; different target labels may still produce identical recipes.
- Inspect which modules the search excludes or protects. Search objectives can optimize checkpoint size while missing active runtime cost.
- Generate a sensitivity report from the saved state when deciding which manual recipe to try next.

## Manual Recipe Checklist

For each candidate, record:

- Module families and quant format: experts/MLP, attention, linear attention, embeddings, `lm_head`, routers/gates, vision encoder, adapters.
- Activation format and whether it is W4A16, W4A8, W8A8, or weight-only.
- Explicit exclusions and why they are excluded.
- Calibration method, dataset, sample count, and batch settings.
- Active bytes/token estimate including scales.
- Checkpoint path, converted checkpoint path, and serving backend.

Build manual candidates as controlled deltas:

1. Start from a known-good baseline.
2. Quantize the largest active-cost family that sensitivity suggests is safe.
3. Evaluate screen benchmarks.
4. Add or remove one module family.
5. Recompute active cost and rerun the same screen.

## Active-Cost Accounting

Checkpoint size is not always the right objective.

- Include weight storage and quantization scales.
- Use active routing ratio for MoE or sparse modules.
- Separate decode active bytes/token from total checkpoint bytes.
- Keep embeddings and prefill-specific costs separate when they do not affect the target path.
- Report assumptions next to the number.

## Sensitivity Interpretation

- A low sensitivity score is a hint, not proof. Validate with benchmarks.
- If a benchmark drops while aggregate sensitivity looked safe, run module-family ablations.
- If a candidate has good accuracy but high verbosity, inspect output samples and generation stats before accepting it.
- If serving output is corrupt, prioritize loader/export/backend issues before recipe tuning.
