# Quantization-Aware Benchmark Recommendations

When evaluating a quantized checkpoint, prioritize benchmarks that are sensitive
to precision loss. The Artificial Analysis (AA) Index v2 suite under
`recipes/tasks/aa/` is the default set for quantized-checkpoint validation.

**Scope rule:**

- **Default quant validation** (when the user just says "evaluate this
  quantized checkpoint"): use the AA suite plus the three always-include
  benchmarks at `recipes/tasks/*.md` (MMLU-Pro, AIME 2025, LiveCodeBench).
- **Explicit AA request** ("AA" / "Artificial Analysis" / "AA Index v2"):
  use **only** `recipes/tasks/aa/`. Do not add the three always-include
  tasks unless the user asks. See the callout at the bottom of this file.

## Available task recipes

| Recipe | Benchmark | What it measures | Quant sensitivity |
|--------|-----------|------------------|-------------------|
| `tasks/mmlu_pro.md` | MMLU-Pro (`ns_mmlu_pro`, nemo-skills, `num_repeats: 1`) | General knowledge (10-choice boxed) | Low — knowledge recall is robust to precision loss; cheap sanity check, not a regression detector |
| `tasks/aime_2025.md` | AIME 2025 (`AIME_2025_aa_v2`, simple-evals) | Competition math (`n_samples: 64`) | High — single-token errors in long chains-of-thought cascade into wrong final answers |
| `tasks/livecodebench.md` | LiveCodeBench v6 (`ns_livecodebench`, nemo-skills) | Code generation (`num_repeats: 8`) | High — code is brittle to single-token errors (one wrong identifier = test failure) |
| `tasks/aa/gpqa_diamond.md` | GPQA Diamond (`ns_gpqa`, nemo-skills, `num_repeats: 16`) | Hard science MCQ (4-choice) | High — MCQ format but answers require multi-step reasoning that quantization can derail |
| `tasks/aa/hle.md` | HLE | Humanity's Last Exam, text-only, judge-scored | High — hard reasoning at the frontier; small precision losses move borderline answers |
| `tasks/aa/lcr.md` | LCR | Long-context reasoning (~120K input, judge-scored) | Very high — KV-cache and attention quant error accumulate across the full context window |
| `tasks/aa/scicode.md` | SciCode | Multi-step scientific code + sandbox execution | Very high — reasoning + code + sandbox stacked; errors compound across subtasks |
| `tasks/aa/ifbench.md` | IFBench | Instruction following | Low — format-compliance is robust; even aggressive FP4 usually shows only small drops |
| `tasks/aa/mmmu_pro.md` | MMMU-Pro | Multimodal reasoning | VLM-only; usually Low/Medium when only the LLM is quantized (vision encoder/adapter typically stay BF16) |
| `tasks/aa/tau2_bench_telecom.md` | Tau2-Bench Telecom | Agentic tool use (user-simulator + judge) | Medium-high — tool-call JSON is brittle, but user-sim + judge variance often dominates the signal |

## Recommended sets by use case

| Use case | Benchmarks |
|----------|-----------|
| Quick sanity check | GPQA |
| Standard quant validation (text LLM) | GPQA, SciCode, LCR |
| AA / Artificial Analysis suite (text LLM) | All `tasks/aa/` text tasks: GPQA, HLE, LCR, SciCode, IFBench, Tau2-Bench Telecom |
| AA / Artificial Analysis suite (multimodal) | AA text suite + MMMU-Pro |
| Code-focused model | LiveCodeBench, SciCode |
| Reasoning model | AIME 2025, GPQA, HLE |

> If the user asks for "AA" or "Artificial Analysis", generate **only** tasks
> under `recipes/tasks/aa/`. Do not silently add MMLU-Pro, AIME 2025, or
> LiveCodeBench — they live at `recipes/tasks/*.md` and are a separate
> always-include set.

## Notes for quantized-checkpoint runs

- **AA-LCR** is the most sensitive task in the set. Include it whenever the
  checkpoint supports the required context length (see the task recipe for
  `--max-model-len 131072`).
- **Repeat / sample counts** in the task recipes are tuned for low variance —
  do **not** lower them for quant comparisons, or noise will mask real
  regressions. The field name differs by harness: `n_samples` for simple-evals
  (AIME `64`) and tau2-bench (Tau2 `8`); `num_repeats` for nemo-skills
  (AA-LCR/GPQA `16`, LiveCodeBench/SciCode `8`, IFBench `5`, MMLU-Pro `1`).
- **Judge / user-simulator endpoints** are required by AA-LCR, HLE AA, and
  Tau2-Bench Telecom. Keep the judge and (for Tau2) user-simulator models
  fixed across baseline and quantized runs for apples-to-apples comparison.
- **IFBench** is the least quant-sensitive in the set but still useful as a
  regression check for aggressive formats (NVFP4, INT4-AWQ).

## How to use

When the user is evaluating a quantized checkpoint, present the recommended set
above and ask which benchmarks to include. If the user already specified a
benchmark list, keep their selection but flag any AA-suite benchmarks they
missed that are commonly used for quant validation. Then read the matching
recipe file(s) before editing the config.
