# Quantization-Aware Benchmark Recommendations

When evaluating a quantized checkpoint, prioritize benchmarks that are sensitive
to precision loss. The Artificial Analysis (AA) Index v2 suite under
`recipes/tasks/aa/` is the default set for quantized-checkpoint validation.
**GDPVal** (`recipes/tasks/gym/gdpval.md`) is also part of the AA suite, but a
different harness (NeMo Gym) — it runs as a **separate standalone config**, never
merged into the `aa/` multi-task list.

**Scope rule:**

- **Default quant validation** (when the user just says "evaluate this
  quantized checkpoint"): use the AA suite — the `aa/` tasks **plus a standalone
  GDPVal config** — plus the three always-include benchmarks at
  `recipes/tasks/*.md` (MMLU-Pro, AIME 2025, LiveCodeBench).
- **Explicit AA request** ("AA" / "Artificial Analysis" / "AA Index v2"):
  use the `aa/` tasks **and** a companion standalone GDPVal config. Do not add
  the three always-include tasks unless the user asks. See the callout at the
  bottom of this file.

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
| `tasks/aa/omniscience.md` | AA-Omniscience | Knowledge reliability (`ns_omniscience`, nemo-skills, `num_repeats: 10`) — correct vs hallucinate vs abstain on obscure facts, judge-scored | Medium — measures the hallucination/abstention balance; aggressive precision loss can erode factual recall and shift the omni-index |
| `tasks/gym/gdpval.md` | GDPVal (`nemo_gym` Stirrup agent, **standalone config**) | Agentic office/PDF deliverables in an Apptainer code-exec sandbox, pairwise/rubric judge | High — long-horizon agentic reasoning + code + judge; precision loss compounds across many turns. **Heaviest task**: multi-hour, often multi-node, needs the SIF sandbox + judge. Runs as its own config, never in the `aa/` list |
| `tasks/gym/mrcr.md` | MRCR (`nemo_gym` simple agent, **standalone config**, **not AA**) | Long-context co-reference retrieval up to 1M tokens; deterministic prefix-gated `SequenceMatcher` grading, stratified by needle count | Very high — the longest-context task available here; KV-cache and attention quant error accumulate over the full window. No judge, so the signal is clean. Opt-in: only when the user asks for MRCR or long-context coverage |

## Recommended sets by use case

| Use case | Benchmarks |
|----------|-----------|
| Quick sanity check | GPQA |
| Standard quant validation (text LLM) | GPQA, SciCode, LCR |
| AA / Artificial Analysis suite (text LLM) | All `tasks/aa/` text tasks: GPQA, HLE, LCR, SciCode, IFBench, Tau2-Bench Telecom, AA-Omniscience — **plus GDPVal** (`tasks/gym/gdpval.md`, a separate standalone config) |
| AA / Artificial Analysis suite (multimodal) | AA text suite (incl. GDPVal) + MMMU-Pro |
| Code-focused model | LiveCodeBench, SciCode |
| Reasoning model | AIME 2025, GPQA, HLE |

> If the user asks for "AA" or "Artificial Analysis", generate the
> `recipes/tasks/aa/` tasks **plus a companion standalone GDPVal config**
> (`recipes/tasks/gym/gdpval.md`) — GDPVal is part of the AA suite but a
> different harness, so it's its own config, never in the `aa/` `tasks` list. Do
> not silently add MMLU-Pro, AIME 2025, or LiveCodeBench — they live at
> `recipes/tasks/*.md` and are a separate always-include set.

## Notes for quantized-checkpoint runs

- **AA-LCR** is the most sensitive task in the set. Include it whenever the
  checkpoint supports the required context length (see the task recipe for
  `--max-model-len 131072`).
- **Repeat / sample counts** in the task recipes are tuned for low variance —
  do **not** lower them for quant comparisons, or noise will mask real
  regressions. The field name differs by harness: `n_samples` for simple-evals
  (AIME `64`) and tau2-bench (Tau2 `8`); `num_repeats` for nemo-skills
  (AA-LCR/GPQA `16`, AA-Omniscience `10`, LiveCodeBench/SciCode `8`, IFBench `5`,
  MMLU-Pro `1`).
- **Judge / user-simulator endpoints** are required by AA-LCR, HLE AA,
  AA-Omniscience, and Tau2-Bench Telecom. Keep the judge and (for Tau2)
  user-simulator models fixed across baseline and quantized runs for
  apples-to-apples comparison.
- **IFBench** is the least quant-sensitive in the set but still useful as a
  regression check for aggressive formats (NVFP4, INT4-AWQ).
- **MRCR** is **not** AA — never add it to an "AA" request. Use it when
  long-context matters: the longest-context task here (1M tokens) and, unlike
  AA-LCR, no judge, so the score is deterministic. Standalone `gym` config
  (`recipes/tasks/gym/mrcr.md`). Two comparability traps: the three variants use
  different datasets and are **not** comparable to each other, and the
  KV dtype must follow each checkpoint's `kv_cache_quant_algo`
  (`hf_quant_config.json`) rather than being pinned to the golden's `fp8` — forcing
  it onto an uncalibrated checkpoint applies KV quantization it was never
  calibrated for. Report the 2/4/8 needle strata — precision
  loss hits the 8-needle stratum while the aggregate still looks flat.
- **GDPVal** is part of the AA suite but the heaviest task and a separate
  harness: it runs as its **own standalone `gym` config** (never in the `aa/`
  `tasks` list), needs the Apptainer SIF sandbox + judge, and is multi-hour /
  often multi-node. Generate it alongside the `aa/` config; see
  `recipes/tasks/gym/gdpval.md` + `references/gym-gdpval.md`. Thinking mode is
  mandatory (non-thinking loses ~86% of pairwise judgements). `num_repeats` is **1** —
  the value both current goldens use, already set by the template; do not raise it.

## How to use

When the user is evaluating a quantized checkpoint, present the recommended set
above and ask which benchmarks to include. If the user already specified a
benchmark list, keep their selection but flag any AA-suite benchmarks they
missed that are commonly used for quant validation. Then read the matching
recipe file(s) before editing the config.
