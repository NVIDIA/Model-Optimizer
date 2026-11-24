# Quantization-Aware Benchmark Choice

When evaluating a quantized checkpoint, prioritize benchmarks that are sensitive to precision loss. The
Artificial Analysis (AA) Index v2 suite is the default set. GDPVal is an AA member that this workflow does not
cover: use the set below for relative baseline-versus-candidate comparison and report per-task scores, not an
aggregate index matched against a published AA Index. When a comparison is framed against a published AA
Index, say that the agentic-deliverables dimension is missing instead of presenting the remaining tasks as
the full suite.

**Scope rule**

- **Default quantization validation** (the user just says "evaluate this quantized checkpoint"): the AA suite,
  plus the three always-include benchmarks MMLU-Pro, AIME 2025 and LiveCodeBench.
- **Explicit AA request** ("AA", "Artificial Analysis", "AA Index v2"): the AA tasks only. Do not add the three
  always-include tasks unless the user asks.

## Benchmarks and how sensitive they are

| Benchmark | What it measures | Quantization sensitivity |
|---|---|---|
| MMLU-Pro | General knowledge (10-choice) | Low. Knowledge recall is robust to precision loss; a cheap sanity check, not a discriminator. |
| AIME 2025 | Competition math | High. Single-token errors in long chains of thought cascade into wrong final answers. |
| LiveCodeBench v6 | Code generation | High. Code is brittle to single-token errors (one wrong identifier fails the test). |
| GPQA Diamond | Hard science, 4-choice | High. The answer needs multi-step reasoning that quantization can disturb. |
| HLE (text-only) | Humanity's Last Exam, judge-scored | High. Hard reasoning at the edge of model ability; small precision losses move borderline answers. |
| AA-LCR | Long-context reasoning (about 120K input, judge-scored) | Very high. KV-cache and attention quantization error accumulates across the context window. |
| SciCode | Multi-step scientific code with sandbox execution | Very high. Reasoning, code and sandbox are stacked; errors compound across subtasks. |
| IFBench | Instruction following | Low. Format compliance is robust; even aggressive FP4 usually shows only small drops. |
| MMMU-Pro | Multimodal reasoning | VLM only; usually low or medium when only the language model is quantized (the vision encoder and adapter typically stay BF16). |
| Tau2-Bench Telecom | Agentic tool use (user simulator and judge) | Medium to high, but simulator and judge variance often dominates the signal. |
| AA-Omniscience | Knowledge reliability: correct, hallucinate or abstain on obscure facts, judge-scored | Medium. |
| MRCR | Long-context co-reference retrieval up to 1M tokens, deterministic grading | Not an AA task; never add it to an "AA" request. Use it when long context matters. |

## Recommended sets by use case

| Use case | Benchmarks |
|---|---|
| Quick sanity check | GPQA |
| Standard quantization validation (text model) | GPQA, SciCode, AA-LCR |
| AA suite (text model) | GPQA, HLE, AA-LCR, SciCode, IFBench, Tau2-Bench Telecom, AA-Omniscience |
| AA suite (multimodal) | the AA text suite plus MMMU-Pro |
| Code-focused model | LiveCodeBench, SciCode |
| Reasoning model | AIME 2025, GPQA, HLE |

## Notes for quantized-checkpoint runs

- **AA-LCR** is the most sensitive task in the set. Include it whenever the checkpoint supports the required
  context length (a maximum model length of 131072).
- **Do not lower repeat or sample counts** for quantization comparisons: fewer repeats let noise mask real regressions.
  The values are the tool's benchmark configuration.
- **Judge and user-simulator endpoints** (AA-LCR, HLE, AA-Omniscience, Tau2-Bench Telecom) must stay fixed
  across the baseline and quantized runs. A judge that is retired or replaced
  makes earlier scores from the old judge unusable as a baseline: rerun both sides with the new judge, and treat the
  result as a new baseline, not as comparable to scores from the old one.
- **MRCR traps:** the KV-cache
  dtype must follow each checkpoint's `kv_cache_quant_algo` (`hf_quant_config.json`) instead of being pinned,
  because forcing it onto an uncalibrated checkpoint applies a quantization it was never calibrated for; and
  report the 2, 4 and 8 needle strata, since precision loss hits the 8-needle stratum while the aggregate still
  looks flat.

## How to use

Present the recommended set and ask which benchmarks to include. If the user already named a list, keep it but
flag any AA-suite benchmarks they missed that are commonly used for quantization validation. Then read the
evaluation tool's own benchmark configuration for the selected ones before composing a run; the tool's reference
says where its benchmark recipes are.
