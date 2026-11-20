# Quantization-Aware KTO (QAKTO): Reducing Doomed Reasoning Loops While Preserving Accuracy

[Daniel Korzekwa](https://github.com/danielkorzekwa), [Keval Morabia](https://github.com/kevalmorabia97), [Sepehr Sameni](https://github.com/Separius) · November 19, 2026

## Key takeaways

- **Cap-hit rates fell on seven of eight benchmarks across the reported evaluations:** SciCode (10.23% → 0% observed on held-out error-prone subtasks; −1.53 pp in broader evaluation), AA-Omniscience (−0.30 pp), GPQA (−0.06 pp), IFBench (−1.42 pp), MMMU-Pro (−0.25 pp), MMLU-Pro (−0.029 pp), and tau2-bench Telecom (−4.09 pp). AA-LCR stayed at zero.
- **Mean benchmark scores increased on seven of eight benchmarks** and were unchanged on AA-LCR. The increases were not statistically significant.
- **Mean completion tokens fell on seven of eight benchmarks:** SciCode (−53.0%), AA-Omniscience (−15.6%), IFBench (−13.5%), AA-LCR (−1.6%), MMMU-Pro (−7.2%), tau2-bench Telecom (−22.5%), and MMLU-Pro (−2.1%). GPQA increased 14.1%.
- **Missing-final-answer counts fell on all seven single-response benchmarks:** SciCode, AA-Omniscience, GPQA Diamond, IFBench, AA-LCR, MMMU-Pro, and MMLU-Pro.

## Contents

- [Introduction](#introduction)
- [Related work](#related-work)
- [KTO method](#kto-method)
- [Experiments and results](#experiments-and-results)
- [Conclusions and Future work](#conclusions-and-future-work)

## Introduction

In a doomed reasoning loop, the model repeats itself without progressing toward a final answer. The NVFP4 Qwen3.6-35B-A3B checkpoint can enter these loops after quantization-aware distillation (QAD), notably on SciCode and IFBench. The [parent tutorial’s output-length analysis](../README.md#output-length) describes the SciCode behavior. We use this checkpoint as our QAD baseline and fine-tune it with quantization-aware [Kahneman–Tversky Optimization](https://arxiv.org/abs/2402.01306) (QAKTO). Training encourages correct responses and discourages incorrect responses stuck in reasoning loops, aiming to reduce looping while preserving accuracy.

Across the reported evaluations, cap-hit rates fell on seven of eight benchmarks, including 10.23% → 0% on the held-out error-prone SciCode subtasks. Mean benchmark scores increased on seven of eight benchmarks, without statistically significant gains or detected accuracy regressions. Mean completion tokens fell on seven of eight benchmarks, including SciCode (−53%) and tau2-bench Telecom (−22.5%), and missing-final-answer counts fell on all seven single-response benchmarks. See the [results](#experiments-and-results) for benchmark-level changes.

## Related work

### Decoding penalties and response-only training

The [parent tutorial’s output-length analysis](../README.md#output-length) shows that quantization increased SciCode mean completion length by roughly 23–25% before QAD; after QAD, responses were 90.9% longer on average than those from the original, unquantized BF16 model. Capped responses consumed 45.7% of completion tokens. A decoding presence penalty of 1.5 reduced cap hits from 96 to 2 out of 2,704 responses, with accuracy changes within evaluation noise, but reduced generation throughput 4.8× on GPQA Diamond.

The [QAD ablations](../ABLATIONS.md#scicode-non-termination-diagnosis-and-decode-time-fix) also tried computing the training loss only on response tokens, excluding prompt tokens. This did not provide a lasting reduction in cap hits: 5.7% versus 3.6% when training on all tokens.

### Earlier attempts related to KTO

An initial QAD fine-tuning pilot used completed responses generated with anti-repetition instructions. On held-out, failure-prone SciCode subtasks, cap-hit rates fell from 10.91% to 6.36%. Using teacher-generated data reduced the rate to 5.45%. A follow-up combined QAD fine-tuning with quantization-aware DPO (QADPO), reducing observed SciCode cap hits to zero while accuracy fell from 43.20% to 39.64%.

This tutorial focuses on preserving accuracy while reducing cap hits, using QAKTO directly from the QAD baseline.

### Related research

- **Quantization-aware preference training:** [Lee et al. (ACL 2024)](https://aclanthology.org/2024.acl-long.612/) introduced QDPO to recover conversational quality after quantization. It keeps quantization active during DPO training and pairs full-precision outputs as preferred responses with quantized-model outputs as rejected responses. QAKTO fine-tunes the QAD baseline with quantization active, using independently labeled correct and looping responses without full-precision teacher outputs.
- **Decoding penalties:** [Lotfi et al.](https://arxiv.org/abs/2606.00206) penalize selected hesitation markers to reduce quantization-related overthinking. [RATIO](https://arxiv.org/abs/2609.39801) compares the quantized model with its original, unquantized version to identify tokens associated with overthinking and set a decoding penalty for each.
- **Targeted anti-loop training:** [AntiDoom](https://www.liquid.ai/blog/antidoom) uses Final Token Preference Optimization (FTPO), usually with LoRA, to prefer alternative tokens where repetition starts. It needs to locate that boundary; QAKTO uses labels for whole responses and, in this experiment, full-parameter fine-tuning with quantization active. Its [automated loop-mining pipeline](https://github.com/Liquid4All/antidoom) could help collect QAKTO training data. Other ideas to test include LoRA to reduce training memory and limiting negative supervision to the repeating part of a response.
- **Learning self-correction:** [RISE](https://arxiv.org/abs/2407.18219) trains models to revise unsuccessful answers across turns. [SCoRe](https://arxiv.org/abs/2409.12917) uses reinforcement learning on model-generated correction traces. Both motivate training a model to recover after a loop has begun.
- **Training against repetition:** [Unlikelihood training](https://arxiv.org/abs/1908.04319) lowers the probability of unwanted tokens and sequences; [DPO](https://arxiv.org/abs/2305.18290) learns from preferred and rejected response pairs. These provide related ways to train on completed versus looping continuations.

## KTO method

### Why train on correct, unpaired responses?

A correct answer to the exact question that triggered a loop can be difficult to collect. [DPO](https://arxiv.org/abs/2305.18290) requires a preferred and a rejected response to the same question. [KTO](https://arxiv.org/abs/2402.01306) learns from correct answers and looping failures that can come from different questions, so it does not require same-question preference pairs.

### How we trained QAKTO?

We selected 55 error-prone SciCode subtasks where the QAD baseline had previously reached the output-token limit, then split them into 44 for training and 11 for held-out evaluation. Training used **104 correct model responses** and **104 reasoning-loop prefixes**. Broader SciCode evaluation excludes the 44 training subtasks, leaving 294 of 338.

```text
QAD baseline
                    ↓
QAKTO training using SciCode responses:
  Desirable: 104 correct responses
  Undesirable: 104 reasoning-loop prefixes
                    ↓
NVFP4 export → evaluation
```

We fine-tune the QAD baseline with quantization active. KTO learns from response likelihoods, encouraging correct answers and discouraging looping responses. No separate SFT cross-entropy or teacher-logit loss is added.

For a batch of *B* labeled responses, the implemented loss is:

$$
\mathcal{L}_{\mathrm{QAKTO}}(\theta) = \frac{1}{B}\sum_{i=1}^{B}\left[1-\sigma\!\left(\beta s_i\left[\log\frac{\pi_\theta(y_i \mid x_i)}{\pi_{\mathrm{ref}}(y_i \mid x_i)}-z\right]\right)\right].
$$

Here $(x_i, y_i)$ is a question and response; $s_i=+1$ for correct responses and $s_i=-1$ for looping prefixes. $\pi_\theta$ is the model being fine-tuned, $\pi_{\mathrm{ref}}$ the frozen QAD baseline, $\sigma$ the sigmoid, and $\beta=0.05$. Log probabilities sum over response tokens.

The shared reference level *z* is a batch KL estimate of the model’s overall likelihood shift from the frozen baseline. Each response is judged relative to that shift. During each update, *z* is treated as constant when computing gradients. See the [KTO paper](https://arxiv.org/html/2402.01306v5#S4.SS1.SSS0.Px1) for details.

## Experiments and results

We evaluated two NVFP4 Qwen3.6-35B-A3B checkpoints: the QAD baseline and QAKTO. We compared accuracy, cap-hit rates, missing final answers, abstention, and completion token use.

See [Reproducibility](reproducibility/README.md) for data collection, training, export and evaluation instructions.

### Evaluation data

| Benchmark | Questions, subtasks, or tasks × repetitions | Responses or conversations per model |
| --- | --- | --- |
| SciCode, 294 subtasks | 294 × 8 | 2,352 |
| AA-Omniscience | 600 × 10 | 6,000 |
| GPQA Diamond | 198 × 16 | 3,168 |
| IFBench | 300 × 16 | 4,800 |
| AA-LCR | 100 × 8 | 800 |
| MMMU-Pro | 1,730 × 16 | 27,680 |
| MMLU-Pro | 12,032 × 2 | 24,064 |
| tau2-bench Telecom | 114 × 9 | 1,026 |

### Benchmark accuracy

We compare benchmark scores below; changes are QAKTO − QAD baseline, in percentage points (pp).

| Benchmark | QAD baseline | QAKTO | Difference [95% CI], pp | QAKTO higher, p | QAD baseline higher, p (context) |
| --- | --- | --- | --- | --- | --- |
| SciCode, 294 subtasks | 42.81% | 44.39% | +1.57 [−0.48, +3.67] | 0.080 | 0.931 |
| AA-Omniscience | 21.62% | 21.68% | +0.07 [−0.85, +0.98] | 0.443 | 0.557 |
| GPQA Diamond | 83.40% | 83.68% | +0.28 [−0.96, +1.53] | 0.327 | 0.673 |
| IFBench | 59.06% | 60.05% | +0.98 [−0.17, +2.14] | 0.047 | 0.953 |
| AA-LCR | 68.50% | 68.50% | 0.00 [−3.73, +3.73] | 0.500 | 0.500 |
| MMMU-Pro | 73.76% | 73.87% | +0.12 [−0.28, +0.51] | 0.283 | 0.717 |
| MMLU-Pro | 84.67% | 84.74% | +0.07 [−0.25, +0.38] | 0.339 | 0.661 |
| tau2-bench Telecom | 94.83% | 95.42% | +0.58 [−1.12, +2.29] | 0.250 | 0.750 |

QAKTO’s mean scores were higher on seven benchmarks and unchanged on AA-LCR. The increases were not statistically significant.

### Check for regressions

The table reports one-sided non-inferiority p-values.

| Benchmark | 2.0 pp | 1.0 pp | 0.5 pp | 0.1 pp |
| --- | --- | --- | --- | --- |
| SciCode, 294 subtasks | <0.001 | 0.009 | 0.028 | 0.060 |
| AA-Omniscience | <0.001 | 0.011 | 0.112 | 0.360 |
| GPQA Diamond | <0.001 | 0.022 | 0.108 | 0.272 |
| IFBench | <0.001 | <0.001 | 0.006 | 0.033 |
| AA-LCR | 0.145 | 0.298 | 0.395 | 0.479 |
| MMMU-Pro | <0.001 | <0.001 | 0.001 | 0.142 |
| MMLU-Pro | <0.001 | <0.001 | <0.001 | 0.149 |
| tau2-bench Telecom | 0.002 | 0.034 | 0.106 | 0.215 |

At p < 0.05 per comparison, these exploratory tests rule out a 1 pp accuracy loss on seven of eight benchmarks. AA-LCR remains inconclusive at every tested margin.

### Assess model reliability

#### Cap-hit rates

A cap hit reaches the output-token limit; it screens for potential loops without confirming repetition. Rates count responses, except tau2-bench Telecom counts conversations with a capped agent turn.

| Benchmark | QAD baseline | QAKTO | Difference [95% CI], pp | QAKTO lower, p | QAKTO higher, p (context) |
| --- | --- | --- | --- | --- | --- |
| SciCode, 294 subtasks | 1.53% (36/2,352) | 0.00% (0/2,352) | −1.53 [−2.13, −1.02] | <0.001 | 1.000 |
| SciCode, held-out error-prone (11 subtasks) | 10.23% (9/88) | 0.00% (0/88) | −10.23 [−14.77, −5.68] | 0.004 | 1.000 |
| AA-Omniscience | 0.50% (30/6,000) | 0.20% (12/6,000) | −0.30 [−0.50, −0.12] | 0.002 | 0.999 |
| GPQA Diamond | 0.22% (7/3,168) | 0.16% (5/3,168) | −0.06 [−0.25, +0.09] | 0.375 | 0.844 |
| IFBench | 6.79% (326/4,800) | 5.38% (258/4,800) | −1.42 [−2.21, −0.65] | <0.001 | 1.000 |
| AA-LCR | 0.00% (0/800) | 0.00% (0/800) | No observed events | 1.000 | 1.000 |
| MMMU-Pro | 0.35% (98/27,680) | 0.10% (28/27,680) | −0.25 [−0.35, −0.16] | <0.001 | 1.000 |
| MMLU-Pro | 0.058% (14/24,064) | 0.029% (7/24,064) | −0.029 [−0.066, +0.004] | 0.084 | 0.968 |
| tau2-bench Telecom, NEL (retries may hide cap hits) | 0.00% (0/1,026) | 0.00% (0/1,026) | No observed events | 1.000 | 1.000 |
| tau2-bench Telecom, custom runner | 5.17% (53/1,026) | 1.07% (11/1,026) | −4.09 [−5.85, −2.44] | <0.001 | 1.000 |

QAKTO had lower observed cap-hit rates on seven of eight benchmarks across the reported evaluations, with zero observed caps on SciCode.

For tau2-bench Telecom, NEL can retry capped responses that contain only reasoning and save only the successful retry, so saved conversations may omit capped attempts. The table also shows results from an earlier custom-runner evaluation that counts capped attempts instead of replacing them with successful retries.

#### Missing final answers

| Benchmark | QAD baseline | QAKTO | Difference [95% CI], pp | QAKTO lower, p | QAKTO higher, p (context) |
| --- | --- | --- | --- | --- | --- |
| SciCode, 294 subtasks | 1.62% (38/2,352) | 0.04% (1/2,352) | −1.57 [−2.17, −1.02] | <0.001 | 1.000 |
| AA-Omniscience | 0.50% (30/6,000) | 0.20% (12/6,000) | −0.30 [−0.50, −0.12] | 0.002 | 0.999 |
| GPQA Diamond | 0.22% (7/3,168) | 0.16% (5/3,168) | −0.06 [−0.25, +0.09] | 0.375 | 0.844 |
| IFBench | 6.73% (323/4,800) | 5.31% (255/4,800) | −1.42 [−2.21, −0.65] | <0.001 | 1.000 |
| AA-LCR | 0.12% (1/800) | 0.00% (0/800) | −0.12 [−0.38, 0.00] | 0.500 | 1.000 |
| MMMU-Pro | 0.33% (92/27,680) | 0.11% (30/27,680) | −0.22 [−0.32, −0.14] | <0.001 | 1.000 |
| MMLU-Pro | 0.058% (14/24,064) | 0.029% (7/24,064) | −0.029 [−0.066, +0.004] | 0.084 | 0.968 |

Missing-final-answer counts were lower on all seven benchmarks, falling from 38 to 1 on SciCode.

#### AA-Omniscience abstention rate

We count a response as an abstention when its final answer is non-empty and the judge labels it `NOT_ATTEMPTED`. This category can include uncertain guesses. Following the [evaluation guide](https://github.com/NVIDIA/Model-Optimizer/blob/c8559d2b4e547960db1713640d3d54bb9b7c8280/docs/source/announcements/evaluating_models_efficiently_during_optimization_experiments/index.rst#abstention-ratio), the abstention rate is `abstentions / (all responses − correct responses)`. Incorrect, partial and missing answers stay in the denominator.

| Measure | QAD baseline | QAKTO | Difference [95% CI], pp | QAKTO higher, p | QAKTO lower, p (context) |
| --- | --- | --- | --- | --- | --- |
| Abstention rate (among non-correct responses) | 33.30% (1,566/4,703) | 35.18% (1,653/4,699) | +1.88 [+0.32, +3.45] | 0.010 | 0.990 |

The abstention rate among non-correct responses rose by 1.88 pp with QAKTO (p=0.010).

### Inference efficiency

| Benchmark | QAD baseline mean / median | QAKTO mean / median | Mean difference [95% CI], tokens | QAKTO shorter, p | QAKTO longer, p (context) |
| --- | --- | --- | --- | --- | --- |
| SciCode, 294 subtasks | 6,512 / 2,993 | 3,063 / 1,803 | −3450 [−4236, −2664] | <0.001 | 1.000 |
| AA-Omniscience | 2,865 / 1,944 | 2,418 / 1,848 | −447 [−707, −187] | <0.001 | 1.000 |
| GPQA Diamond | 11,088 / 5,821 | 12,654 / 6,419 | +1566 [+982, +2150] | 1.000 | <0.001 |
| IFBench | 14,581 / 4,568 | 12,609 / 4,425 | −1972 [−2923, −1022] | <0.001 | 1.000 |
| AA-LCR | 2,769 / 2,454 | 2,724 / 2,410 | −45 [−134, +44] | 0.158 | 0.842 |
| MMMU-Pro | 8,903 / 5,358 | 8,259 / 4,897 | −643 [−770, −517] | <0.001 | 1.000 |
| MMLU-Pro | 2,808 / 2,167 | 2,750 / 2,146 | −58 [−104, −11] | 0.008 | 0.992 |
| tau2-bench Telecom | 5,218 / 4,736 | 4,044 / 3,725 | −1174 [−1344, −1004] | <0.001 | 1.000 |

Mean completion tokens fell on seven of eight benchmarks, most on SciCode (53%) and tau2-bench Telecom (22.5%). GPQA increased by 14.1%.

## Conclusions and Future work

QAKTO reduced doomed reasoning loops in the QAD checkpoint without a statistically detected accuracy regression. The gains extended to fewer missing answers and lower token use on most benchmarks. Because KTO accepts independently labeled responses, it can use correct answers and looping failures from different questions.

Next steps are to automate collecting correct responses and incorrect responses stuck in doomed reasoning loops for KTO training, reduce IFBench’s remaining loops and test more models.
