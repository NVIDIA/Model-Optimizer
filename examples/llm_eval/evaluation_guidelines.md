# Guidelines for Evaluating Models Optimized with NVIDIA Model Optimizer

## Contents

- [Introduction](#introduction)
- [What should you measure, and how?](#what-should-you-measure-and-how)
  - [Show where the model improves or worsens](#show-where-the-model-improves-or-worsens)
  - [Check for regressions](#check-for-regressions)
  - [Assess model reliability](#assess-model-reliability)
  - [Measure inference efficiency](#measure-inference-efficiency)
- [How can you evaluate faster?](#how-can-you-evaluate-faster)
  - [Prioritize question coverage over repetitions](#prioritize-question-coverage-over-repetitions)
  - [Collect as much evaluation data as you need](#collect-as-much-evaluation-data-as-you-need)
  - [Test known weaknesses on a focused subset](#test-known-weaknesses-on-a-focused-subset)
- [Future work](#future-work)
- [Acknowledgments](#acknowledgments)

## Introduction

When you optimize a reasoning model with NVIDIA Model Optimizer, how do you verify improvements without overlooking regressions? Average accuracy can hide doomed reasoning loops (repetitive reasoning that continues until the output-token limit is reached), missing answers, inappropriate guessing, or excessive output length.

You may assume that reliable evaluation requires many repetitions. But could repeating by default waste time? How many questions and repetitions do you need to measure improvements and rule out regressions?

This guide helps you evaluate models optimized with NVIDIA Model Optimizer faster and more reliably.

For benchmark commands, see the [LLM evaluation examples](README.md). For an AI-assisted workflow using NeMo Evaluator Launcher, see the [ModelOpt evaluation skill](../../plugins/modelopt/skills/evaluation/SKILL.md).

**What should you measure, and how?**

- **Show where the model improves or worsens:** use paired statistical tests to distinguish model differences from evaluation noise.
- **Check for regressions:** use confidence intervals and non-inferiority tests to rule out unacceptable regression, even if the baseline is not shown to be better.
- **Assess model reliability:** measure doomed reasoning loops, missing answers, and whether the model abstains appropriately.
- **Measure inference efficiency:** compare output length, including reasoning tokens.

**How can you evaluate faster?**

- **Prioritize question coverage over repetitions:** repeat evaluations when the available questions do not provide enough evidence.
- **Collect as much evaluation data as you need:** use pilot results to choose enough questions or repetitions to assess regression.
- **Test known weaknesses on a focused subset:** failure-prone questions can reveal targeted improvements with fewer responses; check broader performance separately.

## What should you measure, and how?

We compare two NVFP4 Qwen3.6-35B-A3B checkpoints to show you how to put these recommendations into practice. Baseline is after [post-training quantization (PTQ) and quantization-aware distillation (QAD)](https://github.com/NVIDIA/Model-Optimizer/blob/main/examples/megatron_bridge/tutorials/Qwen3.6-35B-A3B/README.md), before recovery training. PTQ reduces numerical precision after training; QAD uses a teacher model to recover accuracy in the quantized model. This baseline preserved accuracy but had more doomed reasoning loops.

Candidate was trained directly from Baseline using Kahneman–Tversky Optimization (KTO), which learns from desirable and undesirable responses, aiming to reduce those loops while preserving accuracy.

We use MMLU-Pro, SciCode and AA-Omniscience to illustrate these evaluation guidelines.

We evaluate on 294 of 338 SciCode subtasks, excluding the 44 used for KTO recovery training. KTO used model-generated responses, not reference answers.

| Benchmark | Questions or subtasks | Repetitions | Responses per model |
| --- | ---: | ---: | ---: |
| MMLU-Pro | 1,000 questions sampled across all 14 subjects | 20 | 20,000 |
| SciCode | 294 subtasks | 20 | 5,880 |
| SciCode focused subset | 11 challenging, loop-prone held-out subtasks | 10 | 110 |
| AA-Omniscience public v1 | 600 questions | 10 | 6,000 |

### Show where the model improves or worsens

We first average each model’s correctness across repetitions of each question, then compute **Candidate − Baseline** for each question. Repeated responses are not additional independent questions. All three benchmarks use t-based 95% confidence intervals and one-sided paired t-tests to check whether the mean difference is above or below zero. Changes are reported in percentage points (pp), not relative percentages. For simplicity, this example treats questions and SciCode subtasks as independent.

| Benchmark | Baseline | Candidate | Difference [95% CI], pp | Candidate higher, p-value | Baseline higher, p-value |
| --- | ---: | ---: | ---: | ---: | ---: |
| MMLU-Pro | 84.51% | 84.26% | −0.26 [−0.61, +0.10] | 0.920 — no evidence | 0.080 — no evidence |
| SciCode | 43.52% | 44.90% | +1.38 [+0.08, +2.67] | 0.019 — yes | 0.981 — no evidence |
| AA-Omniscience | 21.80% | 21.92% | +0.12 [−0.64, +0.88] | 0.381 — no evidence | 0.619 — no evidence |

“Yes” marks p < 0.05; “no evidence” means the test did not establish the stated direction.

On SciCode, the candidate is more accurate than the baseline (+1.38 pp; p = 0.019).

### Check for regressions

No evidence of a difference does not establish no regression. Choose an acceptable accuracy-loss margin, then use a non-inferiority test to assess whether the loss is smaller. The checks below use one-sided paired t-tests.

| Rule out an accuracy loss of | MMLU-Pro p-value | SciCode p-value | AA-Omniscience p-value |
| --- | ---: | ---: | ---: |
| 2.0 pp | <0.001 — yes | <0.001 — yes | <0.001 — yes |
| 1.0 pp | <0.001 — yes | <0.001 — yes | 0.002 — yes |
| 0.5 pp | 0.088 — no evidence | 0.002 — yes | 0.056 — no evidence |
| 0.1 pp | 0.804 — no evidence | 0.013 — yes | 0.288 — no evidence |

### Assess model reliability

Average accuracy alone does not show whether a model behaves reliably: also measure doomed reasoning loops, missing answers, and whether it abstains when appropriate.

| Measure | Recommendation |
| --- | --- |
| Doomed reasoning loops | Use cap-hit rates to screen for potential loops, then inspect the text to confirm repetition. A cap hit reaches the configured response-token limit (131,072 here). |
| Missing final answers | Report how often responses contain no final answer. |
| Abstention | Report intentional abstentions separately from incorrect or missing answers; [AA-Omniscience](https://artificialanalysis.ai/evaluations/omniscience) is an example benchmark. |

Rare failures can make t-based approximations unreliable, so we use paired-bootstrap 95% confidence intervals and one-sided paired permutation tests for the rates below.

#### Doomed reasoning loops

The candidate has fewer cap hits on both SciCode and AA-Omniscience (p < 0.001).

| Benchmark | Baseline cap rate | Candidate cap rate | Difference [95% CI], pp | Candidate lower, p-value | Candidate higher, p-value |
| --- | ---: | ---: | ---: | ---: | ---: |
| MMLU-Pro | 0.05% (10/20,000) | 0.03% (6/20,000) | −0.02 [−0.07, +0.02] | 0.244 — no evidence | 0.878 — no evidence |
| SciCode | 1.29% (76/5,880) | 0.00% (0/5,880) | −1.29 [−1.70, −0.92] | <0.001 — yes | 1.000 — no evidence |
| AA-Omniscience | 1.77% (106/6,000) | 0.85% (51/6,000) | −0.92 [−1.38, −0.48] | <0.001 — yes | 1.000 — no evidence |

#### Missing final answers

The candidate also reduced missing final answers on SciCode and AA-Omniscience (p < 0.001).

| Benchmark | Baseline | Candidate | Difference [95% CI], pp | Candidate lower, p-value | Candidate higher, p-value |
| --- | ---: | ---: | ---: | ---: | ---: |
| MMLU-Pro | 0.05% (10/20,000) | 0.03% (6/20,000) | −0.02 [−0.07, +0.02] | 0.244 — no evidence | 0.878 — no evidence |
| SciCode | 1.31% (77/5,880) | 0.00% (0/5,880) | −1.31 [−1.72, −0.94] | <0.001 — yes | 1.000 — no evidence |
| AA-Omniscience | 1.77% (106/6,000) | 0.85% (51/6,000) | −0.92 [−1.38, −0.48] | <0.001 — yes | 1.000 — no evidence |

#### Abstention

[AA-Omniscience public v1](https://huggingface.co/datasets/ArtificialAnalysis/AA-Omniscience-Public) distinguishes abstention from incorrect guessing. The Codex assistant graded responses against reference answers, blinded to checkpoint identity. The reported configuration was GPT-6.1 Sol (xhigh); a pinned judge snapshot was not independently verified. Deliberate abstentions exclude empty answers and uncertain guesses.

| Measure | Baseline | Candidate | Difference [95% CI], pp | Candidate higher, p-value | Candidate lower, p-value |
| --- | ---: | ---: | ---: | ---: | ---: |
| Deliberate abstention | 13.30% (798/6,000) | 14.55% (873/6,000) | +1.25 [+0.23, +2.27] | 0.009 — yes | 0.993 — no evidence |
| Incorrect answers | 62.03% (3,722/6,000) | 61.72% (3,703/6,000) | −0.32 [−1.57, +0.97] | 0.696 — no evidence | 0.322 — no evidence |

The candidate abstained more often (p = 0.009), with no evidence of a change in incorrect-answer rates.

### Measure inference efficiency

Compare mean and median completion tokens, including reasoning, to assess how much generation each answer requires.

For mean length, we use the same question-level paired t-tests and t-based 95% confidence intervals as accuracy.

| Benchmark | Baseline: mean / median | Candidate: mean / median | Change: mean / median, % | Mean change [95% CI], tokens | Candidate shorter, p-value | Candidate longer, p-value |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MMLU-Pro | 2,803 / 2,157 | 2,759 / 2,145 | −1.6% / −0.6% | −44 [−99, +11] | 0.057 — no evidence | 0.943 — no evidence |
| SciCode | 6,188 / 3,205 | 3,048 / 1,802 | −50.7% / −43.8% | −3,140 [−3,666, −2,614] | <0.001 — yes | 1.000 — no evidence |
| AA-Omniscience | 4,280 / 1,756 | 3,097 / 1,705 | −27.6% / −2.9% | −1,183 [−1,764, −602] | <0.001 — yes | 1.000 — no evidence |

The candidate uses fewer tokens on average on SciCode and AA-Omniscience (p < 0.001).

## How can you evaluate faster?

### Prioritize question coverage over repetitions

For MMLU-Pro, we compared more questions evaluated once with fewer questions evaluated repeatedly, using 1,000 responses per model. Both approaches gave similarly stable estimates of the accuracy difference between models. Prioritize question coverage. Only add repetitions when the available questions do not provide enough evidence. The [plots in the next section](#collect-as-much-evaluation-data-as-you-need) show how these extra responses narrow confidence intervals.

<img src="images/evaluation_guidelines/allocation-accuracy-gap.svg" alt="At a fixed 1,000-response budget, the mean accuracy gap and observed variation remain similar across one to twenty repetitions." width="640" />

Analyze results at the question level: average repeats within each question. Question-level tests can use one repetition; repetition-level t-tests need multiple runs.

### Collect as much evaluation data as you need

You may not need the full evaluation budget to detect an improvement or rule out an unacceptable regression. The data needed depends on the model pair, metric, acceptable regression margin, and statistical target.

The plots below are retrospective examples. Their threshold crossings are not validated adaptive stopping rules or general sample-size prescriptions.

**How many responses show fewer cap hits on SciCode?**

Among the tested sizes, the p-value first fell below 0.05 at 588 responses, and below 0.001 at 882 per model. The full 5,880 responses were unnecessary to establish evidence of fewer candidate cap hits at those thresholds.

<img src="images/evaluation_guidelines/scicode-cap-data-volume.svg" alt="SciCode cap-hit rate evidence excluding the 44 recovery-training subtasks: blue points increase coverage from ten to 294 subtasks with one repetition; green points add repetitions on all 294 subtasks, reaching 5,880 responses per model. Subsets of ten to 150 subtasks observe no cap hits and have no informative bootstrap interval. The lower panel shows permutation p-values." width="640" />

**How many responses rule out a 1 pp accuracy loss on MMLU-Pro?**

To rule out a 1 pp accuracy loss for the candidate versus the baseline, the non-inferiority test first reached p < 0.05 at 750 responses, p < 0.01 at 3,000, and p < 0.001 at 12,000 per model among the tested sizes.

<img src="images/evaluation_guidelines/mmlu-accuracy-data-volume.svg" alt="MMLU-Pro paired accuracy differences with t-based 95% confidence intervals. Blue points increase question coverage from 50 to 1,000 with one evaluation each; green points add repetitions on all 1,000 questions up to 20,000 responses per model. The lower panel shows one-sided non-inferiority p-values for a loss below 1 pp, with a dashed line at p = 0.05." width="640" />

### Test known weaknesses on a focused subset

Freeze a subset from earlier baseline failures to screen for targeted improvements. We report the 11 held-out subtasks, with ten matched repetitions—110 responses per model.

| Benchmark | Baseline cap rate | Candidate cap rate | Difference [95% CI], pp | Candidate lower, p-value | Candidate higher, p-value |
| --- | ---: | ---: | ---: | ---: | ---: |
| SciCode | 1.29% (76/5,880) | 0.00% (0/5,880) | −1.29 [−1.70, −0.92] | <0.001 — yes | 1.000 — no evidence |
| Focused subset | 10.91% (12/110) | 0.00% (0/110) | −10.91 [−18.18, −5.45] | 0.004 — yes | 1.000 — no evidence |

The focused held-out subset detects fewer candidate cap hits with 110 responses per model. Use it for a quick check of known weaknesses, then evaluate broader accuracy and reliability separately.

## Future work

A user should be able to ask “Is there an unacceptable regression?” or “Is the candidate better than the baseline?”, specifying the acceptable regression margin, p-value threshold, and confidence level. Future evaluation tools could choose how many questions and repetitions to use, select appropriate statistical tests, and report accuracy, cap hits, missing final answers, output length, and uncertainty.

Check independence assumptions: this example treats SciCode subtasks as independent. Related subtasks from the same parent problem may require grouped analysis to avoid overstating evidence.

## Acknowledgments

Thanks to Keval Morabia, whose [Qwen3.6 Model Optimizer tutorial](https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/megatron_bridge/tutorials/Qwen3.6-35B-A3B) served as the starting point for this guide and already applies many of the practices described here.
