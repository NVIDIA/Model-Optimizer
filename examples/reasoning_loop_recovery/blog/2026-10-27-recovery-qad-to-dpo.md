# Quantization-Aware DPO (QADPO): Targeting Doomed Reasoning Loops in Quantized Models—SciCode Cap Hits from 10.91% to Zero

Daniel Korzekwa · 2026-10-27

## Key takeaways

- SciCode-targeted QAD fine-tuning followed by QADPO reduced cap hits from **10.91% to zero** on held-out failure-prone SciCode subtasks.
- Recovery training used only SciCode data, yet the MMLU-Pro cap-hit rate also fell from **0.20% to zero**.
- Mean completion tokens fell **51% on full SciCode** (8,269 → 4,092) and **10% on MMLU-Pro** (3,036 → 2,725), including reasoning.
- Full SciCode accuracy fell **3.55 percentage points** (43.20% → 39.64%). MMLU-Pro accuracy fell **0.50 percentage points** (84.70% → 84.20%).
- The next goal is preserving accuracy while reducing doomed reasoning loops.

**Contents**

- [Introduction](#introduction)
- [Related work](#related-work)
- [QADPO method](#qadpo-method)
- [Experiments and results](#experiments-and-results)
- [Future work](#future-work)
- [Appendix](#appendix)

## Introduction

Our [first post](2026-10-20-reducing-reasoning-loops-after-qad.md) studied doomed reasoning loops in NVFP4-quantized Qwen3.6-35B-A3B after quantization-aware distillation (QAD). These loops revisit the same issue without progressing toward a final answer; a cap hit means reaching the output-token limit. PTQ already increased cap-hit rates, but the largest jump came after QAD in the [prior full-SciCode experiments](https://github.com/NVIDIA/Model-Optimizer/blob/f32d623ab3f07658e1ec6c42b38e79a4be8edecb/examples/megatron_bridge/tutorials/Qwen3.6-35B-A3B/ABLATIONS.md#presence_penalty-15-removes-it-at-decode-time). SciCode-targeted QAD fine-tuning on completed responses generated with an anti-repetition prompt reduced the cap-hit rate on held-out SciCode subtasks from 10.91% to 6.36%. This follow-up tests whether more data, iterative QAD recovery, and quantization-aware direct preference optimization (QADPO) further reduce doomed reasoning loops while preserving broader accuracy.

After SciCode-targeted recovery QAD and QADPO, the cap-hit rate fell from **10.91% to zero** on held-out SciCode validation subtasks. Although recovery training used only SciCode data, the MMLU-Pro cap-hit rate also fell from 0.20% to zero. Mean completion tokens, including reasoning, fell **51% on full SciCode and 10% on MMLU-Pro**. Broader evaluation showed accuracy declines: **43.20% → 39.64% on full SciCode (−3.55 percentage points)** and 84.70% → 84.20% on MMLU-Pro (−0.50 points).

## Related work

[Lee et al. (ACL 2024)](https://aclanthology.org/2024.acl-long.612/) introduced quantization-aware DPO (QDPO) to recover conversational quality after quantization.

| | Their QDPO | Our QADPO experiment |
| --- | --- | --- |
| Goal | Recover conversational quality after quantization | Reduce doomed reasoning loops after QAD |
| Preferred responses | Full-precision model outputs | Completed responses from QAD models |
| Rejected responses | Quantized model outputs | Shortened responses showing repeated reasoning |

## QADPO method

```text
Qwen3.6-35B-A3B after NVFP4 PTQ + QAD
                            ↓
Three SciCode recovery QAD rounds using:
  Completed responses with a final answer, without repetition
                            ↓
QADPO using SciCode preference pairs:
  Preferred: completed responses with a final answer, without repetition
  Rejected: doomed reasoning loops (non-terminating responses)
                            ↓
NVFP4 export → evaluation
```

We first ran three recovery QAD rounds from the initial QAD checkpoint. Each round generated fresh responses from the previously improved model and retained completed responses without repetition. The [appendix](#which-response-generation-methods-work-best-for-recovery-qad) compares data-generation methods.

QADPO keeps quantization active while training the model to prefer completed answers over doomed reasoning loops. The [appendix](#how-were-the-qadpo-preference-pairs-generated) describes pair generation.

Correctness was recorded but not required when selecting preferred responses.

See the [GitHub experiment documentation](https://github.com/NVIDIA/Model-Optimizer/tree/dkorzekwa/reasoning-loop-recovery/examples/reasoning_loop_recovery/experiments/dpo_recovery) for implementation, evaluation, and reproduction details.

## Experiments and results

### QADPO reduces cap hits on failure-prone SciCode subtasks

We evaluated the same 44 training and 11 held-out validation subtasks with ten matched attempts each, using original prompts and no inference penalties. One-stage QAD generates recovery data from the initial checkpoint; iterative QAD generates fresh data after each round. Both used 2,181 total training examples. QADPO continued from round 3 with 104 additional training preference pairs.

#### Training subtasks

| Model | Cap-hit rate | Accuracy | Mean completion tokens |
| --- | ---: | ---: | ---: |
| Initial QAD (baseline) | 9.32% (41/440) | 12.95% (57/440) | 23,677 |
| One-stage recovery QAD | 5.68% (25/440) | 12.95% (57/440) | 19,873 |
| Iterative recovery QAD, round 3 | 3.86% (17/440) | 12.05% (53/440) | 17,090 |
| Round 3 + QADPO | 0.00% (0/440) | 14.77% (65/440) | 7,470 |

#### Held-out validation subtasks

| Model | Cap-hit rate | Accuracy | Mean completion tokens |
| --- | ---: | ---: | ---: |
| Initial QAD (baseline) | 10.91% (12/110) | 5.45% (6/110) | 23,742 |
| One-stage recovery QAD | 5.45% (6/110) | 4.55% (5/110) | 18,121 |
| Iterative recovery QAD, round 3 | 0.00% (0/110) | 3.64% (4/110) | 11,736 |
| Round 3 + QADPO | 0.00% (0/110) | 4.55% (5/110) | 8,667 |

At the same cumulative example count and optimization steps, iterative QAD had fewer cap hits than one-stage QAD; token and compute budgets were not exactly matched. Adding more one-stage data alone gave no consistent further improvement.

QADPO removed the remaining training-subtask cap hits. The held-out cap-hit rate fell from **10.91% to zero**, though round-3 QAD had already reached zero on that split. The final model had one fewer correct validation response than initial QAD.

A separate comparison found that a fourth QAD round did not match QADPO’s lower cap-hit rates, although QADPO had two training cap hits and lower held-out accuracy ([appendix](#round-4-qad-versus-qadpo)).

### QADPO on full SciCode and MMLU-Pro

We evaluated three checkpoints on all 338 SciCode subtasks and a fixed 1,000-question MMLU-Pro subset, proportional across all 14 subjects. Each checkpoint answered every SciCode subtask and MMLU-Pro question once, producing 4,014 scored responses across three checkpoints.

| Benchmark | Model | Cap-hit rate | Accuracy | Mean completion tokens |
| --- | --- | ---: | ---: | ---: |
| MMLU-Pro | Initial QAD | 0.20% (2/1,000) | 84.70% (847/1,000) | 3,036 |
| MMLU-Pro | Round-3 QAD | 0.00% (0/1,000) | 84.70% (847/1,000) | 2,733 |
| MMLU-Pro | Round 3 + QADPO | 0.00% (0/1,000) | 84.20% (842/1,000) | 2,725 |
| Full SciCode | Initial QAD | 2.07% (7/338) | 43.20% (146/338) | 8,269 |
| Full SciCode | Round-3 QAD | 0.59% (2/338) | 42.31% (143/338) | 5,818 |
| Full SciCode | Round 3 + QADPO | 0.00% (0/338) | 39.64% (134/338) | 4,092 |

The final checkpoint had **zero observed cap hits on both benchmarks**. On MMLU-Pro, accuracy fell only 0.50 percentage points (84.70% → 84.20%), despite all recovery QAD and QADPO data coming from SciCode. The cap-hit rate also fell from 0.20% to zero, suggesting possible transfer beyond the training benchmark. However, there were only two baseline cap hits, and round-3 QAD had already removed them.

The accuracy tradeoff was larger on full SciCode: accuracy fell **3.55 percentage points (43.20% → 39.64%)** after SciCode-targeted QAD and QADPO. The full decline cannot be attributed to QADPO alone. Mean completion tokens fell **51% on full SciCode and 10% on MMLU-Pro** from initial QAD to the final model, alongside the accuracy declines.

## Future work

The next step is to preserve the initial QAD model's accuracy while retaining the lower cap-hit rates. We will test correctness-aware data selection.

Appropriate abstention is another possible target. In the [Qwen3.6 AA-Omniscience results](https://github.com/NVIDIA/Model-Optimizer/blob/f32d623ab3f07658e1ec6c42b38e79a4be8edecb/examples/megatron_bridge/tutorials/Qwen3.6-35B-A3B/README.md#knowledge-recall-and-calibration), correct answers stayed near 21%, but from BF16 to QAD, abstention fell from 46.0% to 29.8% and incorrect answers rose from 32.3% to 48.5%. QADPO could target appropriate abstention through preferences between correct answers, incorrect guesses, and “I don't know” responses. This remains untested.

## Appendix

### Round-4 QAD versus QADPO

Both methods started from round-3 QAD and trained for 13 steps: round-4 QAD used 104 completed, non-looping responses generated with prompt 7; QADPO used 104 preference pairs. Tokens and compute were not matched. We evaluated 40 fresh answers per subtask on the same 44 training and 11 held-out subtasks.

#### Training subtasks: 1,760 responses per model

| Model | Cap-hit rate | Accuracy | Mean completion tokens |
| --- | ---: | ---: | ---: |
| Round-3 QAD | 2.44% (43/1,760) | 12.84% (226/1,760) | 16,217 |
| Round-4 QAD | 3.18% (56/1,760) | 12.22% (215/1,760) | 16,574 |
| Round 3 + QADPO | 0.11% (2/1,760) | 12.44% (219/1,760) | 7,026 |

#### Held-out subtasks: 440 responses per model

| Model | Cap-hit rate | Accuracy | Mean completion tokens |
| --- | ---: | ---: | ---: |
| Round-3 QAD | 2.50% (11/440) | 5.45% (24/440) | 13,876 |
| Round-4 QAD | 3.86% (17/440) | 3.64% (16/440) | 16,013 |
| Round 3 + QADPO | 0.00% (0/440) | 2.05% (9/440) | 8,007 |

Another QAD round increased cap hits relative to round 3. QADPO produced fewer cap hits and shorter completions, but lower held-out accuracy.

### Recovery-training data

#### How were the QADPO preference pairs generated?

We paired completed answers with doomed reasoning loops to teach the model to finish rather than repeat. Responses came from the initial and round-3 QAD checkpoints. An anti-repetition instruction helped obtain completed answers for the preferred side of each pair. We trained QADPO on 104 preference pairs. Preferences targeted completion, not correctness.

#### Which response-generation methods work best for recovery QAD?

We used the anti-repetition instruction (prompt 7) to generate data for the three recovery QAD rounds. We compared four data-generation methods: an anti-repetition instruction (prompt 7), original prompts, a presence penalty, and the BF16 teacher. Correctness was recorded but not required for data acceptance. Evaluation used 50 matched attempts per subtask, without extra prompts or penalties. Each method is represented by its run with the lowest held-out cap-hit rate.

##### Training subtasks: 2,200 responses per model

| Recovery-data source | Examples | Cap-hit rate | Accuracy |
| --- | ---: | ---: | ---: |
| Initial QAD, no recovery | — | 10.91% (240/2,200) | 12.55% (276/2,200) |
| Prompt 7 | 271 | 2.05% (45/2,200) | 13.68% (301/2,200) |
| Presence penalty | 322 | 4.32% (95/2,200) | 13.55% (298/2,200) |
| BF16 teacher | 322 | 3.73% (82/2,200) | 12.55% (276/2,200) |
| Filtered original prompts | 271 | 2.50% (55/2,200) | 11.91% (262/2,200) |

##### Held-out validation subtasks: 550 responses per model

| Recovery-data source | Examples | Cap-hit rate | Accuracy |
| --- | ---: | ---: | ---: |
| Initial QAD, no recovery | — | 11.27% (62/550) | 4.00% (22/550) |
| Prompt 7 | 271 | 2.36% (13/550) | 3.27% (18/550) |
| Presence penalty | 322 | 3.64% (20/550) | 5.27% (29/550) |
| BF16 teacher | 322 | 3.09% (17/550) | 3.09% (17/550) |
| Filtered original prompts | 271 | 2.18% (12/550) | 4.00% (22/550) |

For future runs, we would start with filtered original-prompt responses: held-out results were comparable to prompt 7, with a simpler generation setup. Prompt 7 remains an option when too few completed responses are available. Training budgets differed across methods, so these results do not establish an overall winner.
