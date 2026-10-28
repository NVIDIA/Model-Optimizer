# Reducing Reasoning Loops with QAD Driven by Anti-Repetition Prompts: First Promising Results

Daniel Korzekwa · 2026-10-20

## Key takeaways

- Further QAD on responses generated with an **anti-repetition prompt** reduced token-limit failures and slightly improved observed held-out accuracy, **without extra inference prompting**.
- On held-out subtasks, **prompt-guided QAD** reduced token-limit failures by **42%**; correct answers increased by **17%**.
- **QAD on teacher-generated responses** reduced token-limit failures by **50%**, but correct answers decreased by **33%**.
- **Next: validate beyond 11 held-out subtasks**, holding out entire problems and testing transfer across benchmarks while checking accuracy.

**Contents**

- [Prior work](#prior-work)
- [The problem: reasoning loops](#the-problem-reasoning-loops)
- [Proposed method](#proposed-method-anti-repetition-prompting-and-further-qad)
- [Experiments and results](#experiments-and-results)
- [Conclusions](#conclusions)
- [Future work](#future-work)

Acknowledgments: Thanks to **Keval Morabia** for his prior work on the [Qwen3.6-35B-A3B QAD tutorial](https://github.com/NVIDIA/Model-Optimizer/blob/main/examples/megatron_bridge/tutorials/Qwen3.6-35B-A3B/README.md), which provided the starting point for this investigation.

Quantization-aware distillation (QAD) can preserve benchmark accuracy while increasing reasoning loops: responses repeat until the token cap without delivering a final answer. We propose using an anti-repetition prompt to generate training responses, then running further QAD on selected completed responses. The added instruction is removed from training inputs and evaluation prompts.

On 11 held-out subtasks from SciCode, a scientific coding benchmark, Qwen3.6-35B-A3B had 42% fewer cap hits—responses reaching the output-token limit—and 17% more correct answers after prompt-guided QAD. Training on responses from the original higher-precision model (the BF16 teacher) reduced cap hits by 50%, but correct answers decreased by 33%. These are relative changes from a small sample. The biggest open issue is whether fewer termination failures generalize without sacrificing correctness. Next, we need a larger evaluation that holds out entire problems, plus full-benchmark accuracy checks.

## Prior work

### Qwen3.6 QAD experiments

Keval Morabia's [Qwen3.6-35B-A3B QAD tutorial](https://github.com/NVIDIA/Model-Optimizer/blob/main/examples/megatron_bridge/tutorials/Qwen3.6-35B-A3B/README.md#output-length) and its ablations reported six findings:

- **Longer outputs began before QAD.** Weight-only and weight-and-activation quantization increased SciCode mean output length by roughly 23–25%; after QAD, the increase reached 90.9% versus BF16.
- **Termination failures dominated token use.** Capped responses consumed 45.7% of completion tokens. Median length also rose, so the increase extended beyond the failing tail.
- **A decoding penalty helped selectively.** A presence penalty of 1.5 reduced QAD cap hits from 96 to 2 out of 2,704 responses, with accuracy changes within evaluation noise. It sped up SciCode generation but cost 4.8× throughput on GPQA Diamond.
- **Hesitation penalties helped less than the presence penalty.** Keval [tested the penalty](https://github.com/NVIDIA/Model-Optimizer/blob/main/examples/megatron_bridge/tutorials/Qwen3.6-35B-A3B/ABLATIONS.md#scicode-non-termination-diagnosis-and-decode-time-fix) from [Lotfi et al.](https://arxiv.org/abs/2606.00206), which targets markers such as “Wait” and “Alternatively.” It helped less and had a small accuracy cost. Their marker-density pattern did not reproduce on W4A4 PTQ: their overthinking opens new branches, while the inspected loops repeat a finished solution.
- **More QAD did not reduce cap rates in the tested setup.** [Fifty additional iterations](https://github.com/NVIDIA/Model-Optimizer/blob/main/examples/megatron_bridge/tutorials/Qwen3.6-35B-A3B/ABLATIONS.md#for-the-next-qad-run) on student- or teacher-generated chat responses brought no improvement. The student data retained loops, and the prompts rarely triggered student-specific failures.
- **Input masking gave no lasting improvement.** With [loss on assistant tokens only](https://github.com/NVIDIA/Model-Optimizer/blob/main/examples/megatron_bridge/tutorials/Qwen3.6-35B-A3B/ABLATIONS.md#data-format-pre-tokenized-blend-vs-chat-sft-data), the early cap-rate reduction faded; at iteration 500, cap rates were 5.7% versus 3.6% with all-token loss. Packing changes and noisy teacher logits prevent attributing the difference to masking alone.

Our pilot instead targets failure-prone SciCode subtasks and keeps completed responses that pass repetition and length checks for further QAD. Correctness is recorded separately, not used for filtering.

### Related research

- **Model-specific decoding interventions.** [RATIO](https://arxiv.org/abs/2609.39801) identifies tokens affected by quantization and uses full-precision guidance to set token-specific penalties. It suggests an alternative way to generate recovery data; we have not tested it on our checkpoints.
- **Learning self-correction.** [RISE](https://arxiv.org/abs/2407.18219) trains models to improve failed attempts across turns; [SCoRe](https://arxiv.org/abs/2409.12917) uses reinforcement learning on self-generated correction traces. Both motivate a next step: train the model to recover from an existing reasoning loop.
- **Training against unwanted continuations.** [Unlikelihood training](https://arxiv.org/abs/1908.04319) directly lowers repetition probability; [DPO](https://arxiv.org/abs/2305.18290) learns from preferred and rejected responses. These motivate comparing completed continuations against looping ones.

These papers motivate diagnostics and training alternatives; they do not establish a recovery-training solution for our QAD reasoning loops.

## The problem: reasoning loops

A reasoning loop repeatedly revisits a decision without reaching a final answer. In one response, the QAD model drafted code, reopened the same grid-design uncertainty, and repeated “Let me try to implement the simulation step by step” 347 times. It exhausted 131,072 tokens without a final answer.

Benchmark accuracy concealed the cost on Qwen3.6: accuracy stayed similar, but the cap-hit rate rose from **0.74% to 3.55%—4.8× (+380%)**. Total completion tokens grew by **90.9%**, from approximately **14.5 million to 27.6 million**. Qwen3-8B showed the same symptom: its cap-hit rate rose from **0.46% to 1.95%—4.2× (+323%)**. The regression therefore appeared in both tested models; both teachers also had cap hits.

### Qwen3.6-35B-A3B

Existing tutorial results cover 338 subtasks × eight attempts: 2,704 responses per checkpoint, with a 131,072-token output cap.

| Metric | BF16 teacher | QAD | Change versus teacher |
| --- | ---: | ---: | ---: |
| Cap hits | 20 (0.74%) | 96 (3.55%) | +380% |
| Missing final answer | 21 (0.78%) | 96 (3.55%) | +357% |
| Mean completion tokens | 5,348 | 10,209 | +90.9% |
| Total completion tokens, approximate | 14.5 million | 27.6 million | +90.9% |
| Subtask accuracy | 39.9% | 40.2% | +0.3 percentage points |

Of the 96 QAD responses missing a final answer, 93 hit the cap and three stopped below it. For the teacher, 20 of 21 missing answers hit the cap.

### Qwen3-8B

A separate study used 338 subtasks × 20 seeds and the Hugging Face QAD workflow (HF QAD). After context-budget skips in preceding-step chains, 6,731 matched responses per checkpoint remained. The output cap was 24,576 tokens.

| Metric | BF16 teacher | HF QAD | Change versus teacher |
| --- | ---: | ---: | ---: |
| Cap hits | 31 (0.46%) | 131 (1.95%) | +323% |
| Missing final answer | 26 (0.39%) | 118 (1.75%) | +354% |
| Mean completion tokens | 6,321 | 6,665 | +5.4% |
| Total completion tokens, approximate | 42.5 million | 44.9 million | +5.4% |
| Subtask accuracy | 31.27% | 29.79% | −1.48 percentage points |

### How to interpret both comparisons

Qwen3-8B used a lower output cap, so its smaller token increase does not establish a weaker looping tendency. Cap hits alone do not prove repetition, and they overlap with missing final answers. We observed termination regressions in both models; the recovery experiments below evaluate Qwen3.6 only.

## Proposed method: anti-repetition prompting and further QAD

We compare **two further-QAD runs** to test which training-data source gives better termination and correctness:

- **Prompt-data QAD:** train on responses generated by the QAD student with an anti-repetition instruction.
- **Teacher-data QAD:** train on responses generated by the BF16 teacher with the original prompt.

During QAD, both students learn from BF16 teacher probabilities along their respective response sequences. The comparison changes the source of training text while keeping the distillation teacher the same.

Both follow the same pipeline:

1. **Generate.** For the same questions with observed termination failures, collect responses from each source.
2. **Filter.** Keep completed responses that pass repetition and length checks. Record correctness separately. Remove the anti-repetition instruction from the prompt-derived training inputs.
3. **Distill.** Train a separate model on each dataset, starting from the same original QAD checkpoint. In both runs, keep quantizers active and match BF16 teacher probabilities along the selected responses. Evaluate each model with the original prompt and no penalty.

Our recovery instruction, called **prompt 7**, was:

> If your reasoning begins repeating, summarize the unresolved issue once, resolve it as best you can, and continue toward a final answer.

The method trains on fresh attempts at the original question. Whether it teaches escape from an established loop remains untested.

## Experiments and results

### 1. Design the anti-repetition prompt

We first selected a prompt on one repeatable failure case, then checked its behavior across the full benchmark.

#### Select a prompt and repeat with fresh seeds

On SciCode 2.1, `Gaussian_Beam_Focus`, we compared the original prompt with 11 added instructions using 100 matched seeds, then repeated with 100 fresh seeds. Cells show **first run → repeat**.

| Prompt | Cap hits | Missing final | Mean tokens |
| --- | ---: | ---: | ---: |
| Original, no extra instruction | 36% → 26% | 38% → 29% | 69,962 → 58,772 |
| Initial (0) | 10% → 5% | 11% → 7% | 26,250 → 22,352 |
| Variant 1 | 24% → 20% | 26% → 21% | 60,112 → 53,088 |
| Variant 2 | 8% → 23% | 10% → 23% | 37,710 → 53,349 |
| Variant 3 | 13% → 15% | 13% → 16% | 42,023 → 45,429 |
| Variant 4 | 35% → 29% | 43% → 33% | 70,200 → 66,617 |
| Variant 5 | 10% → 14% | 10% → 16% | 45,360 → 50,019 |
| Variant 6 | 17% → 27% | 21% → 30% | 52,172 → 61,291 |
| **Variant 7** | **3% → 2%** | **4% → 4%** | **16,884 → 16,145** |
| Variant 8 | 32% → 30% | 39% → 37% | 62,681 → 61,551 |
| Variant 9 | 10% → 20% | 11% → 23% | 40,913 → 56,526 |
| Variant 10 | 30% → 20% | 34% → 26% | 65,740 → 57,469 |

Prompt 7 remained strongest on termination across both seed sets, while several other prompts varied substantially. Every condition scored zero correct: finishing did not mean solving the task. Adding five examples also worsened termination in a separate test.

#### Validate selected prompts across the full SciCode benchmark

We compared four added instructions on 338 subtasks across 80 parent problems, using eight matched seeds: 2,704 responses per instruction. Related subtasks can share a parent problem. Instruction numbers identify prompt variants.

| Instruction | Cap hits | Missing final | Mean tokens | Accuracy |
| --- | ---: | ---: | ---: | ---: |
| Original prompt, no extra instruction (historical) | 96 (3.55%) | 96 (3.55%) | 10,209 | 40.2% |
| Initial (0) | 71 (2.63%) | 72 (2.66%) | 9,241 | 39.16% |
| 4 | 83 (3.07%) | 83 (3.07%) | 9,266 | 40.94% |
| 7 | 57 (2.11%) | 58 (2.14%) | 8,874 | 40.09% |
| 10 | 96 (3.55%) | 97 (3.59%) | 10,920 | 39.28% |

Prompt 7's termination advantage extended beyond the selected failure case, but was smaller across the full benchmark and did not give the highest accuracy. Variant 0 also adds guidance; this matched comparison has no original-prompt arm.

**The historical row is not a matched control:** it covers the same 338 subtasks and 2,704 responses, but comes from separate tutorial runs rather than the matched seeds used for variants 0, 4, 7, and 10. Differences cannot be attributed to the added instructions alone.

### 2. QAD on generated responses

We fixed a split of the 55 subtasks with historical QAD cap hits: 44 for training and 11 held out. Ten attempts per subtask and source supplied 440 training candidates per source. Filtering retained 334 prompt-derived and 322 teacher-derived examples.

Both runs started from the same fake-quantized iteration-500 checkpoint. Training used KL-divergence loss on response tokens, matching the teacher's top-4,096 logits; input tokens were masked from the loss. Quantizers remained active, with **no supervised cross-entropy loss**. Both models were exported to NVFP4.

We evaluated original QAD and both trained models with the original task prompt, no decoding penalty, and ten fresh matched seeds per subtask. Correctness used official SciCode executable tests, not generated training answers.

#### Training subtasks: 440 responses per model

| Model | Cap hits | Missing final | Mean tokens | Correct |
| --- | ---: | ---: | ---: | ---: |
| Original QAD | 41 (9.32%) | 42 (9.55%) | 23,677 | 57 |
| Prompt-data QAD | 17 (3.86%) | 15 (3.41%) | 18,767 | 57 |
| Teacher-data QAD | 14 (3.18%) | 14 (3.18%) | 17,854 | 47 |

#### Held-out subtasks: 110 responses per model

| Model | Cap hits | Missing final | Mean tokens | Correct |
| --- | ---: | ---: | ---: | ---: |
| Original QAD | 12 (10.91%) | 12 (10.91%) | 23,742 | 6 |
| Prompt-data QAD | 7 (6.36%) | 7 (6.36%) | 19,231 | 7 |
| Teacher-data QAD | 6 (5.45%) | 6 (5.45%) | 18,236 | 4 |

Both further-QAD runs reduced cap hits. QAD on prompt-7 responses slightly improved held-out accuracy (6 to 7 correct answers); QAD on teacher responses reduced cap hits further but lowered accuracy on both splits.

## Conclusions

**Prompt-guided QAD reduced termination failures without extra inference guidance** and showed a better accuracy tradeoff than teacher-data QAD. Held-out accuracy improved by one answer; **broader accuracy gains have not yet been measured**.

We speculate that targeted prompts followed by QAD may help with **other stability failures**, potentially offering greater value at **lower precision**. **Neither possibility has been tested.**

Evidence covers **only 11 held-out subtasks** from a failure-selected set, with parent problems shared across splits. Dataset sizes differ; filters can reject useful code, and cap rates do not fully measure repetition. **Statistical significance, root cause, and full-benchmark accuracy gains remain unestablished.**

## Future work

### Extend the anti-repetition-prompt method

- **Scale evaluation.** Expand beyond failure-selected SciCode subtasks, hold out entire parent problems, and check full-benchmark accuracy.
- **Measure cross-task transfer.** Train on SciCode responses, then evaluate without extra prompting on coding, math, and instruction-following benchmarks excluded from prompt design and data generation. Measure termination, repetition, token use, and accuracy to identify transfer and regressions.
- **Design prompts from other tasks.** Use reasoning-loop examples from other coding, math, and instruction-following tasks to develop anti-repetition prompts. Generate responses for QAD and evaluate on held-out tasks.
- **Compare generation methods.** Generate responses with a presence penalty, then run QAD and evaluate without it. Test whether this alternative improves recovery.
- **Test broader applicability.** Complete the Qwen3-8B comparison, then test lower precisions and other stability failures. Check how benefits vary with precision.

- **Teach recovery and discourage loops.** Pair failed reasoning prefixes with completed continuations. Compare QAD, direct preference optimization (DPO), reward-based reinforcement learning (RL, such as GRPO), and unlikelihood loss against repetition. Match task splits and compute budgets; measure correctness, termination, and repetition.

### Other approaches and diagnostics

- **Test quantization's role.** Fine-tune the BF16 teacher on the original QAD data and compare with the unchanged teacher. Test whether loops increase without quantization; data and training effects remain combined.
- **Compare decoding alternatives.** Test temperature settings and model-specific token penalties against the trained models, measuring correctness, token use, and latency.

**KL vocabulary coverage is lower priority:** [Keval's omitted-mass ablation](https://github.com/NVIDIA/Model-Optimizer/blob/main/examples/megatron_bridge/tutorials/Qwen3.6-35B-A3B/ABLATIONS.md#for-the-next-qad-run) did not reduce cap rates. Top-p and full-vocabulary KL remain unproven alternatives.

These experiments remain untested here, including penalty-derived data and Qwen3-8B recovery training.
