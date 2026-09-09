# ModelOpt for Researchers: Fast Experimentation Workflows

Model optimization research depends on short feedback loops: test a hypothesis cheaply, compare candidates
reproducibly, and spend full-scale compute only on the most promising experiments. This guide collects practical
ModelOpt workflows for that iterative research process.

Current workflows include:

- [Efficient model evaluation](#efficient-evaluation-with-lm-eval-harness) with smaller benchmark subsets.

- [Downstream evaluation over time during distillation](#track-downstream-quality-over-time-during-distillation)
  with validation checkpoints.

- [Efficient data blend preparation](#prepare-token-budgeted-data-blends) for distillation experiments.

The guide will grow as additional research workflows are documented. It complements the feature-specific
[examples](../) by connecting them into experimentation strategies rather than replacing their detailed
instructions.

## Efficient evaluation with LM-Eval Harness

[LM-Eval Harness](../llm_eval/README.md) supports many accuracy benchmarks, but full runs are often too slow for
every iteration of model pruning, distillation, or quantization. Use progressively larger evaluation subsets to
reject weak candidates quickly and reserve full runs for the most promising models.

In LM-Eval, `--limit N` evaluates the first `N` samples of each individual task. For task groups such as MMLU and
MMLU-Pro, the limit applies to every subject, not to the group as a whole.

The following table gives a practical progression for LM-Eval's MMLU-Pro task group, which contains 14 subjects
and 12,032 questions. Example times assume Qwen3-8B, a batch size of 4, and subject-level parallelism on eight
H100 80GB GPUs:

| Limit per subject | Questions evaluated | Worst-case 95% margin of error | Example time |
|-------------------|--------------------:|--------------------------------:|-------------:|
| `10` | 140 | ±8.3 percentage points | ~3 minutes |
| `50` | 700 | ±3.7 percentage points | ~14 minutes |
| `100` | 1,400 | ±2.6 percentage points | ~28 minutes |
| `200` | 2,800 | ±1.9 percentage points | ~56 minutes |
| None | 12,032 | ±0.9 percentage points | 4 hours |

The example times scale an approximately four-hour full run by the fraction of questions evaluated. Actual time
depends on the model, hardware, batch size, and parallelism.

The margins of error are conservative planning estimates. They use 50% accuracy, the normal approximation for a
[binomial proportion confidence interval](https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval#Normal_approximation_interval).

These estimates treat benchmark questions as independent random samples from a broader population of possible
questions. Because `--limit` selects the first samples, limited scores may also be affected by dataset ordering
and should not be reported as final benchmark results.

Add `--log_samples` for paired per-question analysis. When multiple GPUs are available, use data parallelism to
split samples across model copies; see the [LM-Eval examples](../llm_eval/README.md) for commands.

## Track downstream quality over time during distillation

Validation KD and CE losses show whether the student is fitting the teacher and validation data, but they do not
necessarily predict downstream accuracy. Keep the Megatron checkpoints saved at validation intervals, export them
to Hugging Face format, and evaluate the resulting checkpoints to see when downstream quality improves, plateaus,
or regresses.

See the [Megatron-Bridge distillation guide](../megatron_bridge/README.md#converting-to-hugging-face-format-optional)
for how to retain and export intermediate distillation checkpoints.

Evaluate the teacher, pruned student, and each exported checkpoint. Follow the
[LM-Eval Harness instructions](../llm_eval/README.md#lm-eval-harness) and use the
[efficient evaluation workflow](#efficient-evaluation-with-lm-eval-harness) to choose limits.

The following experiment pruned Qwen3-8B to 0.7x and distilled the same student for approximately 100 million
tokens using four data recipes. All runs used a global batch size of 8 and sequence length of 4,096. MMLU used 25
questions per subject (1,425 total), and MMLU-Pro used 50 per subject (700 total). The tables show representative
checkpoints; token counts are derived from the consumed fixed-length training sequences.

Measured compute per data recipe on eight H100 80GB GPUs:

| Stage | Checkpoints | Time | GPU use |
|-------|------------:|-----:|---------|
| Distillation to 100M tokens | - | ~2h10m | 8 GPUs |
| MMLU trajectory | 21 | ~51m | 8 GPUs |
| MMLU-Pro trajectory | 13 | ~3h50m | Two checkpoints in parallel, 4 GPUs each |
| Total | - | ~6h50m | Excludes Slurm queue and worker setup |

| Baseline model | MMLU | MMLU-Pro |
|----------------|-----:|---------:|
| Teacher: Qwen3-8B | 74.93% (full) | 58.62% (full) |
| Pruned 0.7x student | 48.69% (full) | 23.09% (full) |

### WikiText

- Dataset: Salesforce/wikitext (`wikitext-103-v1`)
- Teacher CE: 2.6834

| Training tokens | Validation KD | Validation CE | MMLU | MMLU-Pro |
|----------------:|--------------:|--------------:|-----:|---------:|
| 0 | 0.8261 | 3.3458 | 48.69% (full) | 23.09% (full) |
| 0.7M | 0.3031 | 2.6570 | 59.72% | 25.00% |
| 3.3M | 0.2343 | 2.6091 | 63.58% | 29.29% |
| 39.3M | 0.1495 | 2.5665 | 65.89% | 39.86% |
| 78.6M | 0.1315 | 2.5699 | 66.46% | 39.57% |
| 100.0M | 0.1291 | 2.5863 | 67.30% | 40.57% |

### Nemotron v2

- Dataset: nvidia/Nemotron-Post-Training-Dataset-v2 (math and stem)
- Teacher CE: 1.1566

| Training tokens | Validation KD | Validation CE | MMLU | MMLU-Pro |
|----------------:|--------------:|--------------:|-----:|---------:|
| 0 | 0.5187 | 1.4739 | 48.69% (full) | 23.09% (full) |
| 0.7M | 0.1919 | 1.0931 | 58.74% | 13.14% |
| 3.3M | 0.1342 | 1.0550 | 60.56% | 14.29% |
| 39.3M | 0.0675 | 1.0296 | 64.63% | 6.14% |
| 78.6M | 0.0613 | 1.0773 | 65.61% | 7.71% |
| 100.0M | 0.0582 | 1.0516 | 65.75% | 11.29% |

### 50/50 WikiText and Nemotron v2 blend

- Dataset: 50/50 blend of WikiText and Nemotron v2 math and stem
- Teacher CE: 1.9025

| Training tokens | Validation KD | Validation CE | MMLU | MMLU-Pro |
|----------------:|--------------:|--------------:|-----:|---------:|
| 0 | 0.6662 | 2.3780 | 48.69% (full) | 23.09% (full) |
| 0.7M | 0.2479 | 1.8363 | 57.89% | 12.57% |
| 3.3M | 0.1824 | 2.0265 | 62.46% | 23.14% |
| 39.3M | 0.1164 | 1.9157 | 67.44% | 33.86% |
| 78.6M | 0.0973 | 1.8503 | 67.72% | 41.57% |
| 100.0M | 0.0916 | 1.7680 | 68.28% | 41.71% |

### Nemotron 3

- Dataset: Nemotron 3 Nano [distillation blend](#prepare-token-budgeted-data-blends)
- Teacher CE: 1.4702

| Training tokens | Validation KD | Validation CE | MMLU | MMLU-Pro |
|----------------:|--------------:|--------------:|-----:|---------:|
| 0 | 0.6395 | 1.9113 | 48.69% (full) | 23.09% (full) |
| 0.7M | 0.2424 | 1.5910 | 57.05% | 24.86% |
| 3.3M | 0.1604 | 1.5190 | 62.46% | 36.86% |
| 39.3M | 0.0978 | 1.4144 | 67.23% | 45.00% |
| 78.6M | 0.0890 | 1.4112 | 67.93% | 47.14% |
| 100.0M | 0.0845 | 1.4656 | 67.37% | 47.71% |

Interesting observations include:

- All four data recipes recover MMLU to about 66% to 68% by 100 million tokens. The 50/50 blend is numerically
  highest at 68.28%.
- Nemotron 3 produces the strongest MMLU-Pro trajectory, reaching 47.71%.
- Although Nemotron v2 performs poorly alone, its 50/50 blend with WikiText slightly outperforms WikiText alone
  on both final benchmarks.
- Nemotron v2 KD continues to decrease, but its MMLU-Pro score remains below the pruned baseline. The severe
  MMLU-Pro regression appears to come from overfitting to Nemotron v2-style responses, even though MMLU-Pro prompts
  ask the model to answer in a specific multiple-choice style.

## Prepare token-budgeted data blends

Preparing complete distillation datasets can consume unnecessary time and storage during early experiments.
ModelOpt can preserve source weights while preparing only a requested token budget. See
[Prepare token-budgeted data blends](../dataset/MEGATRON_DATA_PREP.md#prepare-token-budgeted-data-blends) for the
configuration format, commands, and generated outputs.

## Planned topics

Future additions can cover:

- Iterative pruning and distillation workflows
