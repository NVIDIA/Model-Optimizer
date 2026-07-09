# ModelOpt for Researchers: Fast Experimentation Workflows

Model optimization research depends on short feedback loops: test a hypothesis cheaply, compare candidates
reproducibly, and spend full-scale compute only on the most promising experiments. This guide collects practical
ModelOpt workflows for that iterative research process.

The guide starts with efficient model evaluation and will grow as additional research workflows are documented.
It complements the feature-specific [examples](../) by connecting them into experimentation strategies rather
than replacing their detailed instructions.

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

## Planned topics

Future additions can cover:

- Iterative pruning and distillation workflows
