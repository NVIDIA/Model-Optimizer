# ModelOpt for Researchers: Fast Experimentation Workflows

Model optimization research depends on short feedback loops: test a hypothesis cheaply, compare candidates
reproducibly, and spend full-scale compute only on the most promising experiments. This guide collects practical
ModelOpt workflows for that iterative research process.

Current workflows include:

- [Efficient model evaluation](#efficient-evaluation-with-lm-eval-harness) with smaller benchmark subsets.
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

## Prepare token-budgeted data blends

Full distillation datasets are often unnecessarily large for testing a pruning or distillation hypothesis. Use
[`prepare_data_blend.py`](../dataset/prepare_data_blend.py) to prepare a smaller weighted blend with a shared token
budget. The utility supports Hugging Face configurations and splits as well as specific JSONL files stored in a
Hugging Face dataset repository.

Define the tokenizer, output directory, and source weights in YAML. Set the optional `target_tokens` field to
prepare a weighted subset, or omit it to prepare every source in full. This example scales the
[Nemotron 3 Nano distillation blend](../megatron_bridge/tutorials/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/README.md#1-data-preparation)
down to one billion tokens while preserving its source weights:

```yaml
tokenizer: nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
output_dir: /path/to/nemotron_3_nano_30b_distillation_blend_1b
# Optional; omit this field to prepare every source in full.
target_tokens: 1000000000
sources:
  - hf_dataset: nvidia/Nemotron-Pretraining-SFT-v1
    config: Nemotron-SFT-Code
    split: train
    max_samples: 10000000
    content_field: text
    weight: 5
  - hf_dataset: nvidia/Nemotron-Pretraining-SFT-v1
    config: Nemotron-SFT-General
    split: train
    max_samples: 10000000
    content_field: text
    weight: 20
  - hf_dataset: nvidia/Nemotron-Pretraining-SFT-v1
    config: Nemotron-SFT-MATH
    split: train
    max_samples: 10000000
    content_field: text
    weight: 5
  - hf_dataset: nvidia/Nemotron-Math-v2
    split: high_part00
    content_field: messages
    weight: 10
  - hf_dataset: nvidia/Nemotron-SFT-Math-v3
    files:
      - data/train.jsonl
    content_field: messages
    weight: 17
  - hf_dataset: nvidia/Nemotron-SFT-Competitive-Programming-v2
    files:
      - data/competitive_programming_python_00.jsonl
    content_field: messages
    weight: 15
  - hf_dataset: nvidia/Nemotron-SFT-Competitive-Programming-v2
    files:
      - data/competitive_programming_cpp_00.jsonl
    content_field: messages
    weight: 5
  - hf_dataset: nvidia/Nemotron-Post-Training-Dataset-v1
    config: default
    split: stem
    max_samples: 5000000
    content_field: messages
    weight: 8
  - hf_dataset: nvidia/Nemotron-Science-v1
    files:
      - data/MCQ.jsonl
    content_field: messages
    weight: 3
  - hf_dataset: nvidia/Nemotron-Science-v1
    files:
      - data/RQA.jsonl
    content_field: messages
    weight: 2
  - hf_dataset: nvidia/Nemotron-SFT-Instruction-Following-Chat-v2
    files:
      - data/reasoning_on.jsonl
    content_field: messages
    weight: 3
  - hf_dataset: nvidia/Nemotron-SFT-Instruction-Following-Chat-v2
    files:
      - data/reasoning_off.jsonl
    content_field: messages
    weight: 2
  - hf_dataset: nvidia/Nemotron-Agentic-v1
    files:
      - data/tool_calling.jsonl
    content_field: messages
    weight: 5
```

Run from the repository root:

```bash
python examples/dataset/prepare_data_blend.py --config blend.yaml
```

The output contains tokenized Megatron `.bin`/`.idx` files, `data_blend.txt` with the weighted paths for training,
and `config.yaml` recording how the blend was generated. The final token count can slightly exceed the target
because the final document from each source is kept whole. See the
[Megatron data preparation guide](../dataset/MEGATRON_DATA_PREP.md) for dataset-specific details.

## Planned topics

Future additions can cover:

- Iterative pruning and distillation workflows
