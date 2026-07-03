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
necessarily predict downstream accuracy. Export the live student at each validation interval and evaluate the
resulting checkpoints to see when downstream quality improves, plateaus, or regresses.

Use `--hf_validation_export_path` with `distill.py` as described in the
[Megatron-Bridge distillation guide](../megatron_bridge/README.md#converting-to-hugging-face-format-optional).

The export path contains one loadable Hugging Face checkpoint per validation iteration:

```text
hf_validation/
├── iter_0000100/
├── iter_0000200/
└── iter_0000300/
```

Evaluate the teacher, pruned student, and each exported checkpoint. Follow the
[LM-Eval Harness instructions](../llm_eval/README.md#lm-eval-harness) and use the
[efficient evaluation workflow](#efficient-evaluation-with-lm-eval-harness) to choose limits.

The following experiment pruned Qwen3-8B to 0.7x and distilled it on either Salesforce/wikitext
(`wikitext-103-v1`) or the math and stem splits of nvidia/Nemotron-Post-Training-Dataset-v2 for 100 iterations,
exporting every 20 iterations. MMLU used 25 samples per subject and MMLU-Pro used 50.

| Model | Iteration | Validation KD | Validation CE | MMLU | MMLU-Pro |
|-------|----------:|--------------:|--------------:|-----:|---------:|
| Teacher: Qwen3-8B | - | - | - | 74.93% (full) | 58.62% (full) |
| Pruned 0.7x student before distillation | 0 | - | - | 48.69% (full) | 23.09% (full) |
| Distilled on Salesforce/wikitext (`wikitext-103-v1`) | 20 | 0.3031 | 2.6570 | 59.72% | 25.00% |
| Distilled on Salesforce/wikitext (`wikitext-103-v1`) | 40 | 0.2935 | 2.6554 | 60.98% | 28.00% |
| Distilled on Salesforce/wikitext (`wikitext-103-v1`) | 60 | 0.2696 | 2.6412 | 62.46% | 27.57% |
| Distilled on Salesforce/wikitext (`wikitext-103-v1`) | 80 | 0.2479 | 2.6262 | 62.74% | 27.14% |
| Distilled on Salesforce/wikitext (`wikitext-103-v1`) | 100 | 0.2343 | 2.6091 | 63.58% | 29.29% |
| Distilled on nvidia/Nemotron-Post-Training-Dataset-v2 (math and stem) | 20 | 0.1919 | 1.0931 | 58.74% | 13.14% |
| Distilled on nvidia/Nemotron-Post-Training-Dataset-v2 (math and stem) | 40 | 0.1731 | 1.0692 | 59.58% | 1.71% |
| Distilled on nvidia/Nemotron-Post-Training-Dataset-v2 (math and stem) | 60 | 0.1510 | 1.0347 | 58.46% | 14.43% |
| Distilled on nvidia/Nemotron-Post-Training-Dataset-v2 (math and stem) | 80 | 0.1343 | 1.0466 | 60.35% | 12.29% |
| Distilled on nvidia/Nemotron-Post-Training-Dataset-v2 (math and stem) | 100 | 0.1342 | 1.0550 | 60.56% | 14.29% |

Interesting observations include:

- On WikiText, both validation losses decrease while MMLU and MMLU-Pro trend upward.
- On Nemotron, KD loss decreases, but CE loss improves only through iteration 60 and then worsens. This may indicate
  diminishing learning signal on the Nemotron validation data; distill longer to determine whether CE has plateaued.
- Investigate the Nemotron iteration-40 MMLU-Pro outlier, which appears despite a stable MMLU score, by inspecting
  saved samples.

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
