# Nemotron-3-Nano-30B-A3B: Prune + Distill + Quantize + vLLM Deployment

End-to-end optimization of [NVIDIA-Nemotron-3-Nano-30B-A3B-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16) demonstrating how ModelOpt techniques stack: Minitron structured pruning → Megatron-Bridge knowledge distillation to recover accuracy → FP8 quantization → vLLM deployment and throughput benchmarking. This document covers:

1. **[Data Preparation](#1-data-preparation)** — tokenizing the training blend for distillation
2. **[Pruning](#2-pruning)** — Minitron structured pruning
3. **[Distillation](#3-distillation)** — recovering accuracy via Megatron-Bridge knowledge distillation
4. **[Evaluation](#4-evaluation)** — benchmarking with NeMo Evaluator across MMLU Pro, GPQA Diamond, AIME, and more
5. **[Quantization](#5-quantization)** — FP8 PTQ on the distilled checkpoint using ModelOpt's `examples/llm_ptq/hf_ptq.py` script
6. **[vLLM Inference Benchmarking](#6-vllm-inference-benchmarking)** — throughput comparison of BF16 vs FP8 on a single H100

**Environment:** Container `nvcr.io/nvidia/nemo:26.02`, ModelOpt 0.45.0. See the [Megatron-Bridge README](../../../megatron_bridge/README.md) for environment setup (including ModelOpt mount path) and container usage.

## Results

TODO

---

## Steps to Reproduce

### 1. Data Preparation

See [examples/dataset/MEGATRON_DATA_PREP.md](../../../dataset/MEGATRON_DATA_PREP.md) for tokenization commands for all datasets used in this blend.

For this experiment: `TOKENIZER=nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`, `OUTPUT_DIR=tokenized_nemotron_3`.

> [!NOTE]
> Compared to experiments in [NVIDIA-Nemotron-Nano-9B-v2](../NVIDIA-Nemotron-Nano-9B-v2/README.md), we use `Nemotron-SFT-Math-v3` instead of `Nemotron-Math-v2 / high_part01` since it is higher quality with full reasoning traces.

#### Data Blend

**30% Pretraining (Code 5, General 20, MATH 5) + 70% Post-training v1/v3 (Math 30, Coding 20, Science 15, IF 5)**

```bash
DATA_BLEND=" \
5  tokenized_nemotron_3/nvidia--Nemotron-Pretraining-SFT-v1_Nemotron-SFT-Code_train_text_max10000000 \
20 tokenized_nemotron_3/nvidia--Nemotron-Pretraining-SFT-v1_Nemotron-SFT-General_train_text_max10000000 \
5  tokenized_nemotron_3/nvidia--Nemotron-Pretraining-SFT-v1_Nemotron-SFT-MATH_train_text_max10000000 \
10 tokenized_nemotron_3/nvidia--Nemotron-Math-v2_default_high_part00_messages \
20 tokenized_nemotron_3/nvidia--Nemotron-SFT-Math-v3_default_train_messages \
15 tokenized_nemotron_3/competitive_programming_python_00_messages \
5  tokenized_nemotron_3/competitive_programming_cpp_00_messages \
10 tokenized_nemotron_3/nvidia--Nemotron-Post-Training-Dataset-v1_default_stem_messages_max5000000 \
3  tokenized_nemotron_3/MCQ_messages \
2  tokenized_nemotron_3/RQA_messages \
3  tokenized_nemotron_3/reasoning_on_messages \
2  tokenized_nemotron_3/reasoning_off_messages \
"
```

| Dataset                                               | Tokens | Weight | Notes                                          |
| ----------------------------------------------------- | ------ | ------ | ---------------------------------------------- |
| Nemotron-Pretraining-SFT-v1 / Code (10M samples)      | 7B     | 5      | Pretraining code                               |
| Nemotron-Pretraining-SFT-v1 / General (10M samples)   | 16B    | 20     | Upweighted to close MMLU gap                   |
| Nemotron-Pretraining-SFT-v1 / MATH (10M samples)      | 13B    | 5      | Pretraining math                               |
| Nemotron-Math-v2 / high_part00                        | 13B    | 10     | Hard math reasoning                            |
| Nemotron-SFT-Math-v3 / train                          | 52B    | 20     | Hard math reasoning with full reasoning traces |
| Nemotron-SFT-Competitive-Programming-v2 / python_00   | 7B     | 15     | Python reasoning traces                        |
| Nemotron-SFT-Competitive-Programming-v2 / cpp_00      | 7B     | 5      | C++ reasoning traces                           |
| Nemotron-Post-Training-Dataset-v1 / stem (5M samples) | 22B    | 10     | Broad STEM                                     |
| Nemotron-Science-v1 / MCQ                             | 0.5B   | 3      | GPQA MCQ format alignment                      |
| Nemotron-Science-v1 / RQA                             | 0.3B   | 2      | GPQA format diversity                          |
| Nemotron-SFT-IF-Chat-v2 / reasoning_on                | 2B     | 3      | Instruction following (thinking on)            |
| Nemotron-SFT-IF-Chat-v2 / reasoning_off               | 1B     | 2      | Instruction following (thinking off)           |

#### General Guidelines

The optimal blend is 30% pretraining and 70% post-training data. Exact proportions may vary depending on the benchmarks you care about. The blend above was designed to maximize recovery on popular General Knowledge, Reasoning, Instruction Following, and Tool Calling benchmarks. The key design decisions were:

- **30% pretraining data** closes the MMLU gap that arises from training exclusively on reasoning-heavy post-training data. The General split (20%) is upweighted specifically to recover general knowledge recall.
- **Math (30%)** is the largest post-training category because AIME and MMLU Pro respond strongly to more math reasoning tokens. We use a mix of `Nemotron-Math-v2` and `Nemotron-SFT-Math-v3` for higher quality math reasoning signal with full reasoning traces.
- **Science (15%)** uses `Nemotron-Post-Training-Dataset-v1 / stem` as the primary source for volume and GPQA stability, with small allocations to `Nemotron-Science-v1` MCQ/RQA subsets for format alignment with GPQA's multiple-choice structure.
- **Instruction following (5%)** saturates quickly so a small allocation is sufficient.

This blend intentionally omits capabilities not targeted in this experiment (e.g. long context and multilingual benchmarks). Depending on what benchmarks matter for your use case, you can substitute or add datasets from the [Nemotron Post-Training v3 collection](https://huggingface.co/collections/nvidia/nemotron-post-training-v3), for example:

| Capability | Relevant datasets |
| --- | --- |
| Multilingual | `Nemotron-SFT-Multilingual-v1` |
| Agentic / tool use | `Nemotron-SFT-Tool-Call-v1`, `Nemotron-SFT-Tool-Call-v2` |
| Software engineering (SWE) | `Nemotron-SFT-SWE-v1` |
| Safety / alignment | `Nemotron-SFT-Safety-v1` |
| Long context | `Nemotron-SFT-Long-Context-v1` |

When adding new datasets, reduce weights of lower-priority categories proportionally to keep the total at 100%.

---

### 2. Pruning

TODO

---

### 3. Distillation

TODO

---

### 4. Evaluation

The eval config in [nemo_evaluator.yaml](nemo_evaluator.yaml) is for Slurm-based evaluation — it submits a vLLM serving job and runs evals against it. For local model execution and evaluation, refer to the [NeMo Evaluator documentation](https://docs.nvidia.com/nemo/evaluator/latest/).

Before running, update the following fields in the yaml:

- `execution.hostname` — your Slurm login node hostname
- `execution.account` — your Slurm account
- `deployment.checkpoint_path` — Hugging Face checkpoint path (original, pruned or quantized)
- `evaluation.nemo_evaluator_config.config.params.extra.tokenizer` — same path as `checkpoint_path`

Set the required environment variables and run:

> [!TIP]
> Uncomment `limit_samples` under any task to run a small subset and verify the end-to-end eval pipeline before launching full evals.

```bash
pip install "nemo-evaluator-launcher[all]==0.1.90"

# Required environment variables
export HF_TOKEN=<your_huggingface_token>
export SLURM_JOB_DIR=<path_to_slurm_job_output_dir>
export HF_HOME=<path_to_huggingface_cache>
export VLLM_CACHE_ROOT=<path_to_vllm_cache>

# Additional unused but required environment variables
export API_KEY=xxxxxx
export INFERENCE_API_KEY=xxxxxx
export OPENAI_CLIENT_ID=xxxxxx
export OPENAI_CLIENT_SECRET=xxxxxx

nemo-evaluator-launcher run --config nemo_evaluator.yaml
```

**Tasks and exact metric names reported in the results table:**

| Benchmark | Tool | Metric name |
| --- | --- | --- |
| MMLU | [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) (5-shot) | `mmlu` |
| MMLU Pro | NeMo Evaluator | `mmlu-pro_pass_at_1_symbolic_correct` |
| GPQA Diamond | NeMo Evaluator | `gpqa_pass_at_1_symbolic_correct` |
| LiveCodeBench v6 | NeMo Evaluator | `livecodebench_pass_at_1_accuracy` |
| AIME 2025 | NeMo Evaluator | `aime25_pass_at_1_symbolic_correct` |
| IFBench | NeMo Evaluator | `ifbench_pass_at_1_average_score` |
| SciCode (Subtask) | NeMo Evaluator | `scicode_pass_at_1_subtask_accuracy` |
| BFCL v3 | NeMo Evaluator | `bfcl_v3_overall_accuracy_accuracy` |
| BFCL v4 | NeMo Evaluator | `bfcl_v4_overall_accuracy_accuracy` |

**Key vLLM settings:** Tool calling is enabled via `--enable-auto-tool-choice --tool-call-parser qwen3_coder`.

For more details on NeMo Evaluator, see the [GitHub repo](https://github.com/NVIDIA-NeMo/evaluator) and [documentation](https://docs.nvidia.com/nemo/evaluator/latest/).

---

### 5. Quantization

TODO

---

### 6. vLLM Inference Benchmarking

TODO
