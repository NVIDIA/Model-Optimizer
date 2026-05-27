# Nemotron-3-Nano-30B-A3B: Prune + Distill + Quantize + vLLM Deployment

End-to-end optimization of [NVIDIA-Nemotron-3-Nano-30B-A3B-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16) demonstrating how ModelOpt techniques stack: Minitron structured pruning → Megatron-Bridge knowledge distillation to recover accuracy → FP8 quantization → vLLM deployment and throughput benchmarking. This document covers:

1. **[Data Preparation](#1-data-preparation)** — tokenizing the training blend for distillation
2. **[Pruning](#2-pruning)** — Minitron structured pruning
3. **[Distillation](#3-distillation)** — recovering accuracy via Megatron-Bridge knowledge distillation
4. **[Evaluation](#4-evaluation)** — benchmarking with NeMo Evaluator across MMLU Pro, GPQA Diamond, AIME, and more
5. **[Quantization](#5-quantization)** — FP8 PTQ on the distilled checkpoint using ModelOpt's `examples/llm_ptq/hf_ptq.py` script
6. **[vLLM Inference Benchmarking](#6-vllm-inference-benchmarking)** — throughput comparison of BF16 vs FP8 on a single H100

**Environment:** Container `nvcr.io/nvidia/nemo:26.04`, ModelOpt 0.45.0. See the [Megatron-Bridge README](../../../megatron_bridge/README.md) for environment setup (including ModelOpt mount path) and container usage.

## Results

![Benchmark Recovery During Knowledge Distillation](figures/learning_curves.png)

| Model | MMLU Pro | GPQA Diamond | LiveCodeBench v6 | AIME 2025 | IFBench | SciCode (Subtask) | Average |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Pruned 22B/A3.0B (no distillation) | 47.1 | 33.5 | 27.4 | 15.5 | 36.9 | 12.1 | 28.8 |
| Distill @ 2.5B tokens (100 iters at 8K Seq Length) | 73.3 | 63.7 | 55.3 | 77.6 | 59.1 | 25.1 | 59.0 |
| Distill @ 20B tokens (800 iters at 8K Seq Length) | 74.8 | 66.0 | 62.3 | 79.6 | 65.4 | 26.1 | 62.4 |
| Distill @ 40B tokens (1600 iters at 8K Seq Length) | 76.4 | 67.2 | 62.3 | 79.8 | 66.0 | 26.6 | 63.1 |
| Distill @ 60B tokens (2400 iters at 8K Seq Length) | 76.1 | 68.1 | 63.6 | 78.8 | 67.3 | 27.0 | 63.5 |
| Distill @ 80B tokens (3200 iters at 8K Seq Length) | 76.5 | 69.1 | 63.9 | 80.7 | 66.5 | 29.0 | 64.3 |
| Distill @ 82.5B tokens (+100 iters at 32K Seq Length) | 76.2 | 69.8 | 64.8 | 87.0 | 68.2 | 27.0 | 65.5 |
| Distill @ 100B tokens (+800 iters at 32K Seq Length) - **BF16** | 76.6 | 69.6 | 66.1 | 87.3 | 68.9 | 28.4 | 66.2 |
| Distill @ 100B tokens + **FP8 Quantize** | 76.3 | 69.8 | 65.5 | 86.0 | 69.7 | 27.9 | 65.9 |
| NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 (official, 31.6B/A3.6B) | 78.0 | 70.3 | 67.9 | 87.1 | 69.1 | 31.8 | 67.4 |

Distillation uses the **30% Pretraining (Code 5, General 20, MATH 5) + 70% Post-training v1/v3 (Math 27, Coding 20, Science 13, IF 5, Tool calling 5)** blend (see [Data Blend](#data-blend) below) with an **80B @ 8K + 20B @ 32K = 100B token** schedule. Blend ablations and long-context phase ablations are in [ABLATIONS.md](ABLATIONS.md).

> [!TIP]
> From the benchmark numbers above, the model is still learning at 100B tokens and that further training (or a higher-quality data blend) would continue to close the gap to the original 31.6B/A3.6B model.

> [!NOTE]
> Exact numbers may vary depending on deployment and evaluation setup. All models above (including the official model) were evaluated once with the same [evaluation setup](#4-evaluation) for fair comparison. These numbers may differ from those reported on the official [Nemotron-3-Nano-30B-A3B-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16) HuggingFace model card.

---

## Steps to Reproduce

### 1. Data Preparation

See [examples/dataset/MEGATRON_DATA_PREP.md](../../../dataset/MEGATRON_DATA_PREP.md) for tokenization commands for all datasets used in this blend.

For this experiment: `TOKENIZER=nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`, `OUTPUT_DIR=tokenized_nemotron_3`.

> [!NOTE]
> Compared to experiments in [NVIDIA-Nemotron-Nano-9B-v2](../NVIDIA-Nemotron-Nano-9B-v2/README.md), we use `Nemotron-SFT-Math-v3` instead of `Nemotron-Math-v2 / high_part01` since it is higher quality with full reasoning traces.

#### Data Blend

**30% Pretraining (Code 5, General 20, MATH 5) + 70% Post-training v1/v3 (Math 27, Coding 20, Science 13, IF 5, Tool calling 5)**

| Dataset                                               | Tokens | Weight | Notes                                          |
| ----------------------------------------------------- | ------ | ------ | ---------------------------------------------- |
| Nemotron-Pretraining-SFT-v1 / Code (10M samples)      | 7B     | 5      | Pretraining code                               |
| Nemotron-Pretraining-SFT-v1 / General (10M samples)   | 16B    | 20     | Upweighted to close MMLU gap                   |
| Nemotron-Pretraining-SFT-v1 / MATH (10M samples)      | 13B    | 5      | Pretraining math                               |
| Nemotron-Math-v2 / high_part00                        | 13B    | 10     | Hard math reasoning                            |
| Nemotron-SFT-Math-v3 / train                          | 52B    | 17     | Hard math reasoning with full reasoning traces |
| Nemotron-SFT-Competitive-Programming-v2 / python_00   | 7B     | 15     | Python reasoning traces                        |
| Nemotron-SFT-Competitive-Programming-v2 / cpp_00      | 7B     | 5      | C++ reasoning traces                           |
| Nemotron-Post-Training-Dataset-v1 / stem (5M samples) | 22B    | 8      | Broad STEM                                     |
| Nemotron-Science-v1 / MCQ                             | 0.5B   | 3      | GPQA MCQ format alignment                      |
| Nemotron-Science-v1 / RQA                             | 0.3B   | 2      | GPQA format diversity                          |
| Nemotron-SFT-IF-Chat-v2 / reasoning_on                | 2B     | 3      | Instruction following (thinking on)            |
| Nemotron-SFT-IF-Chat-v2 / reasoning_off               | 1B     | 2      | Instruction following (thinking off)           |
| Nemotron-Agentic-v1 / tool_calling                    | 1B     | 5      | Tool-use scaffolding; helps SciCode / GPQA     |

<details>
<summary>Data blend for distillation (click to expand)</summary>

```bash
DATA_BLEND=" \
5  tokenized_nemotron_3/nvidia--Nemotron-Pretraining-SFT-v1_Nemotron-SFT-Code_train_text_max10000000 \
20 tokenized_nemotron_3/nvidia--Nemotron-Pretraining-SFT-v1_Nemotron-SFT-General_train_text_max10000000 \
5  tokenized_nemotron_3/nvidia--Nemotron-Pretraining-SFT-v1_Nemotron-SFT-MATH_train_text_max10000000 \
10 tokenized_nemotron_3/nvidia--Nemotron-Math-v2_default_high_part00_messages \
17 tokenized_nemotron_3/nvidia--Nemotron-SFT-Math-v3_default_train_messages \
15 tokenized_nemotron_3/competitive_programming_python_00_messages \
5  tokenized_nemotron_3/competitive_programming_cpp_00_messages \
8  tokenized_nemotron_3/nvidia--Nemotron-Post-Training-Dataset-v1_default_stem_messages_max5000000 \
3  tokenized_nemotron_3/MCQ_messages \
2  tokenized_nemotron_3/RQA_messages \
3  tokenized_nemotron_3/reasoning_on_messages \
2  tokenized_nemotron_3/reasoning_off_messages \
5  tokenized_nemotron_3/nvidia--Nemotron-Agentic-v1_tool_calling_messages \
"
```

</details>

#### General Guidelines

The optimal blend is 30% pretraining and 70% post-training data. Exact proportions may vary depending on the benchmarks you care about. The blend above was designed to maximize recovery on popular General Knowledge, Reasoning, Instruction Following, and Tool Calling benchmarks. The key design decisions were:

- **30% pretraining data** closes the MMLU gap that arises from training exclusively on reasoning-heavy post-training data. The General split (20%) is upweighted specifically to recover general knowledge recall.
- **Math (27%)** is the largest post-training category because AIME and MMLU Pro respond strongly to more math reasoning tokens. We use a mix of `Nemotron-Math-v2` and `Nemotron-SFT-Math-v3` for higher quality math reasoning signal with full reasoning traces.
- **Science (13%)** uses `Nemotron-Post-Training-Dataset-v1 / stem` as the primary source for volume and GPQA stability, with small allocations to `Nemotron-Science-v1` MCQ/RQA subsets for format alignment with GPQA's multiple-choice structure.
- **Instruction following (5%)** saturates quickly so a small allocation is sufficient.
- **Tool calling (5%)** uses `Nemotron-Agentic-v1 / tool_calling`. Our evals run with `--enable-auto-tool-choice`, so the student needs explicit exposure to function-call schemas; this helps SciCode (heavy Python tool use) and GPQA Diamond (which can benefit from calculator tools).

This blend intentionally omits capabilities not targeted in this experiment (e.g. multilingual, SWE). Depending on what benchmarks matter for your use case, you can substitute or add datasets from the [Nemotron Post-Training v3 collection](https://huggingface.co/collections/nvidia/nemotron-post-training-v3), for example:

| Capability | Relevant datasets |
| --- | --- |
| Multilingual | `Nemotron-SFT-Multilingual-v1` |
| Software engineering (SWE) | `Nemotron-SFT-SWE-v2` |
| Safety / alignment | `Nemotron-SFT-Safety-v1` |

When adding new datasets, reduce weights of lower-priority categories proportionally to keep the total at 100%.

---

### 2. Pruning

Here we prune the model from 31.6B/A3.6B to 3.0B active parameters.

Run on **1 node with 8x H100** (~1 hour)

<details>
<summary>Pruning command (click to expand)</summary>

```bash
torchrun --nproc_per_node 8 /opt/Model-Optimizer/examples/megatron_bridge/prune_minitron.py \
  --pp_size 8 \
  --hf_model_name_or_path nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
  --trust_remote_code \
  --prune_target_params 28e9 \
  --prune_target_active_params 3e9 \
  --hparams_to_skip num_attention_heads \
  --seq_length 8192 \
  --output_hf_path /path/to/Nemotron-3-Nano-30B-A3B-Pruned-A3.0B \
  --top_k 20 \
  --max_depth_pruning 0.15 \
  --max_width_pruning 0.30 \
  --prune_score_func mmlu_10pct_bs32 \
  --num_layers_in_first_pipeline_stage 5 \
  --num_layers_in_last_pipeline_stage 5
```

Non-default arguments:

- `--hparams_to_skip num_attention_heads` (default: none) — attention heads pruning is harder to recover, hence skipped
- `--seq_length 8192` (default: 4096) — dataset has longer sequences
- `--prune_target_active_params 3e9` — MoE-specific; the **primary** pruning constraint — targets active params rather than total params, which is what matters for MoE inference cost
- `--prune_target_params 28e9` — upper bound on total params only; the actual pruned model total can range anywhere from ~20B to 28B depending on which architecture wins — see pruning logs below for the top 20 candidates. You may also skip this argument all together for simplicity.
- `--top_k 20` (default: 10) — larger candidate pool for better architecture search
- `--max_depth_pruning 0.15` (default: 0.20) — tighter constraint since candidates with 42–46 layers universally fail for this model
- `--max_width_pruning 0.30` (default: 0.40) — tighter constraint to prevent head_dim≤48 and hidden=2048 dead zones
- `--prune_score_func mmlu_10pct_bs32` (default: `mmlu_10pct_bs1`) — batch_size=32 for ~3–4× faster candidate scoring
- `--num_layers_in_first_pipeline_stage 5 --num_layers_in_last_pipeline_stage 5` — Uneven pipeline parallelism since 52 layers is not divisible by 8 GPUs

**NOTE**: The tighter search space constraints here (`--max_depth_pruning`, `--max_width_pruning`) are specific to Nemotron hybrid models (Mamba + Attention + MoE). Unlike standard transformers which expose only layers/hidden/attention/FFN dimensions, these models add Mamba-specific dimensions (`mamba_num_heads`, `mamba_head_dim`) and MoE dimensions (`num_moe_experts`, `moe_ffn_hidden_size`, `moe_shared_expert_intermediate_size`), making the combined search space much larger. The default 40%/20% bounds cast too wide a net and waste compute on dead-zone architectures.

See [ABLATIONS.md](ABLATIONS.md#pruning) for the full architecture search analysis across various candidates.
</details>

<details>
<summary>Pruning logs (top 20 candidates, best subnet, layer patterns) (click to expand)</summary>

```text
╭──────────────────────────────────────────────────── Original Model Stats ─────────────────────────────────────────────────────╮
│ Total Parameters                              31.58B                                                                          │
│ Active Parameters                             3.58B                                                                           │
│ Memory (BF16, seq_length=8192, batch_size=1)  weights: 60230.1 MB, kv_cache: 48.0 MB, mamba_state: 23.8 MB, Total: 60301.9 MB │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

                              Search Space
                   (≤30% width / ≤15% depth pruning)
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Hyperparameter                      ┃ Choices                        ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ num_layers                          │ [46, 48, 50, 52]               │
│ hidden_size                         │ [2048, 2304, 2560, 2688]       │
│ mamba_num_heads                     │ [48, 56, 64]                   │
│ mamba_head_dim                      │ [48, 56, 64]                   │
│ num_moe_experts                     │ [96, 104, 112, 120, 128]       │
│ moe_ffn_hidden_size                 │ [1536, 1792, 1856]             │
│ moe_shared_expert_intermediate_size │ [2816, 3072, 3328, 3584, 3712] │
├─────────────────────────────────────┼────────────────────────────────┤
│ Search space size                   │ 10800                          │
└─────────────────────────────────────┴────────────────────────────────┘

Top 20 Candidates with Scores
┏━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━┓
┃  # ┃ export_config                                                                                                         ┃ active_params ┃ params ┃  score ┃
┡━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━┩
│  1 │ {'num_layers': 46, 'hidden_size': 2560, 'mamba_num_heads': 56, 'mamba_head_dim': 64, 'num_moe_experts': 120,          │         3.00B │ 27.06B │ 0.3399 │
│    │ 'moe_ffn_hidden_size': 1792, 'moe_shared_expert_intermediate_size': 3072}                                             │               │        │        │
│  2 │ {'num_layers': 48, 'hidden_size': 2560, 'mamba_num_heads': 56, 'mamba_head_dim': 56, 'num_moe_experts': 112,          │         3.00B │ 25.37B │ 0.4650 │
│    │ 'moe_ffn_hidden_size': 1792, 'moe_shared_expert_intermediate_size': 3072}                                             │               │        │        │
│  3 │ {'num_layers': 46, 'hidden_size': 2560, 'mamba_num_heads': 64, 'mamba_head_dim': 56, 'num_moe_experts': 112,          │         3.00B │ 25.37B │ 0.2343 │
│    │ 'moe_ffn_hidden_size': 1792, 'moe_shared_expert_intermediate_size': 3072}                                             │               │        │        │
│  4 │ {'num_layers': 52, 'hidden_size': 2688, 'mamba_num_heads': 56, 'mamba_head_dim': 48, 'num_moe_experts': 96,           │         3.00B │ 20.09B │ 0.2552 │
│    │ 'moe_ffn_hidden_size': 1536, 'moe_shared_expert_intermediate_size': 3072}                                             │               │        │        │
│  5 │ {'num_layers': 52, 'hidden_size': 2688, 'mamba_num_heads': 48, 'mamba_head_dim': 56, 'num_moe_experts': 104,          │         3.00B │ 21.61B │ 0.2601 │
│    │ 'moe_ffn_hidden_size': 1536, 'moe_shared_expert_intermediate_size': 3072}                                             │               │        │        │
│  6 │ {'num_layers': 52, 'hidden_size': 2560, 'mamba_num_heads': 48, 'mamba_head_dim': 64, 'num_moe_experts': 96,           │         3.00B │ 19.28B │ 0.3762 │
│    │ 'moe_ffn_hidden_size': 1536, 'moe_shared_expert_intermediate_size': 3712}                                             │               │        │        │
│  7 │ {'num_layers': 52, 'hidden_size': 2304, 'mamba_num_heads': 64, 'mamba_head_dim': 64, 'num_moe_experts': 104,          │         3.00B │ 22.28B │ 0.4783 │
│    │ 'moe_ffn_hidden_size': 1856, 'moe_shared_expert_intermediate_size': 3072}                                             │               │        │        │
│  8 │ {'num_layers': 52, 'hidden_size': 2560, 'mamba_num_heads': 48, 'mamba_head_dim': 48, 'num_moe_experts': 96,           │         3.00B │ 21.99B │ 0.2420 │
│    │ 'moe_ffn_hidden_size': 1792, 'moe_shared_expert_intermediate_size': 3328}                                             │               │        │        │
│  9 │ {'num_layers': 50, 'hidden_size': 2560, 'mamba_num_heads': 48, 'mamba_head_dim': 48, 'num_moe_experts': 112,          │         3.00B │ 25.37B │ 0.2399 │
│    │ 'moe_ffn_hidden_size': 1792, 'moe_shared_expert_intermediate_size': 3712}                                             │               │        │        │
│ 10 │ {'num_layers': 50, 'hidden_size': 2560, 'mamba_num_heads': 48, 'mamba_head_dim': 48, 'num_moe_experts': 112,          │         3.00B │ 26.17B │ 0.2601 │
│    │ 'moe_ffn_hidden_size': 1856, 'moe_shared_expert_intermediate_size': 3328}                                             │               │        │        │
│ 11 │ {'num_layers': 46, 'hidden_size': 2560, 'mamba_num_heads': 56, 'mamba_head_dim': 64, 'num_moe_experts': 112,          │         3.00B │ 25.37B │ 0.2503 │
│    │ 'moe_ffn_hidden_size': 1792, 'moe_shared_expert_intermediate_size': 3072}                                             │               │        │        │
│ 12 │ {'num_layers': 48, 'hidden_size': 2560, 'mamba_num_heads': 56, 'mamba_head_dim': 56, 'num_moe_experts': 104,          │         3.00B │ 23.68B │ 0.4329 │
│    │ 'moe_ffn_hidden_size': 1792, 'moe_shared_expert_intermediate_size': 3072}                                             │               │        │        │
│ 13 │ {'num_layers': 46, 'hidden_size': 2688, 'mamba_num_heads': 64, 'mamba_head_dim': 64, 'num_moe_experts': 128,          │         3.00B │ 26.17B │ 0.2587 │
│    │ 'moe_ffn_hidden_size': 1536, 'moe_shared_expert_intermediate_size': 2816}                                             │               │        │        │
│ 14 │ {'num_layers': 46, 'hidden_size': 2560, 'mamba_num_heads': 64, 'mamba_head_dim': 56, 'num_moe_experts': 104,          │         3.00B │ 23.68B │ 0.2336 │
│    │ 'moe_ffn_hidden_size': 1792, 'moe_shared_expert_intermediate_size': 3072}                                             │               │        │        │
│ 15 │ {'num_layers': 52, 'hidden_size': 2688, 'mamba_num_heads': 48, 'mamba_head_dim': 56, 'num_moe_experts': 96,           │         3.00B │ 20.09B │ 0.2559 │
│    │ 'moe_ffn_hidden_size': 1536, 'moe_shared_expert_intermediate_size': 3072}                                             │               │        │        │
│ 16 │ {'num_layers': 52, 'hidden_size': 2304, 'mamba_num_heads': 64, 'mamba_head_dim': 64, 'num_moe_experts': 96,           │         3.00B │ 20.70B │ 0.4608 │
│    │ 'moe_ffn_hidden_size': 1856, 'moe_shared_expert_intermediate_size': 3072}                                             │               │        │        │
│ 17 │ {'num_layers': 50, 'hidden_size': 2560, 'mamba_num_heads': 48, 'mamba_head_dim': 48, 'num_moe_experts': 104,          │         3.00B │ 23.68B │ 0.2455 │
│    │ 'moe_ffn_hidden_size': 1792, 'moe_shared_expert_intermediate_size': 3712}                                             │               │        │        │
│ 18 │ {'num_layers': 50, 'hidden_size': 2560, 'mamba_num_heads': 48, 'mamba_head_dim': 48, 'num_moe_experts': 104,          │         3.00B │ 24.42B │ 0.2503 │
│    │ 'moe_ffn_hidden_size': 1856, 'moe_shared_expert_intermediate_size': 3328}                                             │               │        │        │
│ 19 │ {'num_layers': 48, 'hidden_size': 2560, 'mamba_num_heads': 48, 'mamba_head_dim': 48, 'num_moe_experts': 120,          │         3.00B │ 27.92B │ 0.2587 │
│    │ 'moe_ffn_hidden_size': 1856, 'moe_shared_expert_intermediate_size': 3712}                                             │               │        │        │
│ 20 │ {'num_layers': 46, 'hidden_size': 2560, 'mamba_num_heads': 56, 'mamba_head_dim': 64, 'num_moe_experts': 104,          │         3.00B │ 23.68B │ 0.2469 │
│    │ 'moe_ffn_hidden_size': 1792, 'moe_shared_expert_intermediate_size': 3072}                                             │               │        │        │
└────┴───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┴───────────────┴────────┴────────┘

╭──────────────────────────────────────────────────────────────────────── Best Subnet ─────────────────────────────────────────────────────────────────────────╮
│ export_config  {'num_layers': 52, 'hidden_size': 2304, 'mamba_num_heads': 64, 'mamba_head_dim': 64, 'num_moe_experts': 104, 'moe_ffn_hidden_size': 1856,     │
│                'moe_shared_expert_intermediate_size': 3072}                                                                                                  │
│ active_params  3.00B                                                                                                                                         │
│ params         22.28B                                                                                                                                        │
│ score          0.4783                                                                                                                                        │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

Original hybrid_layer_pattern: MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME
Pruned hybrid_layer_pattern:   MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME

╭───────────────────────────────────────────────────── Pruned Model Stats ──────────────────────────────────────────────────────╮
│ Total Parameters                              22.28B                                                                          │
│ Active Parameters                             3.00B                                                                           │
│ Memory (BF16, seq_length=8192, batch_size=1)  weights: 42489.7 MB, kv_cache: 48.0 MB, mamba_state: 23.8 MB, Total: 42561.6 MB │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

</details>

> [!TIP]
> Here we skip the Knowledge Distillation (KD) step for candidates for simplicity. If you want to find a better pruned model, you can take few top candidates' `export_config` from the logs above (where score is in similar range as the best subnet) and then export all models separately and perform KD for ~2B tokens on each of them before selecting the best subnet based on your desired metrics. See [ABLATIONS.md — 1st vs 2nd best candidate](ABLATIONS.md#distillation-results-1st-best-vs-2nd-best-pruning-candidate) for a concrete comparison of distillation results across the top two candidates.

> [!NOTE]
> Copy the `nano_v3_reasoning_parser.py` file from the original HuggingFace checkpoint to the pruned model for evaluation with tool-calling below.

---

### 3. Distillation

Distillation is run in two phases: an 80B-token phase at 8K sequence length, followed by a 20B-token long-context phase at 32K sequence length. The two phases are launched as separate runs with an intermediate Megatron→HF checkpoint conversion, because the long-context phase changes `seq_length`, `gbs`, and `cp_size` — Megatron's checkpoint resume bookkeeping (sample counter is in absolute samples, iteration counter is in iter-units tied to `gbs`) does not handle a mid-run `gbs` change cleanly.

Minimum hardware: **4 nodes × 8x H100 (32 GPUs)** for the 8K phase — required by `TP=4 × EP=8`. The 32K phase additionally requires context parallel to fit the longer sequence, doubling the minimum to **8 nodes × 8x H100 (64 GPUs)**. On **96 nodes × 8x H100 (768 GPUs total)**, it takes ~900 H100 GPU-hours per 10B tokens (400 iters), i.e. ~70 min wall-clock per 10B tokens on 96 nodes. Full schedule (80B @ 8K + 20B @ 32K = 100B tokens, 4k total steps) takes ~9k H100 GPU-hours (~12 hours wall-clock).

#### 3a. Phase 1 — 80B tokens @ 8K seq length

<details>
<summary>Phase 1 distillation command (click to expand)</summary>

```bash
python -u /opt/Model-Optimizer/examples/megatron_bridge/distill.py \
    --teacher_hf_path nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
    --student_hf_path /path/to/Nemotron-3-Nano-30B-A3B-Pruned-A3.0B \
    --trust_remote_code \
    --tp_size 4 \
    --ep_size 8 \
    --data_paths "${DATA_BLEND}" \
    --data_path_to_cache /path/to/cache \
    --seq_length 8192 \
    --mbs 1 \
    --gbs 3072 \
    --train_iters 3200 \
    --lr 1e-4 \
    --min_lr 1e-5 \
    --lr_warmup_iters 25 \
    --eval_interval 200 \
    --eval_iters 8 \
    --log_interval 10 \
    --output_dir /path/to/distill_output_phase1_8k

# Optional: Weights & Biases logging
#     --wandb_project <wandb_project> \
#     --wandb_entity <wandb_entity> \
#     --wandb_exp_name <wandb_exp_name>
```

Non-default arguments:

- `--seq_length 8192` (default: 4096)
- `--gbs 3072` (default: 768) — matches the original Nemotron-3-Nano-30B training GBS from the paper, kept to preserve the training distribution
- `--train_iters 3200` — 80B tokens at GBS 3072 × seq_length 8192
- `--lr 1e-4 --min_lr 1e-5 --lr_warmup_iters 25` — cosine fully decays over 3200 iters; the model is approaching saturation at 8K by this point (see [ABLATIONS.md — 8K trajectory](ABLATIONS.md#effect-of-data-blend-tool_calling)).
- `--eval_interval 200` (default: 100) — less frequent eval to save compute
- `--eval_iters 8` (default: 32) — since GBS is 4× larger than default

All other arguments use defaults.
</details>

#### 3b. Convert Phase 1 final checkpoint to HuggingFace format

Phase 2 starts as a separate run from a fresh HuggingFace student checkpoint, so the final Phase 1 Megatron checkpoint must be exported to HF first using the Megatron-Bridge conversion script (see [Megatron-Bridge README](../../../megatron_bridge/README.md) for full details). You can also use this same script to convert any intermediate Phase 1 checkpoint to HF format for evaluation along the way.

<details>
<summary>Checkpoint conversion command (click to expand)</summary>

```bash
python /opt/Megatron-Bridge/examples/conversion/convert_checkpoints.py export \
    --hf-model /path/to/Nemotron-3-Nano-30B-A3B-Pruned-A3.0B \
    --megatron-path /path/to/distill_output_phase1_8k/checkpoints/iter_0003200 \
    --hf-path /path/to/distill_output_phase1_8k/checkpoints/hf_iter_0003200
```

</details>

#### 3c. Phase 2 — 20B tokens @ 32K seq length

Phase 2 is a **fresh run** with the Phase 1 final checkpoint as the new student. It uses a different `--seed` so the data blend reshuffles (otherwise the model would see overlapping prefix of the same samples it already saw at 8K). The LR is bumped back up modestly to capture the rapid long-context adaptation observed in [ABLATIONS.md — Effect of long context training](ABLATIONS.md#effect-of-long-context-training).

<details>
<summary>Phase 2 distillation command (click to expand)</summary>

```bash
python -u /opt/Model-Optimizer/examples/megatron_bridge/distill.py \
    --teacher_hf_path nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
    --student_hf_path /path/to/distill_output_phase1_8k/checkpoints/hf_iter_0003200 \
    --trust_remote_code \
    --tp_size 4 \
    --cp_size 2 \
    --ep_size 8 \
    --seed 5678 \
    --data_paths "${DATA_BLEND}" \
    --data_path_to_cache /path/to/cache \
    --seq_length 32768 \
    --mbs 1 \
    --gbs 768 \
    --train_iters 800 \
    --lr 2e-5 \
    --min_lr 1e-5 \
    --lr_warmup_iters 10 \
    --recompute_granularity selective \
    --recompute_modules moe \
    --eval_interval 200 \
    --eval_iters 8 \
    --log_interval 10 \
    --output_dir /path/to/distill_output_phase2_32k
```

Changed arguments from Phase 1:

- `--student_hf_path` — points at the HF export of the Phase 1 final checkpoint
- `--seq_length 32768` — long-context phase
- `--gbs 768` — `seq_length × gbs` product unchanged, so each iter still processes the same number of tokens
- `--cp_size 2` — context parallel is needed to fit the longer sequence; doubles the minimum-hardware footprint to 8 nodes
- `--train_iters 800` — 20B tokens at GBS 768 × seq_length 32768
- `--lr 2e-5 --min_lr 1e-5 --lr_warmup_iters 10` — modest LR bump for the long-context adaptation (Phase 1 ended at fully-decayed LR 1e-5); the 10-iter warmup re-populates Adam moment estimates which restart from zero in a fresh run
- `--recompute_granularity selective --recompute_modules moe` — selective MoE recompute further reduces activation memory at 32K. You may skip this if you have more memory.
- `--seed 5678` — different from the Phase 1 seed (default 1234) so the data blend reshuffles
- `--output_dir /path/to/distill_output_phase2_32k` — must be a **fresh directory** different from Phase 1's, so distill.py's resume mechanism (which auto-loads from `<output_dir>/checkpoints` if it exists) does not pull in stale state

</details>

For multi-node Slurm runs, see the [Megatron-Bridge README](../../../megatron_bridge/README.md#slurm-usage) for details.

> [!NOTE]
> This is pure SFT-style distillation — no RL or online reward signal is used. Adding an RL-based post-training step after distillation is a natural next step that could further improve some of these benchmarks.

---

### 4. Evaluation

The eval config in [nemo_evaluator.yaml](nemo_evaluator.yaml) is for Slurm-based evaluation — it submits a vLLM serving job and runs evals against it. For local model execution and evaluation, refer to the [NeMo Evaluator documentation](https://docs.nvidia.com/nemo/evaluator/latest/) or this [blog](https://huggingface.co/blog/nvidia/nemotron-3-nano-evaluation-recipe).

Before running, update the following fields in the yaml or overwrite them in the command line with `-o <option>=<value>`:

- `execution.hostname` — your Slurm login node hostname
- `execution.account` — your Slurm account
- `deployment.checkpoint_path` — Hugging Face checkpoint path (original, pruned or quantized)
- `evaluation.nemo_evaluator_config.config.params.extra.tokenizer` — same path as `checkpoint_path`

> [!TIP]
> Uncomment `limit_samples` under any task to run a small subset and verify the end-to-end eval pipeline before launching full evals.

```bash
pip install "nemo-evaluator-launcher[all]==0.1.90"

# Set required environment variables:
export HF_TOKEN=<your_huggingface_token>
export SLURM_JOB_DIR=<path_to_slurm_job_output_dir>
export HF_HOME=<path_to_huggingface_cache>
export VLLM_CACHE_ROOT=<path_to_vllm_cache>

# Set additional unused but required environment variables:
export API_KEY=xxxxxx
export INFERENCE_API_KEY=xxxxxx
export OPENAI_CLIENT_ID=xxxxxx
export OPENAI_CLIENT_SECRET=xxxxxx

nemo-evaluator-launcher run --config nemo_evaluator.yaml
```

> [!TIP]
> Run same evals multiple times to get a more stable result.

**Tasks and exact metric names reported in the results table:**

| Benchmark | Library | num_repeats | Metric name |
| --- | --- | --- | --- |
| MMLU Pro | NeMo Evaluator | 1 | `mmlu-pro_pass_at_1_symbolic_correct` |
| GPQA Diamond | NeMo Evaluator | 8 | `gpqa_pass_at_1_avg-of-8_symbolic_correct` |
| LiveCodeBench v6 | NeMo Evaluator | 8 | `livecodebench_pass_at_1_avg-of-8_accuracy` |
| AIME 2025 | NeMo Evaluator | 64 | `aime25_pass_at_1_avg-of-64_symbolic_correct` |
| IFBench | NeMo Evaluator | 8 | `ifbench_pass_at_1_avg-of-8_average_score` |
| SciCode (Subtask) | NeMo Evaluator | 8 | `scicode_pass_at_1_avg-of-8_subtask_accuracy` |

**Key vLLM settings:** Tool calling is enabled via `--enable-auto-tool-choice --tool-call-parser qwen3_coder`.

For more details on NeMo Evaluator, see the [GitHub repo](https://github.com/NVIDIA-NeMo/evaluator) and [documentation](https://docs.nvidia.com/nemo/evaluator/latest/).

---

### 5. Quantization

ModelOpt allows stacking multiple optimization techniques. Here we stack FP8 quantization on top of the pruned and distilled model to get an even more optimized model. See [examples/llm_ptq/README.md](../../../llm_ptq/README.md) for the full PTQ documentation.

Similar to the official [Nemotron-3-Nano-30B-A3B-FP8](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8) model, if you want to quantize the pruned 22B/A3.0B model to FP8, the Mamba, MoE, and MLP layers are quantized to FP8, while the attention layers and the Conv1d components within the Mamba layers are kept in BF16 to avoid accuracy degradation.

This is done with the `mtq.MAMBA_MOE_FP8_CONSERVATIVE_CFG` config defined in [`modelopt/torch/quantization/config.py`](../../../../modelopt/torch/quantization/config.py). To apply this, you need to modify `QUANT_CFG_CHOICES["fp8"]` in [`examples/llm_ptq/hf_ptq.py`](../../../llm_ptq/hf_ptq.py) to use `mtq.MAMBA_MOE_FP8_CONSERVATIVE_CFG`. For a faster model at the cost of a larger accuracy drop, you can use `mtq.MAMBA_MOE_FP8_AGGRESSIVE_CFG` instead.

> [!NOTE]
> You can also quantize to NVFP4 using `mtq.MAMBA_MOE_NVFP4_CONSERVATIVE_CFG` (default) or `mtq.MAMBA_MOE_NVFP4_AGGRESSIVE_CFG` (faster, more accuracy drop), which may require further distillation (QAD) to recover accuracy and Blackwell GPU for deployment.

Calibrate and export the Phase 2 final HF checkpoint to FP8 (takes a few minutes on 8x H100):

```bash
python /opt/Model-Optimizer/examples/llm_ptq/hf_ptq.py \
    --pyt_ckpt_path /path/to/distill_output_phase2_32k/checkpoints/hf_iter_0000800 \
    --export_path /path/to/distill_output_phase2_32k/checkpoints/hf_iter_0000800_fp8 \
    --qformat fp8 \
    --calib_size 1024 \
    --trust_remote_code
```

> [!NOTE]
> Copy the `nano_v3_reasoning_parser.py` file from the original checkpoint to the FP8 checkpoint for evaluation with tool-calling — `hf_ptq.py` does not propagate custom-code Python files.

The quantized checkpoint is directly deployable with [vLLM](https://github.com/vllm-project/vllm), [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) and [SGLang](https://github.com/sgl-project/sglang).

> [!TIP]
> You can run the evaluation using the same `nemo_evaluator.yaml` file for the quantized checkpoint also!

See FP8 vs BF16 results in the [Results](#results) section above.

---

### 6. vLLM Inference Benchmarking

Benchmark throughput using [vLLM](https://github.com/vllm-project/vllm) on a single H100 GPU. Run the command once for each HuggingFace checkpoint. vLLM automatically detects FP8 quantization from the embedded `quantization_config` in `config.json` and applies it with no extra flags needed.

Results on a single H100 (ISL=32768, OSL=1024):

```bash
vllm bench throughput \
    --model <checkpoint_path> \
    --random-input-len 32768 \
    --random-output-len 1024 \
    --trust-remote-code \
    --mamba_ssm_cache_dtype float32 \
    --kv-cache-dtype fp8 \
    --load-format safetensors
```

| Checkpoint | Model loading memory | Output tokens/s | Speedup vs Nemotron-3-Nano-30B-A3B BF16 |
| --- | --- | --- | --- |
| Nemotron-3-Nano-30B-A3B-BF16 (official, 31.6B/A3.6B) | 58.9 GiB | 1,006 | 1.00× |
| Nemotron-3-Nano-30B-A3B-FP8 (official) | 31.4 GiB | 1,404 | 1.40× |
| Nemotron-3-Nano-30B-A3B-Pruned-A3.0B (22B/A3.0B) | 41.5 GiB | 1,301 | 1.29× |
| Nemotron-3-Nano-30B-A3B-Pruned-A3.0B-FP8 | 22.8 GiB | 1,653 | 1.64× |

Pruning alone (BF16 → Pruned-A3.0B BF16) gives a **1.29×** throughput speedup with a 30% memory reduction (58.9 → 41.5 GiB), and FP8 quantization alone (BF16 → FP8) gives a **1.40×** speedup with a 47% memory reduction. Stacking both — pruning + FP8 — compounds to a **1.64×** throughput speedup and a **2.6× memory reduction** (58.9 → 22.8 GiB) relative to the original 30B BF16 model, while preserving most of the benchmark accuracy (see [Results](#results)). The NemotronH hybrid architecture (Mamba + Attention + MoE) moderates the FP8 gain relative to pure-transformer models, since Attention and Conv1d layers are not quantized.
