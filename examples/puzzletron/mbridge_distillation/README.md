# Knowledge Distillation with Megatron-Bridge

This guide shows how to perform knowledge distillation on Puzzletron-compressed AnyModel checkpoints using Megatron-Bridge.

## Overview

1. Set up the environment with Megatron-Bridge
2. Prepare tokenized dataset
3. Run knowledge distillation training directly from HuggingFace checkpoints
4. Review MMLU evaluation results (before/after distillation)

## Setup

**Start Docker container:**

Use the [NeMo 26.02 container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo?version=26.02):

```bash
docker run --gpus all -it --rm \
  -v /path/to/your/project:/workspace \
  nvcr.io/nvidia/nemo:26.02 \
  /bin/bash
```

**Set up the environment inside the container:**

```bash
export PYTHONPATH="/workspace/Model-Optimizer:${PYTHONPATH}"
```

## Dataset Preparation

This section describes how to prepare datasets for knowledge distillation. We provide examples using a toy dataset (WikiText-103) for illustration purposes, and note how to adapt the process for production datasets like Nemotron-Post-Training-Dataset-v2.

> **Note:** The WikiText-103 dataset is a small toy dataset used here only for illustration. For actual knowledge distillation, use a larger, more representative dataset like [Nemotron-Post-Training-Dataset-v2](https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v2).

### Download Dataset

Download the dataset and save it in JSONL format. For WikiText-103, you can use the following script:

```python
# download_hf_wikitext_dataset.py
import json
import os
from datasets import load_dataset

DATA_PATH = "path/to/hf_datasets/wikitext-103-v1"
# Load the WikiText-103 dataset
dataset = load_dataset("wikitext", "wikitext-103-v1", split="train")

# Define the destination folder
os.makedirs(DATA_PATH, exist_ok=True)

# Save splits to JSONL files
with open(f"{DATA_PATH}/wikitext-train.jsonl", "w") as file:
    file.writelines(json.dumps(item) + "\n" for item in dataset)

print(f"Raw dataset saved to {DATA_PATH}/wikitext-train.jsonl")
```

### Tokenize Dataset

Tokenize the JSONL dataset using the tokenizer from your model. This converts the text data into token IDs that can be used for training:

```python
# tokenize_wikitext_dataset.py
from modelopt.torch.utils.plugins import megatron_preprocess_data

DATA_PATH = "path/to/hf_datasets/wikitext-103-v1"
HF_MODEL_NAME_OR_PATH = "path/to/your/model/checkpoint"

megatron_preprocess_data(
    input_path=f"{DATA_PATH}/wikitext-train.jsonl",
    output_dir=DATA_PATH,
    tokenizer_name_or_path=HF_MODEL_NAME_OR_PATH,
    json_keys=["text"],
    workers=32,
    log_interval=100000,
)
```

## Run Knowledge Distillation

Run distillation directly from HuggingFace checkpoints (student and teacher) with tokenized dataset:

```bash
torchrun --nproc_per_node=8 examples/puzzletron/mbridge_distillation/distill_hf.py \
    --student_hf_path /path/to/student/huggingface/checkpoint \
    --teacher_hf_path /path/to/teacher/huggingface/checkpoint \
    --data_paths 1.0 /path/to/tokenized/dataset \
    --output_dir /path/to/distilled/checkpoint \
    --hf-export-path /path/to/exported/hf/model \
    --hf-model meta-llama/Llama-3.1-8B-Instruct \
    --seq_length 4096 \
    --tp_size 8 \
    --pp_size 1 \
    --mbs 1 \
    --gbs 4 \
    --train_iters 100 \
    --lr 0.0001 \
    --min_lr 1e-05 \
    --lr_warmup_iters 10 \
    --eval_interval 10 \
    --eval_iters 10 \
    --log_interval 1
```

**Notes:**

- The distilled Megatron-Bridge checkpoint will be saved to `--output_dir/checkpoints/iter_<train_iters>`.
- Add `--hf-export-path` to automatically export the final checkpoint to HuggingFace format after distillation. When using `--hf-export-path`, you must also provide `--hf-model` to specify the HuggingFace model ID to use as a template for export (e.g., `meta-llama/Llama-3.1-8B-Instruct`). The `--hf-model` should match the base architecture of the student model. The exported model can be evaluated for accuracy using the evaluation tools described in the main [README.md](../README.md#evaluation).
- For production use, use larger datasets like [Nemotron-Pretraining-SFT-v1](https://huggingface.co/datasets/nvidia/Nemotron-Pretraining-SFT-v1) and train for more iterations. See the [Megatron-Bridge distillation tutorial](https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/megatron_bridge#distillation) for best practices.

## MMLU Evaluation Results

This section presents MMLU evaluation results for knowledge distillation experiments compressing Qwen3-8B and Llama-3.1-8B-Instruct.

### Successful Case: Qwen3-8B (80% of original)

Distillation results for a memory-compressed Qwen3-8B checkpoint (80% of original size):

| Model | MMLU | Humanities | Other | Social Sci | STEM |
|-------|------|------------|-------|------------|------|
| 80% pre-distillation | 0.5910 | 0.5046 | 0.6363 | 0.6831 | 0.5855 |
| 80% post-distillation | 0.6921 | 0.5906 | 0.7316 | 0.7975 | 0.7016 |
| Original Qwen3-8B | 0.7493 | 0.6648 | 0.7856 | 0.8385 | 0.7526 |

**Key observations:**

- MMLU accuracy improved from 59.10% to 69.21% (+10.11 percentage points) after distillation
- Achieved with just 100 iterations on WikiText-103, demonstrating efficient knowledge transfer
- Recovery of 64% of the gap to the teacher model (from 59.10% to 69.21%, closing 64% of the gap from 59.10% to 74.93%)
- All individual category scores (Humanities, Other, Social Sciences, STEM) improved significantly

### Successful Case: Llama-3.1-8B-Instruct (50% of original, 56,810 MiB)

Distillation results for a pruned Llama-3.1-8B-Instruct checkpoint (50% of original size, 56,810 MiB memory constraint):

| Model | MMLU | Humanities | Other | Social Sciences | STEM |
|-------|------|------------|-------|-----------------|------|
| Before distillation | 0.2316 | 0.2462 | 0.2292 | 0.2250 | 0.2274 |
| After distillation | 0.2960 | 0.3146 | 0.3085 | 0.2925 | 0.2768 |
| Original Llama-3.1-8B-Instruct | 0.6839 | 0.7231 | 0.7038 | 0.7667 | 0.5911 |

**Key observations:**

- MMLU accuracy (average across all categories) improved from 23.16% to 29.60% (+6.44 percentage points)
- All individual category scores (Humanities, Other, Social Sciences, STEM) improved, demonstrating effective knowledge transfer from teacher to student

### Regression Case: Llama-3.1-8B-Instruct (69% of original, 78,000 MiB)

Distillation results for a pruned Llama-3.1-8B-Instruct checkpoint (approximately 69% of original size, 78,000 MiB memory constraint) showing regression due to overfitting on the small WikiText-103 dataset (evaluated with limit 100):

| Model | MMLU | Humanities | Other | Social Sciences | STEM |
|-------|------|------------|-------|-----------------|------|
| Before distillation | 0.6626 | 0.7069 | 0.6892 | 0.7525 | 0.5574 |
| After distillation | 0.6496 | 0.6862 | 0.6677 | 0.7433 | 0.5532 |
| Original Llama-3.1-8B-Instruct | 0.6839 | 0.7231 | 0.7038 | 0.7667 | 0.5911 |

**Key observations:**

- MMLU accuracy (average across all categories) decreased from 66.26% to 64.96% (-1.30 percentage points) after distillation
- The model overfitted to the small WikiText-103 dataset, causing performance regression
- This demonstrates the critical importance of using larger, more diverse datasets for knowledge distillation

### Recommendations

- **For successful distillation:** Use larger production datasets like [nvidia/Nemotron-Pretraining-SFT-v1](https://huggingface.co/datasets/nvidia/Nemotron-Pretraining-SFT-v1) instead of WikiText-103
- **Training duration:** Train for more iterations to ensure proper convergence
- **See the [Megatron-Bridge distillation tutorial](https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/megatron_bridge#distillation) for best practices**
