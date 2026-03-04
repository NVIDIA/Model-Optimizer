# Knowledge Distillation with Megatron-Bridge

This guide shows how to perform knowledge distillation on Puzzletron-compressed AnyModel checkpoints using Megatron-Bridge.

## Overview

1. Set up the environment with Megatron-Bridge
2. Prepare tokenized dataset
3. Run knowledge distillation training directly from HuggingFace checkpoints

## Setup

**Clone Model-Optimizer repo:**

The NeMo container does not include Model-Optimizer examples, so you need to clone the Model-Optimizer repo:

```bash
export MODELOPT_DIR=${PWD}/Model-Optimizer
git clone https://github.com/NVIDIA/Model-Optimizer.git ${MODELOPT_DIR}
```

**Start Docker container:**

Use the [NeMo 26.02 container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo?version=26.02):

```bash
# Recommended to mount a workspace directory for storing datasets and distilled models
docker run --gpus all -it --rm \
  -v /path/to/your/project:/workspace \
  -v ${MODELOPT_DIR}:/opt/Model-Optimizer \
  -v ${MODELOPT_DIR}/modelopt:/opt/venv/lib/python3.12/site-packages/modelopt \
  nvcr.io/nvidia/nemo:26.02 \
  /bin/bash
```

## Dataset Preparation

This section describes how to prepare datasets for knowledge distillation. We provide examples using a toy dataset (WikiText-103) for illustration purposes, and note how to adapt the process for production datasets like Nemotron-Post-Training-Dataset-v2.

> **Note:** The WikiText-103 dataset is a small toy dataset used here only for illustration. For actual knowledge distillation, use a larger, more representative dataset like [Nemotron-Post-Training-Dataset-v2](https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v2).

### Step 1: Download Dataset

First, download the dataset and save it in JSONL format. For WikiText-103, you can use the following script:

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

### Step 2: Tokenize Dataset

Next, tokenize the JSONL dataset using the tokenizer from your model. This converts the text data into token IDs that can be used for training:

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

## Step 1: Run Knowledge Distillation

Run distillation directly from HuggingFace checkpoints (student and teacher) with tokenized dataset:

```bash
# Run from /opt/Model-Optimizer directory inside the container
cd /opt/Model-Optimizer
torchrun --nproc_per_node=8 examples/puzzletron/mbridge_distillation/distill_hf.py \
    --student_hf_path /path/to/student/huggingface/checkpoint \
    --teacher_hf_path /path/to/teacher/huggingface/checkpoint \
    --data_paths 1.0 /path/to/tokenized/dataset \
    --output_dir /workspace/mbridge_distillation/distilled_student \
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

The distilled checkpoint will be saved to `--output_dir`.

**Note:** The script automatically converts HuggingFace checkpoints to Megatron-Bridge format on-the-fly, so no separate import step is needed.
