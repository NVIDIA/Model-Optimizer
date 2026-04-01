# Knowledge Distillation with Megatron-Bridge

This guide shows how to perform knowledge distillation on Puzzletron-compressed AnyModel checkpoints using Megatron-Bridge.

## Overview

1. Set up the environment with Megatron-Bridge
2. Prepare tokenized dataset
3. Run knowledge distillation training directly from HuggingFace checkpoints
4. Review MMLU evaluation results (before/after distillation)

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
  -w /opt/Model-Optimizer \
  nvcr.io/nvidia/nemo:26.02 \
  /bin/bash
```

## Dataset Preparation

This section describes how to prepare datasets for knowledge distillation. We provide examples using WikiText-103, which is a small dataset that can still produce decent results (see the Qwen3-8B example below showing +10.11 percentage point improvement). For production use, larger datasets like [Nemotron-Post-Training-Dataset-v2](https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v2) are recommended.

### Download and Tokenize Dataset

Download and tokenize the dataset in a single step. This downloads the dataset from HuggingFace, tokenizes it, and saves it in the Megatron format (`.bin` and `.idx` files):

```bash
python -m modelopt.torch.utils.plugins.megatron_preprocess_data \
    --hf_dataset Salesforce/wikitext \
    --hf_name wikitext-103-v1 \
    --hf_split train \
    --output_dir path/to/hf_datasets/wikitext-103-v1 \
    --tokenizer meta-llama/Llama-3.1-8B-Instruct \
    --json_keys text \
    --workers 32
```

This will create:

- `Salesforce--wikitext_wikitext-103-v1_train_text_document.bin` - Binary tokenized data
- `Salesforce--wikitext_wikitext-103-v1_train_text_document.idx` - Index file for the binary data
- `Salesforce--wikitext_wikitext-103-v1_train_text_document/cache/` - Cache directory (created after running distillation)

## Run Knowledge Distillation

Run distillation directly from HuggingFace checkpoints (student and teacher) with tokenized dataset:

```bash
torchrun --nproc_per_node=8 examples/puzzletron/mbridge_distillation/distill_hf.py \
    --student_hf_path /path/to/student/puzzletron/checkpoint \
    --student_hf_model meta-llama/Llama-3.1-8B-Instruct \
    --teacher_hf_path /path/to/teacher/huggingface/checkpoint \
    --data_paths 1.0 /path/to/hf_datasets/wikitext-103-v1/Salesforce--wikitext_wikitext-103-v1_train_text_document \
    --output_dir /path/to/distilled/checkpoint \
    --hf_export_path /path/to/exported/hf/model \
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

- Add `--trust_remote_code` if student or teacher checkpoints need HuggingFace custom modeling code.
- The distilled Megatron-Bridge checkpoint will be saved to `--output_dir/checkpoints/iter_<train_iters>`.
- Add `--hf_export_path` to automatically export the final checkpoint to HuggingFace format after distillation. When exporting, you must also provide `--student_hf_model` as the HuggingFace model ID for the export template (e.g., `meta-llama/Llama-3.1-8B-Instruct`). It should match the base architecture of the student model. The exported model can be evaluated for accuracy using the evaluation tools described in the main [README.md](../README.md#evaluation).
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

- **For production distillation:** Use larger production datasets like [nvidia/Nemotron-Pretraining-SFT-v1](https://huggingface.co/datasets/nvidia/Nemotron-Pretraining-SFT-v1) for better results and to avoid overfitting (see regression case above)
- **Training duration:** Train for more iterations to ensure proper convergence
- **See the [Megatron-Bridge distillation tutorial](https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/megatron_bridge#distillation) for best practices**
