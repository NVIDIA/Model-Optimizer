# Dataset Preparation

<div align="center">

| **Section** | **Description** | **Link** |
| :------------: | :------------: | :------------: |
| Building Chat Datasets | Scripts to build conversation datasets from Nemotron and other HuggingFace sources | \[[Link](#building-chat-datasets)\] |
| Tokenizing for Megatron Frameworks | Convert JSONL or HF datasets to Megatron binary format for distillation and pre-training | \[[Link](#tokenizing-for-megatron-frameworks)\] |

</div>

## Building Chat Datasets

Utilities for building conversation datasets from NVIDIA Nemotron Post-Training
collections and other HuggingFace sources.  These scripts produce datasets in
**standard OpenAI chat format** (`{"messages": [{"role": ..., "content": ...}]}`)
and can be used for any downstream fine-tuning task — SFT, distillation,
speculative decoding draft-model training, etc.

### Files

| File | Description |
| --- | --- |
| `make_nemotron_ptv3_dataset.py` | Build a dataset from the [Nemotron PT v3 collection](https://huggingface.co/collections/nvidia/nemotron-post-training-v3) using a configurable YAML mix |
| `make_nemotron_ptv2_dataset.py` | Build a dataset from [Nemotron-Post-Training-Dataset-v2](https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v2) |
| `make_dataset.py` | General-purpose mixer for arbitrary HuggingFace datasets (mtbench, sharegpt, ultrachat, magpie, etc.) |
| `conversation_utils.py` | Shared utilities: augmentation, role normalization, assistant-turn stripping |
| `add_nemotron_chat.py` | Add Nemotron v2 chat conversations to an existing dataset |
| `augmentations.yaml` | Augmentation variants (language redirects, style hints) for `make_nemotron_pt*.py` |
| `nemotron_ptv3_datasets.yaml` | Dataset mix config for `make_nemotron_ptv3_dataset.py` |
| `example_data_config.yaml` | Example YAML config for `make_dataset.py` |

### Quick Start

#### Install dependencies

```bash
pip install nvidia-modelopt[hf]
hf auth login --token <your token> # required for gated datasets
```

#### Build a Nemotron PT v3 dataset

```bash
# Synthetic data generation inputs (strips last assistant turn so a model can regenerate it)
python make_nemotron_ptv3_dataset.py --output-dir /tmp/ptv3_gen

# Full conversations for direct SFT training
python make_nemotron_ptv3_dataset.py --mode train --output-dir /tmp/ptv3_train

# Use a custom dataset mix
python make_nemotron_ptv3_dataset.py --config my_mix.yaml --output-dir /tmp/ptv3_custom
```

#### Build a Nemotron PT v2 dataset

```bash
python make_nemotron_ptv2_dataset.py --output-dir /tmp/ptv2_gen
python make_nemotron_ptv2_dataset.py --mode train --output-dir /tmp/ptv2_train
```

#### Build a general-purpose mixed dataset

```bash
python make_dataset.py --config example_data_config.yaml --output-dir /tmp/mixed
```

### Dataset Modes

Both `make_nemotron_pt*.py` scripts support two modes:

| Mode | Description | Use case |
| --- | --- | --- |
| `generate` (default) | Strips assistant turns, optionally augments prompts | Input data for synthetic generation (query a target model to produce training responses) |
| `train` | Keeps all turns, normalizes to clean OpenAI format | Direct SFT / distillation training |

### Synthetic Generation Pipeline

The `generate` mode produces conversation skeletons that are fed to a target model
via `tools/launcher/common/query.py` (vLLM or TRT-LLM).  The output becomes training
data for a draft model (e.g. EAGLE3 speculative decoding) or a distilled student:

```bash
make_nemotron_ptv3_dataset.py --mode generate  →  skeleton.jsonl
        ↓
query.py  (target model generates responses turn-by-turn)
        ↓
training data for draft model / student
```

### Augmentations

`augmentations.yaml` defines language-redirect and style-hint variants that are
applied cyclically across the dataset.  Each enabled entry produces one augmented
copy of the source rows.

To customize augmentations:
- **Disable** a variant: add `enabled: false`
- **Add** a language redirect: append a `user_suffix` entry
- **Add** a system prompt: append a `system_prompt` entry

```yaml
augmentations:
  - type: user_suffix
    text: " Please reply in French instead of English."
  - type: system_prompt
    content: "You are a helpful assistant."
    enabled: false   # disable without deleting
```

### Dataset Mix Config (`nemotron_ptv3_datasets.yaml`)

Edit this file to add, remove, or re-weight datasets without touching the script:

```yaml
datasets:
  - repo_id: nvidia/Nemotron-Math-v2
    splits: [high_part00, high_part01]
    cap_per_split: 200000
    augment: true

  - repo_id: nvidia/OpenMathReasoning-mini
    splits: [train]
    augment: false   # multilingual — skip language-redirect augmentation
```

### Output Format

Every output row is a JSONL object with a single `messages` key:

```json
{"messages": [
  {"role": "system",    "content": "You are a helpful assistant."},
  {"role": "user",      "content": "What is 2+2?"},
  {"role": "assistant", "content": "4"}
]}
```

In `generate` mode, assistant turns are stripped so the row ends with a user turn.

## Tokenizing for Megatron Frameworks

The distillation and pre-training scripts in Megatron-Bridge or Megatron-LM expect data pre-tokenized in Megatron's binary indexed format (`.bin` / `.idx`).
Use the `megatron_preprocess_data` utility to tokenize any JSONL or Hugging Face dataset.
The tokenization scripts below prints the list of output prefixes (e.g. `tokenized_qwen3/data1_text`) that you can use for the `data_paths` argument (with relative weights on different files) in Megatron training scripts.

**Important Notes:**

- For Pretraining / raw-text data (`text` key) — use `--append_eod` so Megatron can tell where documents end when concatenating them into long sequences.
- For Post-training chat data (`messages` key) — omit `--append_eod`; the chat template already appends EOS at the end of each conversation.
- Set `--max_sequence_length 256_000` to avoid rare OOM errors if some text is very long.

### From JSONL files

```bash
python -m modelopt.torch.utils.plugins.megatron_preprocess_data \
    --jsonl_paths /path/to/data1.jsonl /path/to/data2.jsonl ... \
    --json_keys text \
    --tokenizer Qwen/Qwen3-0.6B \
    --output_dir tokenized_qwen3 \
    --workers 32 \
    --append_eod
```

```bash
python -m modelopt.torch.utils.plugins.megatron_preprocess_data \
    --jsonl_paths /path/to/sft_data.jsonl \
    --json_keys messages \
    --tokenizer Qwen/Qwen3-0.6B \
    --output_dir tokenized_qwen3 \
    --workers 32
```

Instead of `--jsonl_paths`, pass `--input_dir /path/to/dir` to tokenize all JSONL files in a directory (`.jsonl` and `.jsonl.gz` are both supported).

### From Hugging Face Hub

To tokenize a dataset directly from Hugging Face Hub:

```bash
python -m modelopt.torch.utils.plugins.megatron_preprocess_data \
    --hf_dataset nvidia/Nemotron-Pretraining-SFT-v1 \
    --hf_name Nemotron-SFT-Code \
    --hf_split train \
    --hf_max_samples_per_split 10_000_000 \
    --json_keys text \
    --tokenizer Qwen/Qwen3-0.6B \
    --output_dir tokenized_qwen3 \
    --workers 32 \
    --append_eod
```

Omit `--hf_name` to process all subsets, `--hf_split` for all splits, or `--hf_max_samples_per_split` for all samples.
To quickly test, use [nvidia/Nemotron-Pretraining-Dataset-sample](https://huggingface.co/datasets/nvidia/Nemotron-Pretraining-Dataset-sample).

For **very large datasets** (tens of millions of documents), add `--hf_streaming --hf_max_samples_per_split <num_samples>` to avoid downloading the full dataset — only the rows actually consumed are fetched.

> **Performance note:** Non-streaming mode downloads all Parquet shards once and caches them as Arrow files on disk.
> Re-runs read from cache and are much faster.
> Streaming re-downloads on every run with no cache, so it is slower for full-dataset processing.

### Nemotron Post-Training v3 (`reasoning_content`)

v3 datasets include a `reasoning_content` field in assistant messages (chain-of-thought separate from
the final answer). Use `--reasoning_content` to control how it is handled:

| Value | Behaviour |
| --- | --- |
| `strip` (default) | Field is discarded before `apply_chat_template`. Safe for any tokenizer. |
| `inline` | Wrapped as `<think>…</think>` and prepended to `content`. Preserves reasoning in a tokenizer-agnostic way. |
| `native` | Passed unchanged. Requires the tokenizer's chat template to handle the field (e.g. Qwen3). |

```bash
python -m modelopt.torch.utils.plugins.megatron_preprocess_data \
    --hf_dataset nvidia/Nemotron-Post-Training-Dataset-v3 \
    --json_keys messages \
    --tokenizer Qwen/Qwen3-0.6B \
    --output_dir tokenized_qwen3 \
    --workers 32 \
    --reasoning_content inline
```

## Synthetic Test Dataset

`synthetic_conversations_1k.jsonl` is a 1,000-sample dataset in OpenAI messages format
(900 single-turn + 100 two-turn conversations) covering writing, reasoning, math, coding,
STEM, extraction, humanities, and roleplay categories.

This dataset was synthesized by Claude (Anthropic) and is licensed under Apache-2.0.
It is intended for testing and CI regression — not for production training.

```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```
