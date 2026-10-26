# Dataset Preparation

<div align="center">

| **Section** | **Description** | **Link** |
| :------------: | :------------: | :------------: |
| Building Chat Datasets | Scripts to build conversation datasets from Nemotron and other HuggingFace sources | \[[Link](#building-chat-datasets)\] |
| Tokenizing for Megatron Frameworks | Convert JSONL or HF datasets to Megatron binary format for distillation and pre-training | \[[Link](MEGATRON_DATA_PREP.md)\] |

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

See **[MEGATRON_DATA_PREP.md](MEGATRON_DATA_PREP.md)** for full documentation: general usage with JSONL and Hugging Face Hub datasets, handling of Nemotron Post-Training v3 `reasoning_content` fields, and ready-to-run tokenization commands for all Nemotron Pre/Post-Training datasets.

## Synthetic Test Dataset

`synthetic_conversations_1k.jsonl` is a 1,000-sample dataset in OpenAI messages format
(900 single-turn + 100 two-turn conversations) covering writing, reasoning, math, coding,
STEM, extraction, humanities, and roleplay categories.

This dataset was synthesized by Claude (Anthropic) and is licensed under Apache-2.0.
It is intended for testing and CI regression — not for production training.

```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```
