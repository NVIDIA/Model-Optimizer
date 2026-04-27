# Dataset Blend Configuration

Dataset blends are defined in YAML files that specify which datasets to mix,
how to sample from them, and how to tokenize them.

See [`blend_example.yaml`](blend_example.yaml) for a runnable example with all options.

## Blend YAML Structure

Blend YAML files define the dataset size, split ratios, and sources:

```yaml
blend_size: 100000        # total samples to download across all sources
splits:                   # train/eval/test split ratios (must sum to 1.0)
  train: 0.80
  eval: 0.10
  test: 0.10

sources:
  - hf_path: nvidia/Nemotron-SWE-v1
    split: r2e_gym
    ratio: 6000
    category: code
```

Processing parameters (`cache_dir`, `shuffle`, `num_proc`, etc.) are set via
`DataArguments` in the training config YAML or CLI flags. `train_samples` and
`eval_samples` in training configs are runtime caps on the pre-split dataset —
changing them does not invalidate the cache.

## Top-Level Fields

| Field | Required | Default | Description |
|-------|----------|---------|-------------|
| `blend_size` | no | `100000` | Total samples to download across all sources |
| `splits` | no | `{train: 0.8, eval: 0.1, test: 0.1}` | Relative split weights; e.g. `{train: 8, eval: 1, test: 1}` gives 80/10/10 |

## Per-Source Fields

| Field | Required | Default | Description |
|-------|----------|---------|-------------|
| `hf_path` | yes | - | HuggingFace dataset path or local path |
| `ratio` | yes | - | Relative weight (normalized across all sources) |
| `split` | yes | - | Split(s) to load (auto train/eval). See below |
| `dataset_kwargs` | no | `{}` | Extra kwargs passed to `datasets.load_dataset()` (e.g. `{name: "3.0.0"}`) |
| `apply_chat_template` | no | `true` | If true, expects OpenAI messages format |
| `chat_key` | no | `"messages"` | Key containing conversations |
| `category` | no | `""` | Label for logging |

## Split Modes

Specifies which HuggingFace split(s) to load from each source. Samples are pooled across all sources, then globally split into train/eval/test by the top-level `splits` ratios.

```yaml
# Single split
split: train

# Comma-separated (equal weight per split)
split: code,math,stem

# Dict (weighted per split: 3:2:1 ratio)
split:
  code: 3
  math: 2
  stem: 1
```

## Dataset Kwargs

Pass any extra keyword arguments to `datasets.load_dataset()` via `dataset_kwargs`:

```yaml
# HF config name (e.g. cnn_dailymail)
dataset_kwargs: {name: "3.0.0"}

# Multiple kwargs
dataset_kwargs:
  name: "3.0.0"
  trust_remote_code: true
  revision: main
```

## Streaming and Shuffle

All HuggingFace datasets are loaded with `streaming=True` to avoid downloading
entire datasets.

- `shuffle: true` - Reservoir sampling: `dataset.shuffle(buffer_size=N).take(n)`.
  Accurate but slower with large buffers.
- `shuffle: false` - Take first N samples: `dataset.take(n)`. Fast and deterministic.

## Pre-tokenize and Cache

Pre-tokenize the dataset before training to avoid repeated work:

```sh
python dataset_utils.py \
    --dataset_config configs/dataset/blend.yaml \
    --model_name_or_path Qwen/Qwen3-8B
```

The cached dataset is saved to `dataset_cache_dir` (default: `.dataset_cache/tokenized/`).
Subsequent runs with the same dataset config and tokenizer reuse the cache.
The cache key depends on `blend_size`, `splits`, sources, and the tokenizer —
changing `train_samples` or `eval_samples` does **not** invalidate the cache.

## Adding New Datasets

Add a source entry to your blend YAML:

```yaml
sources:
  # Chat dataset (OpenAI messages format)
  - hf_path: your/dataset
    split: train
    ratio: 1000

  # Dataset with different chat key
  - hf_path: your/sharegpt-dataset
    split: train
    ratio: 500
    chat_key: conversations

  # Plain text dataset (pretraining-style)
  - hf_path: your/text-corpus
    split: train
    ratio: 500
    apply_chat_template: false
```
