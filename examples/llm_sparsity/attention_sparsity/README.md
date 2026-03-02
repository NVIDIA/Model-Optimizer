# Attention Sparsity for HuggingFace Models

In this tutorial, we demonstrate how to use NVIDIA Model Optimizer to apply attention sparsity to HuggingFace models. Two methods are supported:

- **Skip-Softmax**: Threshold-based skipping of near-zero attention scores during softmax (requires `attn_implementation="eager"`)
- **Sparse24 Triton**: Fine-grained 2:4 sparsity on attention scores via a fused Triton kernel with autograd support (uses `attn_implementation="modelopt_triton"`)

## Getting Started

### Quick Example

```python
import modelopt.torch.sparsity.attention_sparsity as mtsa
from modelopt.torch.sparsity.attention_sparsity.config import SKIP_SOFTMAX_DEFAULT

# Load your model
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-8B",
    attn_implementation="eager",  # Required for sparse attention
    torch_dtype=torch.bfloat16,
)

# Apply sparse attention
model = mtsa.sparsify(model, config=SKIP_SOFTMAX_DEFAULT)
```

> [!Note]
> `attn_implementation="eager"` is required for sparse attention to work properly. Flash Attention 2 or SDPA would bypass the softmax patching needed for stats collection.

## Configuration Options

Two pre-defined configurations are available:

### 1. Fixed Threshold (SKIP_SOFTMAX_DEFAULT)

Uses a fixed threshold value. Simple but may not be optimal for all sequence lengths.

```python
from modelopt.torch.sparsity.attention_sparsity.config import SKIP_SOFTMAX_DEFAULT

model = mtsa.sparsify(model, config=SKIP_SOFTMAX_DEFAULT)
```

### 2. Calibrated Threshold (SKIP_SOFTMAX_CALIB)

Uses RULER-based calibration to determine an optimal dynamic threshold that adapts to sequence length. Recommended for production use.

```python
from modelopt.torch.sparsity.attention_sparsity.config import SKIP_SOFTMAX_CALIB

model = mtsa.sparsify(model, config=SKIP_SOFTMAX_CALIB)
```

## Prerequisites

### Local Installation

For Hugging Face models, install Model Optimizer with `hf` dependencies using `pip` from [PyPI](https://pypi.org/project/nvidia-modelopt/) and install the requirements for the example:

```bash
pip install nvidia-modelopt[hf]
```

### Download RULER Calibration Data (Required for Calibration)

If using `SKIP_SOFTMAX_CALIB`, you need to download the RULER calibration dataset first:

```bash
bash ./download_ruler_data.sh
```

This downloads the Paul Graham essays dataset used for generating calibration samples.

## Run Sparse Attention on HuggingFace Models

### Basic Usage (Without Calibration)

Apply sparse attention with a fixed threshold:

```bash
python hf_sa.py \
    --pyt_ckpt_path Qwen/Qwen3-8B \
    --sparse_attn skip_softmax
```

### With RULER Calibration

Apply sparse attention with calibrated thresholds for optimal sparsity:

```bash
python hf_sa.py \
    --pyt_ckpt_path Qwen/Qwen3-8B \
    --sparse_attn skip_softmax_calib
```

The calibration process:

1. Generates RULER calibration samples
2. Collects attention statistics during forward passes
3. Determines optimal threshold scale factor for target sparsity ratio

### Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--pyt_ckpt_path` | Required | HuggingFace model path or name |
| `--sparse_attn` | `skip_softmax` | Configuration: `skip_softmax` or `skip_softmax_calib` |
| `--backend` | `pytorch` | Backend: `pytorch` (only supported backend) |
| `--seq_len` | `2048` | Maximum sequence length for input prompts |
| `--export_dir` | `None` | Directory to export the sparsified model |

## Output Comparison

The script automatically compares outputs before and after applying sparse attention:

1. Loads a test sample from the NarrativeQA dataset
2. Generates text before sparse attention is applied
3. Applies sparse attention (with optional calibration)
4. Generates text after sparse attention is applied
5. Compares and displays both outputs

## Export Model

Export the sparsified model to a HuggingFace checkpoint:

```bash
python hf_sa.py \
    --pyt_ckpt_path Qwen/Qwen3-8B \
    --sparse_attn skip_softmax_calib \
    --export_dir ./exported_sparse_model
```

The exported model can be loaded and used with standard HuggingFace APIs.

## Custom Configuration

You can create custom sparse attention configurations:

```python
custom_config = {
    "sparse_cfg": {
        "calibration": {  # Optional: omit for fixed threshold
            "target_sparse_ratio": {"prefill": 0.5, "decode": 0.5},  # Target 50% sparsity
            "samples": 128,              # Number of calibration samples
            "max_seqlen": 8192,          # Maximum sequence length
            # Optional: customize threshold trials for calibration
            "threshold_trials": [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1, 2e-1, 3e-1, 5e-1, 7e-1],
        },
        "*attn*": {  # Pattern to match attention modules
            "method": "flash_skip_softmax",
            "threshold": {"prefill": 1e-3, "decode": 1e-4},  # Phase-specific thresholds (ignored if calibration is used)
            "br": 128,          # Flash Attention block rows
            "bc": 128,          # Flash Attention block columns
            "backend": "pytorch",
            "collect_stats": True,
            "enable": True,
        },
        "default": {"enable": False},
    },
}

model = mtsa.sparsify(model, config=custom_config)
```

## Fine-grained 2:4 Sparse Attention

In addition to skip-softmax, Model Optimizer supports **fine-grained 2:4 sparsity** on attention scores via a fused Triton kernel. For every 4 attention scores along the key dimension, the kernel keeps only the top 2 and zeros out the rest — achieving 50% fixed sparsity with no calibration needed.

### Quick Example

```python
import modelopt.torch.sparsity.attention_sparsity as mtsa
from modelopt.torch.sparsity.attention_sparsity.config import SPARSE24_TRITON

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B",
    torch_dtype=torch.bfloat16,
)

model = mtsa.sparsify(model, config=SPARSE24_TRITON)
```

> [!Note]
> Unlike skip-softmax, sparse24 does **not** require `attn_implementation="eager"`. The `mtsa.sparsify` call automatically registers the Triton kernel as `attn_implementation="modelopt_triton"`.

### Running via Command Line

```bash
python hf_sa.py \
    --pyt_ckpt_path meta-llama/Llama-3.1-8B \
    --sparse_attn sparse24_triton \
    --backend triton
```

### Key Differences from Skip-Softmax

| | Skip-Softmax | Sparse24 Triton |
|---|---|---|
| Method | Threshold-based softmax skipping | 2:4 structured sparsity on attention scores |
| Attention backend | `eager` (patches `F.softmax`) | `modelopt_triton` (fused Triton kernel) |
| Calibration | Optional (RULER-based) | Not needed (fixed top-2-of-4 selection) |
| Sparsity ratio | Variable (depends on threshold) | Fixed 50% |
| Diagonal preservation | N/A | Yes (tiles near the causal diagonal are kept dense) |
| Training support | No | Yes (autograd-compatible forward/backward) |
| Decode support | Yes | Yes (same kernel, `is_causal=False`) |

### Training with Sparse24 Attention

The Triton kernel supports autograd. When `requires_grad=True`, the HF integration automatically uses the backward-capable path:

```python
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B", torch_dtype=torch.bfloat16)
model = mtsa.sparsify(model, config=SPARSE24_TRITON)
model.train()

# Gradients flow through the sparse attention
output = model(input_ids=ids, labels=labels)
output.loss.backward()  # dQ, dK, dV computed via Triton backward kernels
```

### Custom Sparse24 Configuration

```python
custom_config = {
    "sparse_cfg": {
        "*attn*": {
            "method": "sparse24_triton",
            "backend": "triton",
            "skip_diagonal_blocks": True,  # Keep diagonal tiles dense (recommended)
            "enable": True,
        },
        "default": {"enable": False},
    },
}

model = mtsa.sparsify(model, config=custom_config)
```

Set `skip_diagonal_blocks: False` to apply 2:4 sparsity to all tiles including the diagonal (more aggressive but may hurt quality for local attention patterns).

## References

- [Model Optimizer Documentation](https://nvidia.github.io/Model-Optimizer/)
- [RULER: What's the Real Context Size of Your Long-Context Language Models?](https://github.com/NVIDIA/RULER)
