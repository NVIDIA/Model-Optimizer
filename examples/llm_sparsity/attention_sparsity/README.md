# Attention Sparsity for HuggingFace Models

In this tutorial, we demonstrate how to use NVIDIA TensorRT Model Optimizer to apply attention sparsity to HuggingFace models. Attention sparsity reduces computational cost by skipping near-zero attention scores during the softmax computation.

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
bash modelopt/torch/sparsity/attention_sparsity/calibration/download_ruler_data.sh
```

This downloads the Paul Graham essays dataset used for generating calibration samples.

## Run Sparse Attention on HuggingFace Models

### Basic Usage (Without Calibration)

Apply sparse attention with a fixed threshold:

```bash
python examples/llm_sparsity/attention_sparsity/hf_sa.py \
    --pyt_ckpt_path Qwen/Qwen3-8B \
    --sparse_attn skip_softmax
```

### With RULER Calibration

Apply sparse attention with calibrated thresholds for optimal sparsity:

```bash
python examples/llm_sparsity/attention_sparsity/hf_sa.py \
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
python examples/llm_sparsity/attention_sparsity/hf_sa.py \
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

## References

- [TensorRT Model Optimizer Documentation](https://nvidia.github.io/TensorRT-Model-Optimizer/)
- [RULER: What's the Real Context Size of Your Long-Context Language Models?](https://github.com/NVIDIA/RULER)
