# Video Sparse Attention (VSA) Example

This example demonstrates how to apply Video Sparse Attention (VSA) optimization to video diffusion models for faster inference.

## Overview

VSA is a two-branch sparse attention architecture designed specifically for video diffusion models:

1. **Compression Branch**: Averages tokens within 3D video blocks (default 4x4x4 = 64 tokens) and computes coarse-grained attention for global context.

2. **Sparse Branch**: Selects the top-K most important blocks based on attention scores and computes fine-grained attention only for those blocks.

The branches are combined using learned gating: `output = compression * gate_compress + sparse`

## Requirements

```bash
pip install torch>=2.0
pip install modelopt
# Optional: pip install diffusers  # For real video diffusion models
```

## Quick Start

### Using LTX-2 Trainer (Recommended)

```bash
# Full video generation with VSA vs baseline comparison
python test_ltx2_vsa_integration.py \
    --checkpoint path/to/model.safetensors \
    --text-encoder-path path/to/gemma \
    --compare

# Generate video with custom sparsity
python test_ltx2_vsa_integration.py \
    --checkpoint path/to/model.safetensors \
    --text-encoder-path path/to/gemma \
    --top-k-ratio 0.3 --output my_video.mp4
```

## Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--checkpoint` | (required) | Path to model checkpoint (.safetensors) |
| `--text-encoder-path` | (required) | Path to Gemma text encoder directory |
| `--prompt` | A serene mountain... | Text prompt for generation |
| `--top-k-ratio` | 0.5 | Ratio of blocks to keep (0.0-1.0). Lower = more sparse |
| `--num-frames` | 121 | Number of video frames (must be k*8 + 1) |
| `--height` | 512 | Video height (must be divisible by 32) |
| `--width` | 768 | Video width (must be divisible by 32) |
| `--num-inference-steps` | 30 | Number of denoising steps |
| `--guidance-scale` | 4.0 | Classifier-free guidance scale |
| `--seed` | 42 | Random seed for reproducibility |
| `--device` | cuda | Device (cuda/cpu) |
| `--compare` | off | Run both baseline and VSA for comparison |
| `--no-vsa` | off | Disable VSA (baseline only) |

## API Usage

```python
import modelopt.torch.sparsity.attention_sparsity as mtsa
from modelopt.torch.sparsity.attention_sparsity.config import VSA_DEFAULT

# Load your video diffusion model
model = load_video_diffusion_model()

# Apply VSA with default settings
model = mtsa.sparsify(model, config=VSA_DEFAULT)

# Or with custom configuration
custom_config = {
    "sparse_cfg": {
        "*attn*": {
            "method": "vsa",
            "block_size_3d": (4, 4, 4),
            "top_k_ratio": 0.3,  # 70% sparsity
            "video_shape": (16, 28, 48),
            "enable": True,
        },
        "default": {"enable": False},
    },
}
model = mtsa.sparsify(model, config=custom_config)

# Run inference
output = model(video_latents)
```

## Model Requirements

For optimal VSA performance, video diffusion models should expose a `gate_compress` parameter in their attention layers. This is a learned parameter that controls the balance between the compression and sparse branches.

Example attention layer interface:

```python
class VideoAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.to_q = nn.Linear(hidden_dim, hidden_dim)
        self.to_k = nn.Linear(hidden_dim, hidden_dim)
        self.to_v = nn.Linear(hidden_dim, hidden_dim)
        # VSA-specific: learned gating
        self.to_gate_compress = nn.Linear(hidden_dim, hidden_dim)
```

If `gate_compress` is not available, VSA will use equal weighting (sum of both branches).

## Expected Performance

| Top-K Ratio | Sparsity | Typical Speedup |
|-------------|----------|-----------------|
| 0.5 | 50% | 1.5-2x |
| 0.3 | 70% | 2-3x |
| 0.2 | 80% | 3-4x |

*Actual speedup depends on model architecture, video resolution, and hardware.*

## Troubleshooting

### "video_shape must be set" error

Make sure to provide `video_shape` in the configuration matching your video dimensions after patchification.

### Low speedup

- VSA is most effective for long sequences (high video resolution or many frames)
- For short sequences, the overhead of block operations may reduce gains
- Ensure you're using GPU with CUDA

### Quality degradation

- Increase `top_k_ratio` to keep more blocks
- Ensure your model has `gate_compress` for optimal branch balancing

## LTX-2 Integration

LTX-2 is a state-of-the-art video diffusion model that is well-suited for VSA optimization due to its high token count.

### LTX-2 Architecture Summary

| Component | Description |
|-----------|-------------|
| **Transformer** | 48 layers, 32 heads x 128 dim = 4096 hidden |
| **Compression** | 1:8192 pixels-to-tokens (aggressive) |
| **Attention Types** | Self-attn (attn1), Cross-attn (attn2), Audio attn, Cross-modal |

### Example Scripts

| Script | Purpose |
|--------|---------|
| `test_ltx2_vsa_integration.py` | Test VSA with LTX-2 trainer pipeline |

### VSA Targets for LTX-2

VSA is applied only to **self-attention (attn1)** modules:

```python
vsa_config = {
    "sparse_cfg": {
        "*.attn1": {          # [OK] Self-attention - VSA enabled
            "method": "vsa",
            "top_k_ratio": 0.5,
            "block_size_3d": [4, 4, 4],
        },
        "*.attn2": {"enable": False},           # [NO] Text cross-attention
        "*.audio_attn*": {"enable": False},     # [NO] Audio attention
        "*.audio_to_video*": {"enable": False}, # [NO] Cross-modal
        "*.video_to_audio*": {"enable": False}, # [NO] Cross-modal
    },
}
```

### Expected Token Counts for LTX-2

| Resolution | Frames | Tokens | VSA Tiles | Recommendation |
|------------|--------|--------|-----------|----------------|
| 512x768 | 121 | ~5,808 | 91 | Excellent for VSA |
| 384x384 | 49 | ~907 | 14 | Marginal |
| 256x256 | 25 | ~200 | 3 | Too small |

For best VSA performance, use **121+ frames @ 512x768+** resolution.
