# Dynamic Memory Sparsification (DMS)

A minimal, optimized implementation of the DMS algorithm for KV-cache compression, as described in:

> **Inference-Time Hyper-Scaling with KV Cache Compression**  
> Adrian Łańcucki, Konrad Staniszewski, Piotr Nawrot, Edoardo M. Ponti  
> Paper: [https://arxiv.org/abs/2506.05345](https://arxiv.org/abs/2506.05345)  
> NeurIPS: [https://neurips.cc/virtual/2025/loc/san-diego/poster/119605](https://neurips.cc/virtual/2025/loc/san-diego/poster/119605)  

Inference-time scaling trades efficiency for improved reasoning by generating longer sequences. In Transformer LLMs, generation cost is often bottlenecked by the size of the key-value (KV) cache. DMS addresses this by learning a KV cache eviction policy that compresses the cache while preserving accuracy.

## How it works

DMS learns a per-head eviction policy that determines which KV cache entries to keep during generation. Rather than immediately discarding tokens, DMS delays eviction decisions, implicitly merging representations and preserving critical information. During training, the compression ratio is gradually increased from 1× to a target value (e.g., 8×), using knowledge distillation to match the outputs of an uncompressed teacher model.

## What makes DMS practical

- Achieves **8× compression** with minimal accuracy loss
- Adapter training: the default recipe trains eviction adapters only and freezes base weights for efficiency
- Requires **~250 training steps** (about **4 hours on 8× H100**) to adapt Qwen3-8B
- Drop-in replacement for Hugging Face models via a custom cache that supports variable sequence lengths across attention heads

| Model family | Size | Training time (8× H100) |
|------------|------|--------------------------|
| Qwen3      | 8B   | ~4 hours                 |

---

## Quick start: Retrofitting Qwen3-8B with DMS

### Installation

This repository is designed to run inside an NVIDIA PyTorch container:

```bash
docker pull nvcr.io/nvidia/pytorch:25.11-py3
```

Clone and install:

```bash
git clone https://github.com/NVIDIA/Model-Optimizer
cd experimental/dms
pip install -e .
```

This single install provides everything needed for training and evaluation (including lm-eval-harness).

### Train DMS adapters

**Note:** The number of GPUs determines the effective batch size. The configuration below was tested on a DGX H100 with 8× H100 80GB GPUs. For debugging with a smaller compute budget (e.g., a single RTX 5090), see [`scripts/train_small_debug.sh`](scripts/train_small_debug.sh).

```bash
bash scripts/train.sh configs/qwen3_8b.yaml
```

This freezes the original Qwen3-8B weights and trains only the DMS eviction-policy parameters using knowledge distillation. Training completes in ~4 hours on a single DGX H100 node.

The trained student model is saved to `outputs/qwen3_8b/student_model/` at the end of training.

To resume training from the latest checkpoint, set `resume_from_checkpoint: "auto"` in the YAML config.

### Extract from an intermediate checkpoint (optional)

To extract a model from an intermediate checkpoint, run:

```bash
python -m models.qwen3.extract \
    --config outputs/qwen3_8b/config.yaml \
    --checkpoint outputs/qwen3_8b/checkpoint-238
```

### Evaluate

Evaluate on the RULER long-context benchmark:

```bash
bash scripts/evaluate.sh outputs/qwen3_8b/student_model
```

**Prerequisite:** The saved model relies on the `dms` package for its attention and cache implementations. Ensure `dms` is installed (`pip install -e .`) in any environment where you load the model for inference or evaluation.

---

## Repository structure

```bash
.
├── configs                   # YAML experiment configs
│   └── qwen3_8b.yaml
├── dms                       # Core DMS library (pip install -e .)
│   ├── attention_prefill.py  # Exact prefill with eviction-based masking
│   ├── attention.py          # DMS attention: train + inference modes
│   ├── cache_paged.py        # Paged cache with block-based memory management
│   ├── cache.py              # KV cache: HF wrapper + combined + contiguous
│   ├── core.py               # Shared ops: prepare_attention_input, gating, chunked prefill
│   └── training
│       ├── data.py           # Data pipeline: loading, blending, tokenization
│       └── engine.py         # Distillation, model config, noise, trainer state
├── ARCHITECTURE.md
├── example_inference.ipynb
├── models                    # Model-specific adaptations
│   └── qwen3
│       ├── configuration_qwen3_dms.py   # Qwen3ConfigDMS
│       ├── extract.py                   # Checkpoint extraction
│       ├── modeling_qwen3_dms.py        # Qwen3ForCausalLMDMS
│       └── train.py                     # Training entry point
└── scripts                   # Launch scripts
    ├── evaluate.sh
    └── train.sh
```

For code details, advanced options, and guides on extending DMS, see [ARCHITECTURE.md](ARCHITECTURE.md).

## Limitations

This repository currently supports training eviction adapters only and keeps base model weights frozen. This training approach can achieve comparable accuracy while being roughly two orders of magnitude cheaper than full fine-tuning. In contrast, the original recipe used in the paper updates all model weights during training; we plan to support it in the near future.

For inference, this repository currently supports a single prefill-then-generate workflow. Multi-turn conversations with interleaved `prefill, generate, prefill, ...` steps are not yet optimized: the cache must be reset between independent sequences, and a slow fallback is used that simulates generation via repeated prefill steps. See [example_inference.ipynb](./example_inference.ipynb) for details.

## Citation

If you found DMS useful, please cite:

```bibtex
@inproceedings{
  lancucki2025inferencetime,
  title={Inference-Time Hyper-Scaling with {KV} Cache Compression},
  author={Adrian {\L}a{\'n}cucki and Konrad Staniszewski and Piotr Nawrot and Edoardo Ponti},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
  year={2025},
  url={https://openreview.net/forum?id=8ZiElzQxf1}
}
```
