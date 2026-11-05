# ROCm Model Optimizer

AMD ROCm port of [NVIDIA/Model-Optimizer](https://github.com/NVIDIA/Model-Optimizer).

Targets AMD MI300X / MI325X (gfx942) with ROCm 7.x and PyTorch-ROCm.

## Quick start

```bash
# Install with ROCm PyTorch
pip install torch --index-url https://download.pytorch.org/whl/rocm6.2
pip install -e ".[hf]"

# Verify ROCm detection
python -c "from modelopt._rocm_compat import is_rocm, get_gpu_arch; print(is_rocm(), get_gpu_arch())"
```

## What's supported on ROCm

- ✅ PyTorch quantization (INT8, FP8 PTQ/QAT)
- ✅ Neural Architecture Search (NAS)  
- ✅ Structured pruning (Minitron / magnitude)
- ✅ Distillation
- ✅ PEFT / LoRA
- ✅ Speculative decoding (Medusa, EAGLE)
- ✅ ONNX export and graph surgery
- ✅ Triton FP8 / GPTQ kernels (via triton-rocm)

## Not yet supported on ROCm

- ❌ TensorRT deployment backend (use MIGraphX instead — Phase 2)
- ❌ NVFP4 / FP4 quantization (Hopper-only format)
- ❌ cuDNN INT8 conv kernel (use PyTorch conv + hipBLASLt)

## Gap analysis

See [`workspace/gap_analysis.md`](workspace/gap_analysis.md) for full details.

## License

Apache 2.0 — same as upstream NVIDIA/Model-Optimizer.
