# Megatron-Bridge Tutorials

End-to-end tutorials that combine ModelOpt optimization techniques on [NVIDIA Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge) models.
Each one walks through a complete workflow using the scripts in [examples/megatron_bridge](../README.md) (`prune_minitron.py`, `distill.py`, `quantize.py`, `export.py`).

## Available tutorials

| Tutorial | What it covers |
| --- | --- |
| [NVIDIA-Nemotron-3-Nano-30B-A3B-BF16](NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/README.md) | End-to-end optimization of the Nemotron-3-Nano-30B-A3B-BF16 (MoE + Mamba-Transformer hybrid) model: Minitron structured pruning (31.6B/A3.6B → 22B/A3.0B) → two-phase knowledge distillation (100B tokens, 8K then 32K seq length) → quantization → vLLM deployment. Includes data-blend preparation, evaluation setup, and detailed pruning / data-blend / long-context ablations. |
