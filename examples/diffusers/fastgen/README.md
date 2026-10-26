# FastGen diffusion examples

This directory contains training and inference examples for diffusion distillation methods in
`modelopt.torch.fastgen`.

- [DMD2 for Qwen-Image](dmd2/README.md)
- [PDD for Qwen-Image](pdd/README.md)

The `fastgen_data/` and `preprocess/` packages are shared utilities. Algorithm-specific entrypoints,
configs, checkpoint helpers, and documentation live in their corresponding subdirectory.

## References

- Upstream FastGen: [NVlabs/FastGen](https://github.com/NVlabs/FastGen)
- Training framework: [NeMo AutoModel](https://github.com/NVIDIA-NeMo/Automodel)
- ModelOpt implementation: [`modelopt/torch/fastgen/`](../../../modelopt/torch/fastgen/)
