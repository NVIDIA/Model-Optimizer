# MiniMax-M3 mixed MXFP8 and NVFP4 quantization

This example produces a MiniMax-M3 checkpoint with an MXFP8 base and NVFP4
routed experts. It copies non-routed-expert tensors from the vendor MXFP8
checkpoint and quantizes routed-expert weights from the BF16 checkpoint one MoE
layer at a time, without loading the complete model.

The vision branch, routers, shared experts, `lm_head`, and KV cache retain their
vendor checkpoint formats. Routed-expert activation `input_scale` is fixed to
1.0.

## Setup

Install ModelOpt with its Hugging Face dependencies:

```bash
pip install -e ".[hf]"
```

The script requires local vendor MXFP8 and BF16 checkpoints. It quantizes one
BF16 MoE layer at a time and copies the vendor MXFP8 base one shard at a time,
so it never loads either complete model.

## Usage

```bash
python examples/minimax_m3/hf_ptq_mixed_mxfp8_nvfp4.py \
    --mxfp8_ckpt /models/minimax-m3-mxfp8 \
    --bf16_ckpt /models/minimax-m3-bf16 \
    --recipe huggingface/minimax_m3_vl/ptq/nvfp4_experts_only \
    --output_ckpt /models/minimax-m3-mxfp8-nvfp4 \
    --device cuda
```

The script writes a Hugging Face checkpoint with standard safetensor shard
names and mixed-precision metadata in `config.json` and
`hf_quant_config.json`. This workflow was tested with PyTorch 26.05.
