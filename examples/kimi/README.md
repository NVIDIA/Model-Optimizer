# Kimi post-training quantization

## Kimi-K3: NVFP4 routed experts and block-FP8 attention

`moonshotai/Kimi-K3` is released with its routed experts already packed as
MXFP4. Loading the 2.8T-parameter model and running the normal in-memory
`examples/hf_ptq/hf_ptq.py` flow would both discard that source representation
and require impractical host memory. The Kimi-K3 converter therefore streams
one safetensors shard at a time:

- routed-expert `w1`, `w2`, and `w3` weights are cast from MXFP4 to NVFP4;
- expert activation `input_scale` is fixed to `1.0`;
- KDA and MLA projection weights are quantized to FP8 in 128x128 blocks;
- attention activations are dynamically quantized by the inference runtime;
- shared experts, latent experts, routers, convolution weights, norms, the
  vision tower, and `lm_head` remain BF16.

The exact quantization map is recorded in
`modelopt_recipes/huggingface/models/moonshotai/Kimi-K3/ptq/nvfp4_experts-fp8_pb_attention.yaml`.
Because the source checkpoint uses packed MXFP4 tensors rather than ordinary
Hugging Face `Linear.weight` tensors, run the streaming converter instead of
passing this recipe to `hf_ptq.py`:

```bash
python examples/kimi/kimi_k3/quantize_to_nvfp4.py \
    --source_ckpt /models/moonshotai/Kimi-K3 \
    --output_ckpt /models/Kimi-K3-NVFP4 \
    --recipe huggingface/models/moonshotai/Kimi-K3/ptq/nvfp4_experts-fp8_pb_attention \
    --jobs 8
```

The conversion is calibration-free: it does not run a forward pass, require a
dataset, or require a GPU. For multi-node conversion, launch one process per
node against shared storage and set `--rank`, `--world_size`, and a common
`--run_id`. Use `--help` for the full set of conversion and synchronization
options.
