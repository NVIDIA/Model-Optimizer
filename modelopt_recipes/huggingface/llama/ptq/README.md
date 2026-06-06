# Llama PTQ recipes

Recipes for Hugging Face models with `model_type: llama` (Llama 3.x, Llama 3.1,
Llama 3.2, Llama 3.3, etc.).

## Choosing a recipe

| Recipe | When to use |
|--------|-------------|
| `nvfp4_mlp_only-kv_fp8_cast.yaml` | **Recommended starting point** for NVFP4 on Llama when full W4A4 hurts accuracy. Quantizes MLP (and MoE expert) layers only; attention stays unquantized. FP8 KV cache uses cast mode (no extra KV calibration). |
| `nvfp4_default-kv_fp8_cast.yaml` | Full dynamic NVFP4 W4A4 on all linear layers when you need maximum compression and can accept more accuracy loss. |

General-model equivalents live under `modelopt_recipes/general/ptq/` with the
same numerics. These `huggingface/llama/ptq/` paths exist so users can pass a
model-family recipe without guessing:

```bash
python examples/llm_ptq/hf_ptq.py \
  --pyt_ckpt_path meta-llama/Llama-3.1-8B-Instruct \
  --recipe huggingface/llama/ptq/nvfp4_mlp_only-kv_fp8_cast \
  --export_path ./llama-3.1-8b-nvfp4-mlp-only
```

For data-driven KV calibration, use the general recipes with `kv_fp8` instead of
`kv_fp8_cast` (e.g. `general/ptq/nvfp4_mlp_only-kv_fp8`).

NVFP4 inference requires NVIDIA Blackwell GPUs and a compatible runtime (TensorRT-LLM,
vLLM, or SGLang). Recipe validation and unit tests run on CPU; end-to-end PTQ
export still requires a CUDA GPU.
