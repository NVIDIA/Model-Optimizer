# Kimi-K3 PTQ recipes

vLLM's Kimi-K3 model code (`vllm/models/kimi_k3/nvidia/`) builds some layers with
`quant_config=None`, whatever the checkpoint holds, so a checkpoint that stores one of them
quantized does not load. The general `nvfp4_experts_only-kv_fp8_cast` recipe quantizes two of
them, the latent MoE projections, through its `*block_sparse_moe*` patterns. These recipes keep
every such layer in BF16:

| Kimi-K3 layers (count) | The model card's vLLM image (2026-07-27 build) | vLLM main |
|------------------------|------------------------------------------------|-----------|
| `block_sparse_moe.routed_expert_down_proj`, `routed_expert_up_proj` (92 each) | `quant_config=None` | quantizable since vllm-project/vllm#57430 |
| `self_attention_res_proj`, `mlp_res_proj` (93 each), `output_attn_res_proj` (1) | `quant_config=None` | `quant_config=None` |
| KDA `self_attn.{q,k,v}_conv1d` (69 each), loaded into one `conv1d` | built without the quantization config | built without the quantization config |

Keeping the latent projections in BF16 loads on both. nvidia/Kimi-K3-NVFP4 keeps all of these in
BF16 too.

| File | What's model-specific |
|------|-----------------------|
| `disabled_quantizers.yaml` | Reusable unit (`QuantizerCfgListConfig`). The standard `default_disabled_quantizers` exclusions plus the six patterns above. Imported by the recipe below in place of `default_disabled_quantizers`. |
| `nvfp4_experts_only-kv_fp8_cast.yaml` | The general `nvfp4_experts_only-kv_fp8_cast` recipe with `disabled_quantizers.yaml` from this folder. Everything else is the general recipe's: dynamic NVFP4 W4A4 on the routed experts, FP8 KV-cache cast with constant amax, max calibration, no layerwise calibration. Like the general recipe, it also quantizes the shared experts, which sit under `block_sparse_moe`: they load in vLLM, though nvidia/Kimi-K3-NVFP4 keeps them in BF16. |

Run it with `--recipe model_type/kimi_k3/ptq/nvfp4_experts_only-kv_fp8_cast`.
