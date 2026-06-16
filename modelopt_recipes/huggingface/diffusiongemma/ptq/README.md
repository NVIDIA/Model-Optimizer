# DiffusionGemma PTQ recipes

DiffusionGemma is a block-diffusion encoder-decoder text LLM with a Gemma4 MoE
backbone shared between an encoder pass and a 48-step iterative decoder.
Quantization targets the MoE experts; the self-conditioning network is
text-only and is not exercised by standard PTQ calibration data, so its
``TensorQuantizer`` observers never see input and ``_export_quantized_weight``
crashes on the missing ``_amax``. These recipes apply the model-specific
``*self_conditioning*`` exclude on top of the standard default exclusions.

| File | What's model-specific |
|------|-----------------------|
| `disabled_quantizers.yaml` | Reusable unit (`QuantizerCfgListConfig`). Merges the standard `default_disabled_quantizers` exclusions with the DiffusionGemma-specific `*self_conditioning*` exclude. Imported by the recipe below as the single `disabled_quantizers` slot so it doesn't pull in two disabled-quantizer sets. |
| `nvfp4_experts_only.yaml` | Dynamic W4A4 NVFP4 quantization on MoE experts only (attention and dense MLP stay bf16). Identical numerics to the general `nvfp4_experts_only` preset; what makes it model-specific is that it imports `disabled_quantizers.yaml` from this folder to skip the self-conditioning network. |
