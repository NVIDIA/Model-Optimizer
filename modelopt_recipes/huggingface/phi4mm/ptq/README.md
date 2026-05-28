# Phi-4-Multimodal PTQ recipes

Phi-4-Multimodal is a multimodal model. Quantization should be applied only to
the language model; the speech, audio, image, and vision branches are kept in
full precision to avoid accuracy regressions on those modalities.

| File | What's model-specific |
|------|-----------------------|
| `disabled_quantizers.yaml` | Reusable unit (`QuantizerCfgListConfig`). Merges the standard `default_disabled_quantizers` exclusions with Phi-4-MM ones (`*speech*`, `*audio*`, `*image*`, `*vision*`). Imported by recipes below as the single `disabled_quantizers` slot so they don't pull in two disabled-quantizer sets. |
| `nvfp4-kv_fp8_cast.yaml` | NVFP4 W4A4 model quantization + FP8 KV-cache cast (constant amax, no KV calibration). Identical numerics to the general `nvfp4` preset / `kv_fp8_cast` unit; what makes it model-specific is that it imports `disabled_quantizers.yaml` from this folder to skip the non-language branches. |

Additional `<qformat>-kv_fp8_cast.yaml` recipes can be generated for other formats
if needed; only `nvfp4-kv_fp8_cast.yaml` is shipped by default.
