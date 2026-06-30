# PTQ Config Units

Reusable building blocks for composing PTQ quantization configurations.
Each file defines one or more `quant_cfg` entries that can be imported
into recipes or presets via `$import`.

Every reusable snippet imported via an `imports` section must declare a
`# modelopt-schema: ...` preamble. Unit snippets use either
`QuantizerCfgEntry` for one entry or `QuantizerCfgListConfig` for a list
of entries; the loader uses that schema to validate the snippet and to
choose append vs splice semantics for list imports.

Units are **not** standalone configs — they don't have `algorithm` or
`metadata`. They are meant to be composed into complete configs by
recipes (under `general/` or `models/`) or presets (under `presets/`).

| File | Description |
|------|-------------|
| `base_disable_all.yaml` | Deny-all entry: disables all quantizers as the first step |
| `default_disabled_quantizers.yaml` | Standard exclusions (LM head, routers, BatchNorm, etc.) |
| `kv_fp8.yaml` | FP8 E4M3 KV cache quantizer entry; supported on Hopper+ GPUs |
| `kv_fp8_cast.yaml` | FP8 E4M3 KV cache with constant amax (skips KV calibration); supported on Hopper+ GPUs |
| `kv_nvfp4_cast.yaml` | NVFP4 KV cache with constant amax (skips KV calibration); supported on Blackwell+ GPUs |
| `w8a8_fp8_fp8.yaml` | FP8 weight + activation quantizer entries (W8A8); supported on Hopper+ GPUs |
| `w4a4_nvfp4_nvfp4.yaml` | NVFP4 weight + activation quantizer entries (W4A4); supported on Blackwell+ GPUs |
