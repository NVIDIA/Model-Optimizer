# PTQ Preset Configs

This directory holds preset quantization configurations that serve as the
YAML source of truth for the hardcoded `*_CFG` dicts in
`modelopt.torch.quantization.config` (e.g., `FP8_DEFAULT_CFG`).

Each preset is a self-contained config with `quant_cfg` that can be
passed to `mtq.quantize()`. Presets compose from the reusable snippets
in `configs/numerics/` and `configs/ptq/units/` via the `$import` system.

**Note:** The main purpose of these presets is to support the existing
`hf_ptq.py` script's `--qformat` / `--kv_cache_qformat` flags and other
code paths that reference
the hardcoded `*_CFG` dicts, maintaining backward compatibility during
the transition to recipe-based workflows. Users are encouraged to use
`load_recipe` with full recipe files under `general/` or `models/`
instead. Some or all of these presets may be deprecated or removed in
future releases as the recipe-based workflow becomes the standard entry
point.
