# PTQ Preset Configs

This directory holds preset quantization configurations that serve as the
single source of truth for the hardcoded `*_CFG` dicts in
`modelopt.torch.quantization.config` (e.g., `FP8_DEFAULT_CFG`).

Each preset is a complete, self-contained config with `algorithm` and
`quant_cfg` — ready to pass directly to `mtq.quantize()`. Presets compose
from the reusable snippets in `configs/numerics/` and `configs/ptq/` via
the `$import` system.

When adding a new preset, use existing snippets where possible and keep
the YAML as the authoritative definition — the Python config should load
from here rather than hardcoding the dict.
