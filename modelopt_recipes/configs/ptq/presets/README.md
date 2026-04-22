# PTQ Preset Configs

This directory holds preset quantization configurations that serve as the
YAML source of truth for the hardcoded `*_CFG` dicts in
`modelopt.torch.quantization.config` (e.g., `FP8_DEFAULT_CFG`,
`FP8_KV_CFG`).

Presets compose from the reusable snippets in `configs/numerics/` and
`configs/ptq/units/` via the `$import` system, and are split into two
kinds:

- **`model/`** — *full* quantization presets. Each file is a complete,
  self-contained config (it sets `algorithm` and a full `quant_cfg` with
  a base-disable-all prefix + standard exclusions) and can be passed
  directly to `mtq.quantize()`. Example: `model/fp8.yaml`
  (the YAML source of `FP8_DEFAULT_CFG`).
- **`kv/`** — *partial* KV-cache quantization fragments. Each file
  contains only the KV-specific `quant_cfg` entries (no `algorithm`, no
  base-disable-all). They are **not** standalone — they are designed to
  be merged on top of a `model/` preset via `$import` to produce a
  complete config. Example: `kv/fp8.yaml` (the YAML source of
  `FP8_KV_CFG`).

**Note:** The main purpose of these presets is to support the existing
`hf_ptq.py` script's `--qformat` / `--kv_cache_qformat` flags and other
code paths that reference
the hardcoded `*_CFG` dicts, maintaining backward compatibility during
the transition to recipe-based workflows. Users are encouraged to use
`load_recipe` with full recipe files under `general/` or `models/`
instead. Some or all of these presets may be deprecated or removed in
future releases as the recipe-based workflow becomes the standard entry
point.
