# ModelOpt Recipes

This folder is the library of **ModelOpt optimization recipes** — declarative
YAML files that describe a complete model-optimization workflow (post-training
quantization, speculative-decoding training, diffusion distillation).

**Purpose:** a recipe is the single, version-controlled source of truth for *how*
a model is optimized — algorithm, per-layer numeric formats, and calibration —
expressed as data instead of code. That makes an optimization run reproducible,
diffable, and shareable without hand-writing Python config, and lets a tuned
configuration be looked up by name. The same YAML drives the Python API
(`load_recipe`), the example CLIs (`--recipe`), and — for the presets under
`configs/` — the built-in `*_CFG` constants.

Recipes are composed from small, reusable building blocks via an `$import`
system, then loaded by path relative to this folder, e.g.:

```python
# PTQ recipe -> mtq.quantize()
from modelopt.recipe import load_recipe
cfg = load_recipe("general/ptq/nvfp4_default-kv_fp8_cast")

# distillation recipe -> DMDConfig
from modelopt.torch.fastgen import load_dmd_config
cfg = load_dmd_config("general/distillation/dmd2_qwen_image")
```

or selected from a script/CLI flag, e.g. `hf_ptq.py --recipe
huggingface/qwen3_5/ptq/w4a16_nvfp4-fp8_attn-kv_fp8_cast`.

> 📖 **Must-read for PTQ recipe tuning → [`ptq.md`](ptq.md).** It is the
> guide to every PTQ scheme — body scopes (NVFP4/FP8, experts-only / mlp-only /
> weight-only), KV-cache modes, and calibration variants — with concrete guidance
> on **choosing and tuning a recipe** for your model and deployment. Start there
> before picking a recipe.
>
> This README is the **catalog** across all recipe families; `ptq.md` is the
> how-to for PTQ.

## Layout

| Directory | What lives here |
|-----------|-----------------|
| `general/` | **Model-agnostic** recipes — a good starting point for any model. PTQ combos, speculative-decoding training, and distillation. |
| `huggingface/<model_type>/` | **Model-specific** recipes keyed by a HF `model_type`, optionally nested by released checkpoint. Use these first if your model has an entry. |
| `models/<model_name>/` | **Instance-specific** recipes that mirror a particular published checkpoint's quantization config. |
| `configs/` | Shared building blocks (`numerics/`, `ptq/units/`, `ptq/presets/`) that recipes compose from via `$import`. Not run directly. |

**Choosing where to look:** check `huggingface/<model_type>/` (then any nested
`<checkpoint>/`) for your model first; if there's no entry, fall back to
`general/`. The presence of a model folder signals a recommended, tuned recipe.

---

## General recipes

The model-agnostic recipes live under `general/`. For **PTQ**, recipes are
mix-and-match combinations of formats, scope, KV-cache mode, and calibration —
**[`ptq.md`](ptq.md) is the guide; read it to understand the schemes and choose
one.**

Other general recipe families are documented inside their own folders:
`general/speculative_decoding/` (EAGLE3 / DFlash draft-head training) and
`general/distillation/` (diffusion distillation, e.g. DMD2).

---

## `huggingface/` — model-specific recipes

Each lives under its HF `model_type`. The point of a model folder is to capture
**what differs from the generic preset** — usually an algorithm tweak or a
disabled-quantizer pattern for non-text branches. The numerics and standard
exclusions are still inherited from `configs/`. Browse
[`huggingface/`](huggingface/) for the available `model_type`s; each `<task>/`
folder has a `README.md` describing the exact delta. See [`ptq.md`](ptq.md) for
how the model-specific recipes compare to the general ones and why they deviate.

## `models/` — checkpoint-specific recipes

These mirror a single **published checkpoint's** quantization config exactly —
a per-component mixed-precision scheme tuned to match a specific release. Browse
[`models/`](models/) for the available checkpoints.

---

## Adding a recipe

- **New combo for any model** → add to `general/ptq/` by composing existing
  `configs/` units; follow the `<formats-scope>-<kv-mode>[-<algorithm>]` naming.
- **Tuned for a HF architecture** → `huggingface/<model_type>/<task>/`, with a
  `README.md` documenting the delta from the generic preset. Verify the exact
  `model_type` against the checkpoint's `config.json` before placing it.
- **Mirrors a specific released checkpoint** → `models/<model_name>/`.
- Share reused bodies via a `# modelopt-schema:`-tagged snippet and `$import`
  it; keep recipe wrappers thin.
