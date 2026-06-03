# Design proposal: `nvfp4_act_max` calibration algorithm

**Status:** Reviewed — all decisions locked; ready to implement.
**Revision:** rev 5 — name **decided: `nvfp4_act_max`**. All prior decisions
locked: v1 strategy `b_min_anchored`; `rho` config param (default 16384, `0 < rho
< 28672`); defaults `b_min_percentile=1.0`, `b_max_percentile=99.99`,
`margin=1.0`; KV-cache out of scope (recipe uses the `kv_fp8_cast` unit);
write-back is a plain `_amax` overwrite (NVFP4 input quantizer is `_dynamic =
False` and already stores its global scale in `_amax`; old §5 Option-A/B decision
withdrawn). See §9.
**Author:** (drafted with Claude Code)
**Scope:** Implement the activation g_amax calibration **Recipe** from
[`README.md` §4](./README.md) as a first-class ModelOpt calibration algorithm
named `nvfp4_act_max`, and ship a recipe config that uses it.

---

## 1. Goal

Add a calibration algorithm `nvfp4_act_max` that:

- calibrates **weights** with the existing **max** path (unchanged), and
- calibrates the **NVFP4 input (activation) quantizer's per-tensor global amax
  (`g_amax`)** using the study's Recipe (§4 of the README) instead of the naive
  per-tensor running max.

The naive default (plain `max`) sets `g_amax = B_max` (the largest block amax
seen in calibration). The study shows this is the **zero-headroom, saturation-prone**
choice: any inference block larger than the calibrated `B_max` saturates
(catastrophic, hard clip), and saturation is exactly the failure that grows
fastest with unseen outliers (40×+ MSE degradation in the study's robustness
experiment). The Recipe spends the format's wide (~4.46-decade) safe window as
**upward headroom** so unseen activation outliers degrade gracefully instead of
clipping.

---

## 2. Background: how the activation global scale actually flows

This is the mechanism the Recipe must hook. All references are to the current
tree.

- **NVFP4 two-level scaling.** `global_scale = g_amax / (6·448)`; each 16-wide
  block's FP8-E4M3 `block_scale = fp8(b_amax / (6·global_scale))`. The per-tensor
  `g_amax` is a *range-only* knob (README §1–2).

- **At inference**, for a dynamic-block input quantizer, `TensorQuantizer._fake_quantize`
  calls `dynamic_block_quant(inputs, block_size, amax, ...)`
  (`nn/modules/tensor_quantizer.py:817`). The `amax` it passes is
  `self._get_amax(inputs)` (`:807`).

- **`_get_amax`** (`:655`) returns `self._amax` **if the buffer exists**, else
  computes the per-tensor amax dynamically from the input:

  ```python
  if hasattr(self, "_amax"):
      amax = self._amax            # static (calibrated) global scale
  else:
      amax = reduce_amax(inputs)   # fully dynamic global scale
  ```

  So the calibrated `g_amax` lives in the ordinary `_amax` buffer; the block
  scales stay dynamic and only the per-tensor normalization is frozen.

- **At export**, `NVFP4QTensor.get_activation_scaling_factor` (`qtensor/nvfp4_tensor.py:201`)
  emits the activation global scale as `export_amax() / (maxbound·448)` — i.e.
  the calibrated `g_amax / (6·448)`. This becomes the deployed `input_scale`.

### Verified: the default NVFP4 input quantizer is already calibrated (no blocker)

There are **two independent `type` flags**, which an earlier draft of this design
conflated:

- **top-level `type`** → maps to `self._dynamic` (`tensor_quantizer.py:239`),
  **defaults to `"static"`**. The nvfp4 config never sets it ⇒ `_dynamic = False`.
- **`block_sizes["type"] = "dynamic"`** → makes only the *per-block scales*
  dynamic; it does **not** touch `_dynamic`.

So the default NVFP4 input quantizer has `_dynamic = False`, and **plain
`max_calibrate` already calibrates it**: it gets a `MaxCalibrator` (`axis=None`,
per-tensor), `collect()` runs in the forward loop (`:1093`, gated on
`not self._dynamic`), and `load_calib_amax()` writes the per-tensor running max
into the standard `_amax` scalar buffer. `export_amax()` returns it cleanly — no
`assert not self._dynamic` is hit because `_dynamic` is `False`.

Empirically confirmed (CPU, default nvfp4 input-quantizer config): `_dynamic` is
`False`, calibrator is `MaxCalibrator`, and after two calibration forwards
`_amax` holds the literal per-tensor max (= literal `B_max`) and exports cleanly.

**Consequence:** there is **no storage blocker and no invariant to relax.** The
calibrated `g_amax` already lives in `_amax`. `nvfp4_act_max` simply **overwrites that
same `_amax` scalar** with the recipe value instead of the literal max (§5). The
only genuinely new state is the transient per-block-amax histogram needed to
derive `B_min` — plain `max` keeps only the per-tensor max, not the block
distribution.

---

## 3. The algorithm (Recipe → procedure)

**v1 scope (decided):** implement the **`b_min_anchored`** strategy only — the
study's robust default that tracks the oracle across unseen outlier growth
(README §4 "Robustness… anchor to B_min"). `b_max_anchored` / `log_center` are
deferred to a later iteration (see §10); the config still carries a `strategy`
field so they can be added without a config break.

Applied **only** to NVFP4 dynamic-block input quantizers (other quantizers fall
back to plain max — see §4). Weights are always plain max.

**Per input quantizer:**

1. **Collect the per-block amax distribution** over the calibration forward pass.
   For each activation tensor of shape `[..., H]`, reduce over 16-wide blocks
   along the last dim (`reduce_block_amax(x, {-1: 16})`) to get per-block amaxes,
   and accumulate them into a **base-2 log-spaced histogram** (bounded memory,
   distributed-syncable by summation). Bin over `log2(b_amax)` for **non-dead**
   blocks; track the dead-block fraction separately. "Dead" = exact zero or below
   a relative floor (e.g. `< B_max / 1e6`); near-zero blocks "go gracefully
   subnormal" (README §4) and must not be allowed to drag `B_min` to 0.
   Base 2 is chosen so bin edges align with FP8-E4M3 exponent boundaries
   (one octave = one FP8 exponent step), making the histogram directly readable
   in regime terms; the base does not change the statistics.

2. **Robust statistics** from the histogram:
   - `B_min` = **low percentile** (default **1st**, `b_min_percentile`) of the
     *represented* (non-dead) blocks. This is the stable anchor — governed by the
     preceding normalization layer, consistent across data.
   - `B_max` = **high percentile** (default **99.99th**, `b_max_percentile`) of
     all block amaxes. Used only for the guardrails below in this strategy.
     `b_max_percentile = 100.0` ⇒ literal running max (no histogram needed for
     this field; reuses `max_calibrate`'s running max).

   Asymmetric percentiles are intentional: saturation (top) is catastrophic so we
   keep all but the top 0.01%; subnormal (bottom) is graceful so we can drop the
   bottom ~1%.

3. **Choose `g_amax`** (`b_min_anchored`). Anchor the bottom edge of the
   normal-FP8 window near the stable `B_min` and spend the format's full range as
   upward (saturation) insurance:

   ```text
   g_amax = rho · B_min            # rho default 16384, must be in (0, 28672)
   ```

   `rho` is the deliberate split of the fixed 28672× window: `cushion_below ·
   headroom_above = 28672`, so `rho = 16384` gives ~16384× upward headroom and a
   `28672/16384 = 1.75×` downward cushion. `rho` must be **< 28672** (the cliff at
   which `B_min` sits exactly on the subnormal edge, zero cushion). See §3a for
   the config field.

4. **Guardrails** (README §4 "Guardrails"):
   - **Sanity floor:** clamp `g_amax >= margin · B_max` (default `margin = 1.0`,
     `B_max` = the chosen percentile). Binds only when the dynamic range is large
     (`B_max/B_min` between `rho` and 28672); otherwise `rho·B_min` already
     dominates.
   - **Range-exceeds-format:** if `B_max / B_min > 28672`, no single `g_amax`
     covers the range. `log()`/`warn` loudly, fall back to `g_amax = B_max`
     (no-saturation edge), and recommend outlier mitigation (SmoothQuant /
     per-channel / higher-precision fallback) — **do not** silently pick a value
     that saturates the tail.

5. **Write `g_amax`** onto the input quantizer as the static activation global
   scale (storage mechanism per §5).

### 3a. `rho` is a quant-config parameter

`rho` is exposed on the algorithm config (§6, `NVFP4ActMaxCalibConfig.rho`):

- **default `16384`**,
- **constraint `0 < rho < 28672`** — enforced by a pydantic validator;
  `28672` is FP8-E4M3's normal dynamic range, the hard cliff where the bottom of
  the window reaches the subnormal edge and the downward cushion vanishes,
- documented as "larger `rho` ⇒ more upward outlier headroom but smaller downward
  (subnormal) cushion; keep `rho < 28672`, recommended `~16384`."

> **Note — co-scaling with `b_min_percentile`.** Because `g_amax = rho ·
> B_min(percentile)`, both `rho` and the `B_min` percentile scale `g_amax`. Keep
> their roles distinct: `b_min_percentile` selects *which* blocks define the floor
> (data property — "ignore the dead bottom 1%"), `rho` sets the *budget split*
> (window placement). Tune one at a time.

### 3b. The result is NOT guaranteed ≥ plain-`max` (by design)

Plain `max` calibration sets `g_amax = literal B_max` (the per-tensor literal
max). Because `b_min_anchored` derives `g_amax` from the **robust** `B_min`/`B_max`
percentiles, a **single super-large outlier** that lands above the 99.99th
percentile is excluded from the statistics and therefore **does not raise**
`g_amax`. So the calibrated `g_amax` can be **smaller** than `plain_amax` when
such a freak outlier exists.

This is intentional — the same rationale as percentile/histogram calibration for
INT8/FP8: chasing one freak block (1 in millions) would inflate `g_amax` for the
whole tensor and push every other block toward subnormal. Letting that rare block
clip preserves bulk fidelity. In the **typical** case (no freak beyond p99.99 and
dynamic range `< rho`) the result is several-fold **larger** than `plain_amax`
(more headroom).

**Escape hatch** for callers who require "nothing seen in calibration ever clips"
(`g_amax >= plain_amax`):

- set `b_max_percentile = 100.0` (literal max), and/or
- the sanity floor `margin · B_max` with `B_max` = literal max then guarantees
  `g_amax >= plain_amax`.

This assumes the extreme tail is non-representative (one-off spike). If an outlier
**recurs** every inference, it is not an outlier — fix it with outlier mitigation,
not `g_amax`.

---

## 4. Scope of application within a model

`nvfp4_act_max` is a drop-in superset of `max`:

- **Weights:** always plain max calibration (existing `max_calibrate`).
- **NVFP4 dynamic-block input quantizers:** the Recipe above.
- **All other input quantizers** (FP8 per-tensor, INT8, static NVFP4, disabled,
  etc.): plain max calibration — identical to today. This matters for mixed
  recipes like `super-nvfp4` where only MoE experts are NVFP4 and shared
  experts / mamba linears are FP8.

Detection predicate for "apply Recipe":
`is_enabled and block_sizes.type == "dynamic" and num_bits == (2,1) and
block_sizes.scale_bits == (4,3)`.

---

## 5. Write-back: overwrite the existing `_amax` (no new state)

As verified in §2, the NVFP4 input quantizer's global scale already lives in the
standard `_amax` buffer, written by plain `max_calibrate`. `nvfp4_act_max` therefore
just **replaces that scalar** with the recipe-computed `g_amax`:

```python
# input_quantizer already holds the plain-max _amax after max_calibrate;
# overwrite it in place with the recipe value (same shape: scalar).
input_quantizer.amax = g_amax        # _amax_setter_helper copies into the buffer
```

- `_dynamic` is `False`, so the `amax` setter/getter and `export_amax()` all work
  unchanged — no assert is tripped, no invariant relaxed.
- Shape and dtype are identical to plain max (scalar per-tensor), so save/restore
  and export are unaffected.
- **No** dedicated `_global_amax` buffer and **no** `tensor_quantizer.py` /
  `nvfp4_tensor.py` plumbing changes are needed. (An earlier draft proposed an
  Option A/B storage decision here; it was based on the mistaken belief that the
  quantizer was `_dynamic` — see §2. That decision is withdrawn.)

The only new state is the **transient per-block-amax histogram** used during
calibration to derive `B_min`/`B_max`; it is discarded once `g_amax` is written.

---

## 6. Code changes (registration choreography)

The codebase registers a calibration algorithm in four coordinated places (same
pattern as `mse`, `local_hessian`, etc.):

1. **`modelopt/torch/quantization/model_calib.py`**
   - New `nvfp4_act_max_calibrate(model, forward_loop, *, strategy="b_min_anchored",
     rho=16384, b_max_percentile=99.99, b_min_percentile=1.0, margin=1.0,
     distributed_sync=True, **kwargs)`.
   - Body:
     1. Attach lightweight per-block-amax histogram collectors to the qualifying
        NVFP4 dynamic-block input quantizers, then run the standard
        `max_calibrate(model, forward_loop)` — this calibrates weights and all
        input quantizers (incl. our NVFP4 ones) into `_amax` and, in the same
        forward pass, fills our histograms. No second forward pass.
     2. For each qualifying NVFP4 input quantizer, compute `g_amax` per §3
        (`b_min_anchored` for v1) and **overwrite `_amax`** in place (§5).
        Non-NVFP4 quantizers keep their plain-max `_amax` untouched.
     3. Distributed sync (sum histograms, or reduce the resulting `g_amax`).
   - Add `"nvfp4_act_max_calibrate"` to `__all__`.
   - *(Possible)* a small `NVFP4ActMaxCalibrator` in `calib/` (log-histogram
     of block amax + percentile/strategy `compute_amax`) to keep `model_calib.py`
     lean and mirror existing calibrators.

2. **`modelopt/torch/quantization/config.py`**
   - `class NVFP4ActMaxCalibConfig(QuantizeAlgorithmConfig)` with
     `method: Literal["nvfp4_act_max"]` and fields:

     | field | type | default | notes |
     |---|---|---|---|
     | `strategy` | `Literal["b_min_anchored"]` | `"b_min_anchored"` | only value in v1; widened later |
     | `rho` | `float` | `16384.0` | **validator: `0 < rho < 28672`**; window-split knob (§3a) |
     | `b_min_percentile` | `float` | `1.0` | low percentile of non-dead blocks |
     | `b_max_percentile` | `float` | `99.99` | high percentile; `100.0` ⇒ literal max |
     | `margin` | `float` | `1.0` | sanity-floor multiplier on `B_max` |
     | `distributed_sync` | `bool` | `True` | sync stats across DP/TP/EP |

   - Add a `@model_validator` (or `Field(gt=0, lt=28672)`) enforcing
     `0 < rho < 28672`, with a message pointing at the cliff rationale (§3a).

3. **`modelopt/torch/quantization/mode.py`**
   - Import `nvfp4_act_max_calibrate`; register
     `@CalibrateModeRegistry.register_mode class NVFP4ActMaxCalibrateModeDescriptor(BaseCalibrateModeDescriptor)`
     with `config_class = NVFP4ActMaxCalibConfig` and `_calib_func = nvfp4_act_max_calibrate`.
     Mode name resolves automatically to `nvfp4_act_max_calibrate`
     (`_get_mode_name("nvfp4_act_max")`).

4. **Recipe config (the "recipes/ folder" = `modelopt_recipes/`)**
   - The quantization config presets live in the `modelopt_recipes` package
     (`config_loader.py:81` `BUILTIN_CONFIG_ROOT = files("modelopt_recipes")`),
     under `modelopt_recipes/configs/ptq/presets/model/`.
   - Add `modelopt_recipes/configs/ptq/presets/model/nvfp4_act_max.yaml`
     (algorithm name pending final confirmation — see note): like `nvfp4.yaml`
     (reuses the `w4a4_nvfp4_nvfp4` unit for weight+input NVFP4) **plus a FP8
     KV-cache via the `kv_fp8_cast` unit** (KV-cache is out of scope for the
     algorithm itself — §4 — so it is handled purely by the recipe). Sketch:

     ```yaml
     # modelopt-schema: modelopt.torch.quantization.config.QuantizeConfig
     imports:
       base_disable_all: configs/ptq/units/base_disable_all
       w4a4_nvfp4_nvfp4: configs/ptq/units/w4a4_nvfp4_nvfp4
       kv_fp8_cast: configs/ptq/units/kv_fp8_cast
       default_disabled_quantizers: configs/ptq/units/default_disabled_quantizers
     algorithm: nvfp4_act_max          # or {method: nvfp4_act_max, rho: 16384}
     quant_cfg:
       - $import: base_disable_all
       - $import: w4a4_nvfp4_nvfp4
       - $import: kv_fp8_cast
       - $import: default_disabled_quantizers
     ```

   - Optionally expose `NVFP4_ACT_MAX_CFG` in `config.py` via
     `_load_quantize_config_dict(...)`, matching the other `*_CFG` constants.

> **Note on naming (decided: `nvfp4_act_max`).** The `act` (activation)
> qualifier distinguishes it from plain `max` / the existing
> `super-nvfp4-max-calib.yaml` (`algorithm: max`): weights are calibrated with
> `max`, while the **activation** global scale uses this recipe. (Caveat for
> readers: the activation path is not literally "max" — it is the robust
> percentile / `B_min`-anchored global-scale calibration of §3; the name reflects
> the weight calibration and the max-family lineage.) Derived identifiers:
> `nvfp4_act_max_calibrate` (function/mode), `NVFP4ActMaxCalibConfig` (config),
> `NVFP4ActMaxCalibrateModeDescriptor` (descriptor).

---

## 7. Distributed / parallelism

- **Histogram collection** is summable across DP/TP/EP groups (all-reduce SUM),
  giving identical `g_amax` across ranks — the cleanest sync.
- Alternatively, compute `g_amax` per rank and reduce the **scalar** (max of
  `g_amax`, or recompute from reduced `B_max`/`B_min`). Histogram-sum is preferred
  for determinism.
- Reuse the existing `max_calibrate` DP/TP/EP sync for the weight + non-NVFP4
  portions.

---

## 8. Testing plan

- **Unit (math):** feed synthetic activations with known block-amax spread; assert
  `g_amax == rho · B_min` when the dynamic range is small, the sanity floor binds
  to `margin · B_max` when `rho·B_min < B_max`, and the range-exceeds-format
  guardrail triggers (and falls back to `B_max`) when `B_max/B_min > 28672`.
  Assert the `rho` validator rejects `rho >= 28672` and `rho <= 0`.
- **Outlier handling:** with a single super-large block injected above the 99.99th
  percentile, assert `g_amax < plain_amax` by default, and `g_amax >= plain_amax`
  once `b_max_percentile = 100.0` (the escape hatch, §3b).
- **Integration:** small model, `mtq.quantize` with the `nvfp4_act_max` recipe;
  assert (a) weight amax equals the plain-`max` result, (b) a static activation
  global scale is present on NVFP4 input quantizers, (c) export emits a finite
  positive `input_scale`, (d) non-NVFP4 quantizers are byte-identical to plain
  `max`.
- **Numerical validation:** reuse `nvfp4_global_scale_study.py` harness to confirm
  the calibrated `g_amax` reduces MSE vs plain `max` under simulated unseen
  outlier growth (reproduce the README §4 robustness table).
- **Regression:** `nvfp4` (plain) recipe unchanged; round-trip save/restore of the
  new static global scale.

---

## 9. Decisions and open questions

**Resolved in review (this iteration):**

- **Strategy for v1 — `b_min_anchored` only.** The robust default; the literal
  `b_max_anchored` Recipe and `log_center` are deferred (§10). `strategy` stays in
  the config so they can be added without a config break.
- **`rho` is a config parameter** — `NVFP4ActMaxCalibConfig.rho`, default `16384`,
  constraint `0 < rho < 28672`, documented as "keep below the 28672 cliff" (§3a).
- **`B_max` uses a percentile (99.99th), not the literal max** — with
  `b_max_percentile = 100.0` as the literal-max escape hatch. Consequence: the
  calibrated `g_amax` is **not guaranteed ≥ plain-`max`** on a freak outlier, and
  that is intended (§3b).
- **`B_min` uses the 1st percentile over non-dead blocks**; **`B_max` the
  99.99th**; **`margin = 1.0`** (the `B_max` sanity floor — `g_amax` never sits
  below the calibrated top). All confirmed (§3, step 2; §3 step 4).

- **KV-cache is out of scope** for the algorithm. The recipe gets its FP8 KV
  cache from the `kv_fp8_cast` unit (§6 item 4); `nvfp4_act_max` touches only
  NVFP4 linear input quantizers.

- **Algorithm name — `nvfp4_act_max`** (decided). Derived identifiers:
  `nvfp4_act_max_calibrate`, `NVFP4ActMaxCalibConfig`,
  `NVFP4ActMaxCalibrateModeDescriptor` (§6 naming note).

- **Storage mechanism — resolved by verification (§2, §5).** The earlier
  Option-A/B decision is **withdrawn**: the NVFP4 input quantizer is `_dynamic =
  False`, already carries its global scale in `_amax`, so the algorithm just
  overwrites that scalar. No new buffer, no invariant change.

**No open decisions remain — the design is ready to implement.** Any later
tuning (additional strategies, MSE refinement, KV-cache support) is tracked in
§10 and does not block v1.

---

## 10. Deferred to later iterations

- `b_max_anchored` (`g_amax = B_max · slack^0.65`) and `log_center`
  (`sqrt(B_max · 28672·B_min)`) strategies — config-compatible additions to the
  `strategy` field.
- Optional MSE/Hessian-weighted 1-D refinement constrained to the feasible window
  (README §4 step 5).
- NVFP4 KV-cache quantizer support, pending the §9.3 decision.
