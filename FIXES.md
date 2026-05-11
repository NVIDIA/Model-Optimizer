# Model-Optimizer2 — Fixes for NVFP4 MoE Expert Quantization

Fixes applied 2026-05-01 targeting NVFP4 W4A4 experts-only MSE quantization of
`_QuantFusedExperts` models (Qwen3.6-35B-A3B, Qwen3-MoE, GLM-5.1-style fused experts).
All fixes verified via end-to-end run: 0/2,013,265,920 zero `weight_scale` tensors,
20,480/20,480 MSE weight calibrations completed.

---

## Fix 1 — FP8 underflow in per-block weight scales (both static and dynamic export paths)

**File:** `modelopt/torch/quantization/qtensor/nvfp4_tensor.py`  
**Functions:** `get_weights_scaling_factor` (~line 169), `get_weights_scaling_factor_from_quantizer` (~line 126)

**Export path split:** `_export_quantized_weight` in `unified_export_hf.py` (line 556) branches on
`NVFP4QTensor._is_static_quantizer(weight_quantizer)`:
- **Static** (MSE-calibrated, `global_amax` set): calls `get_weights_scaling_factor_from_quantizer`
  — reads stored per-block `_amax` from the quantizer; does NOT recompute from weights.
- **Dynamic** (uncalibrated, no `global_amax`): calls `get_weights_scaling_factor`
  — recomputes per-block amax from the actual weight tensor.

**Root cause (both paths):** Before casting to FP8 E4M3FN, neither path clamped the
float32 scale. The existing `per_block_scale[per_block_scale == 0] = 1.0` guard only
catches exact float32 zeros. Any value in the range `(0, 2^-9)` passes through and
silently underflows to `0.0` on `.to(torch.float8_e4m3fn)`, making those weight
blocks produce zero output at inference.

- **Dynamic path formula:** `per_block_amax / (6 * wsf2)` — underflows when
  `per_block_amax < global_amax * 4.35e-6`.
- **Static path formula:** `per_block_amax * 448 / global_amax` — same underflow
  condition; the MSE-clamped amaxes (`min=2e-3`) in practice prevent this for typical
  models, but the gap exists without the clamp.

**Fix:** Add `.clamp(min=2**-9)` before the FP8 cast in both paths:

```python
# Dynamic path — get_weights_scaling_factor (~line 174):
_FP8_E4M3FN_MIN = 2**-9
per_block_scale = per_block_scale.clamp(min=_FP8_E4M3FN_MIN)
per_block_scale = per_block_scale.to(torch.float8_e4m3fn)

# Static path — get_weights_scaling_factor_from_quantizer (~line 127):
_FP8_E4M3FN_MIN = 2**-9
per_block_scale = (per_block_scale * 448.0 / per_block_scale_max).clamp(
    min=_FP8_E4M3FN_MIN
).to(torch.float8_e4m3fn)
```

---

## Fix 2 — MSE weight calibration skips all _QuantFusedExperts experts (0 it/s)

**File:** `modelopt/torch/quantization/model_calib.py`  
**Function:** `mse_calibrate` (~line 424)

**Root cause:** The weight-quantizer discovery loop uses
`quantizer_attr_names(weight_name).weight_quantizer` which yields the *singular*
attribute name `gate_up_proj_weight_quantizer`. `_QuantFusedExperts` stores per-expert
quantizers in a *plural* `nn.ModuleList` named `gate_up_proj_weight_quantizers`.
`getattr(module, "gate_up_proj_weight_quantizer", None)` returns `None`, so all
20,480 expert quantizers are silently dropped → "MSE weight calibration: 0it".

**Fix:** After the standard singular-attr loop, add a second pass that detects the
plural `{param}_weight_quantizers` ModuleList pattern and enqueues each per-expert
quantizer individually using a `(param_name, expert_idx)` tuple as the weight key.
Step 3 unpacks the tuple to extract the per-expert weight slice:

```python
# New block after existing weight_attr_names loop (~line 424):
for param_name, _ in parent_module.named_parameters(recurse=False):
    qlist = getattr(parent_module, f"{param_name}_weight_quantizers", None)
    if not isinstance(qlist, nn.ModuleList):
        continue
    for expert_idx, wq in enumerate(qlist):
        if isinstance(wq, TensorQuantizer) and wq.is_enabled:
            if getattr(wq, "_calibrator", None) is not None:
                weight_quantizers.append((parent_module, (param_name, expert_idx), wq))

# In step 3, unpack tuple weight_name (~line 448):
if isinstance(weight_name, tuple):
    param_name, expert_idx = weight_name
    weight = getattr(parent_module, param_name)[expert_idx]
else:
    weight = getattr(parent_module, weight_name)
```

---

## Fix 3 — Zero/invalid amax for uncalibrated experts at export

**File:** `modelopt/torch/export/moe_utils.py`  
**Function:** `_export_fused_experts`

**Root cause:** Experts not reached by any calibration batch have `_amax = 0` or
garbage-large/NaN values. These produce zero or corrupt FP8 per-block scales. In
addition, `global_amax` was not recomputed when it was already set (stale zero from
an uncalibrated expert), causing division-by-zero in the static FP8 scale formula
`per_block_scale * 448 / (global_amax / 6)`.

**Why the existing expert-amax sync doesn't help here:**

`max_calibrate` calls `layer_sync_moe_local_experts_amax` after the forward pass,
which calls `sync_moe_expert_amax`. This syncs amax across uncalibrated experts —
but only on modules that have the `layer_sync_moe_local_experts_amax` method.
That method is defined on `_QuantSparseSequentialMoe` (transformers <5.0 style
sequential experts: `nn.ModuleList` of standalone `nn.Linear`-per-expert).

Qwen3.6 and GLM-5.1 use `_QuantFusedExperts` (transformers 5.0+ fused style:
single 3-D `nn.Parameter` for all experts), which has **no**
`layer_sync_moe_local_experts_amax` method. The sync is intentionally absent:

- **Input quantizers** are *shared* across all experts in `_QuantFusedExperts`
  (`gate_up_proj_input_quantizer`, `down_proj_input_quantizer` — single objects).
  They accumulate amax from every token regardless of routing, so they are always
  calibrated and need no cross-expert sync.
- **Weight quantizers** are *per-expert* (`gate_up_proj_weight_quantizers[i]`).
  Syncing weight amaxes *across* experts would be semantically wrong — different
  experts have different weight distributions and should each have their own scale.

So for `_QuantFusedExperts`, there is deliberately no sync, and experts that receive
zero routing tokens during calibration are left with `_amax = 0`. The right place
to handle this is at export time, where each expert's weight is available and a
per-block fallback can be computed from the actual weights.

**Fixes (three layers of defence):**

1. **Per-block amax patching** (~line 140): invalid blocks (NaN / inf / negative /
   `< 1e-4` / `> 1e6`) are replaced with per-block weight-derived fallback, clamped
   to `min=2e-3` (above FP8 E4M3FN min subnormal). Reshape bug fixed: fallback is
   reshaped to match `amax_cpu.shape` (was producing `(H*W, 1)` instead of `(H, W)`).

2. **Scalar-amax fallback → per-block** (~line 192): when `_amax` is missing or a
   bad scalar, a full `(H, num_blocks_per_row)` per-block tensor is computed from
   weights so the static export path can call `.view(expected_shape)` without error.

3. **Always recompute global_amax** (~line 222): removed the
   `not (hasattr(wq, "global_amax") and wq.global_amax is not None)` guard.
   A stale zero `global_amax` from an uncalibrated expert must always be
   overwritten with `wq._amax.float().amax().clamp(min=2e-3)`.

---

## Fix 4 — quant_summary save fails before export dir exists

**File:** `modelopt/torch/quantization/model_quant.py`  
**Function:** `print_quant_summary` (~line 598)

**Root cause:** `print_quant_summary` is called right after calibration, before the
export step creates the output directory. `open(path, "w")` raises
`FileNotFoundError: [Errno 2] No such file or directory`.

**Fix:** Add `os.makedirs(output_dir, exist_ok=True)` before `open`:

```python
# Before:
with open(path, "w", encoding="utf-8") as f:

# After:
os.makedirs(output_dir, exist_ok=True)
with open(path, "w", encoding="utf-8") as f:
```

---

## Fix 5 — quant_summary / extra_repr shows 0.0000 for tiny amax values

**File:** `modelopt/torch/quantization/nn/modules/tensor_quantizer.py`  
**Functions:** `_short_amax`, `_short_tensor` (~line 1115, 1133)

**Root cause:** Default format `".4f"` rounds values like `3.5e-7` (a typical
per-block scale for small-weight experts) to `0.0000`, hiding whether the scale
is legitimately zero or just very small.

**Fix:** Change default `fmt` from `".4f"` to `".2e"`:

```python
# Before:
def _short_amax(self, fmt=".4f"): ...
def _short_tensor(self, tensor: torch.Tensor, fmt=".4f"): ...

# After:
def _short_amax(self, fmt=".2e"): ...
def _short_tensor(self, tensor: torch.Tensor, fmt=".2e"): ...
```

Values now display as e.g. `3.50e-07` instead of `0.0000`.

---

## Fix 6 — Preview input shows only pad tokens (degenerate example output)

**File:** `examples/llm_ptq/hf_ptq.py`  
**Functions:** `pre_quantize` (~line 800), `input_decode` (~line 906)

**Root cause:** The calibration dataloader left-pads batches to `calib_seq=2048`.
Qwen3 uses EOS (`<|im_end|>`) as its pad token. The first item in the first batch
starts with ~1,999 `<|im_end|>` tokens, producing the preview generation
`['<|im_end|>']` before and after PTQ — masking any real quality signal.
Additionally `tokenizer.batch_decode` without `skip_special_tokens=True` pollutes
the displayed input with special tokens.

**Fix — strip leading padding from preview input** (~line 800):

```python
if model_type not in ("whisper",) and tokenizer is not None and tokenizer.pad_token_id is not None:
    first_non_pad = (preview_input_ids[0] != tokenizer.pad_token_id).nonzero(as_tuple=True)[0]
    if first_non_pad.numel() > 0:
        preview_input_ids = preview_input_ids[:, first_non_pad[0]:]
```

**Fix — skip special tokens in input_decode** (~line 906):

```python
# Before:
return tokenizer.batch_decode(input_ids)

# After:
return tokenizer.batch_decode(input_ids, skip_special_tokens=True)
```

---

## New artifact — NVFP4 experts-only MSE recipe

**File:** `modelopt_recipes/general/ptq/nvfp4_experts_only_mse.yaml`

Quantization recipe for NVFP4 W4A4 routed-experts-only with MSE + FP8 scale sweep.
Targets `_QuantFusedExperts` models (Qwen3.6-35B-A3B, Qwen3-MoE) and standard
`nn.Linear`-based expert layouts.

Key settings:
- All quantizers disabled by default; only routed expert weight/input quantizers enabled
- Weight: `static` NVFP4, block size 16, `scale_bits: e4m3`, `num_bits: e2m1`
- Input: `dynamic` NVFP4, same block/bit config
- `algorithm: mse` with `fp8_scale_sweep: true`
- Explicit exclusions: shared experts, attention, routers, `lm_head`, MTP layers

---

## Verification (Run 8, 2026-05-01)

```
Model:    Qwen/Qwen3.6-35B-A3B
Repo:     Model-Optimizer2
Hardware: 8× B200

Quantizers inserted:      21,740
MSE weight calibrations:  20,480 / 20,480  (11:22, ~31 it/s)
Zero weight_scale values: 0 / 2,013,265,920
Shards exported:          3
Total time:               104.7s
Export path:              models/Qwen3.6-35B-A3B-nvfp4-mo2-20260501-155053
```
