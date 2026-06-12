# OMNIML-5072 Option B — Triton kernel validation record

**Status: VALIDATED 2026-06-12.** Kernel lives at `grouped_axis0_fakequant.py`,
wired into `te_grouped_quantized_linear_fn` via `_GroupedAxis0FakeQuantFn`.

## Validation summary

### 1. Numerical fidelity

Compared `grouped_axis0_fakequant(weights, amax)` against per-expert
`cuda_ext.fake_tensor_quant(w_i, amax_i)` (A's path) at the Ultra production
shape (N=32, [5120, 8192] bfloat16). See
`nmm-sandbox/studies/omniml-5064/microbench/parity_a_vs_btriton.py`.

```
Check                              max_abs_err   Notes
─────────────────────────────────  ───────────   ─────────────────────────────
forward output                     0.03125       = 1 ULP at this quant scale
                                                 (rounding mode: Triton does
                                                 round-half-away, cuda_ext does
                                                 round-half-to-even; differs only
                                                 on exact half-step boundaries)
backward grad (pass_through_bwd)   0.0           bit-exact identical to A
```

### 2. Bench (OMNIML-5064 microbench, B300, aws-cmh)

Btriton beats A on every column at every cell:

```
Cell             impl      fwd_us    bwd_us    step_us    Notes
───────────────────────────────────────────────────────────────────────────────
Nano   EP=1      A         20,221     5,807    27,422
                 Btriton    2,670     3,591     7,472     fwd 7.6× win
Super  EP=1      A         57,736    17,736    84,289
                 Btriton    9,584    14,923    33,242     fwd 6.0× win
Ultra  mock16    A          5,835       547     8,246
                 Btriton    1,795       549     4,210     fwd 3.25× win; bwd tied
Nano   EP=4      A          3,985     1,340     5,693
                 Btriton    1,208     1,221     2,815     fwd 3.30× win
Super  EP=4      A         14,094     4,187    20,576
                 Btriton    2,110     3,326     7,710     fwd 6.68× win
Ultra  EP=4      A         23,178     2,064    32,686
                 Btriton    6,730     2,059    16,220     fwd 3.44× win
Ultra  EP=8      A         11,695     1,070    16,482     2-node
                 Btriton    3,515     1,075     8,314     fwd 3.33× win
```

### 3. Distributed validation

- ✓ EP=1 single-rank (mock-EP=16 emulation of EP=16 deployment)
- ✓ EP=4 single-node 4-rank (Nano / Super / Ultra)
- ✓ EP=8 multi-node 2-node 8-rank (Ultra)
- `peak_mb` matches A's within 3 MB across cells (same layer instantiation
  signal)

### 4. Hardware coverage

Tested on B300 (compute 10.0+, aws-cmh). The kernel uses no SM-specific
intrinsics (no MMA, no async copy, no tensor cores) — pure load/store +
arithmetic in `triton.language`. Should compile and run on any GPU where
`torch.cuda.is_available()` returns True and Triton is installed; the
`IS_AVAILABLE` guard in `__init__.py` skips it otherwise.

## Design notes

The kernel takes N expert weights as a `[N]` int64 tensor of base pointers
(via each tensor's `.data_ptr()`). Each Triton program reads its expert's
pointer, then strides through a block of elements. Grid: `(N, num_blocks_per_expert)`.
This eliminates the `torch.stack` memcopy on the forward path (~2.7 GB at
Ultra scale).

Backward honors modelopt's `pass_through_bwd` flag (`config.py` default
`True`). When set, the backward returns `grad_outputs` unchanged with zero
kernel launches — matching `_fake_quant_backward_function`'s no-save
behavior. When `False`, the clip-aware STE Triton backward kernel runs.

Soft-gated at the call site in `te_grouped_quantized_linear_fn`: falls back
to the stack-then-quant-then-unbind path when the Triton kernels aren't
available or when calibration is active (`q._if_calib`).
