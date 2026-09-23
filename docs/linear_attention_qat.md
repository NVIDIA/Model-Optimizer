# Linear-attention numerical emulation for training

This experimental integration adds dynamic FP8 E4M3 and signed INT8 state fake quantization to Megatron
GatedDeltaNet's chunked training path. It rounds the recurrent state at chunk boundaries
and supports FP8 QDQ on the materialized WY activation `W` before the state-read matmul. Both sites use an
identity straight-through estimator (STE). The underlying tensors remain floating point;
this feature does not provide compressed inference state or native FP8 matmul acceleration.

The implementation adapts [PR #2455](https://github.com/NVIDIA/Model-Optimizer/pull/2455)
at `13c7e2456f2e9d079c9ef822742eeaa634353802` onto ModelOpt
`051d6adb204f10cd3e78d0f824f31a5a01d54831`. It provides the shared state-QDQ foundation.
[Decode-aware training](linear_attention_decode.md) adds GDN/KDA token-state quantization,
decay approximation, and SSM replay with an exact-prefix handoff. Prefill operand QDQ
and approximate inverse are subsequent deliveries; neither is required by this branch.

## Requirements and scope

- Install `fla-core==0.5.1` alongside the training framework. FLA and Triton remain optional
  for ordinary ModelOpt imports and the PyTorch references.
- On Hopper with Triton 3.4 or newer, install `tilelang==0.1.8` and `apache-tvm-ffi==0.1.9`
  for FLA's backward fallback. This path supports **BF16 Q/K/V only** and rejects FP32
  before launch. The adapter expands grouped Q/K heads for that backend; autograd sums their
  gradients back to the original heads. This adds temporary Q/K activation storage.
- The fused GDN implementation accepts **chunk size 64 only**, including backward.
- Fused GDN FP8 state QDQ requires NVIDIA SM89 or newer for native E4M3 conversion.
  INT8 state QDQ and W-only QDQ can run on SM86.
- State supports dynamic E4M3 or signed narrow-range INT8; W supports dynamic E4M3.
  Both require `pass_through_bwd=True`. Static scales,
  clipping-aware backward, real quantization, rotations, and custom format backends are rejected.
- Context parallelism with either numerical site enabled is rejected. Distributed checkpoint
  save/restore is qualified at TP=1 and TP=2 with unchanged topology and PP=1. Pipeline
  parallelism and resharding across topology changes remain unqualified.
- Megatron's dynamic-batching `ssm_prefill` and `ssm_decode` paths are outside this integration.

## Configure training

Apply the recipe after constructing the Megatron model. The module pattern must match
at least one converted GatedDeltaNet layer; dense attention layers in a hybrid model are
ignored by the execution policy.

```python
import modelopt.torch.quantization as mtq

config = {
    "quant_cfg": [
        {"quantizer_name": "*", "enable": False},
        {
            "quantizer_name": "*gdn_state_quantizer",
            "cfg": {"num_bits": (4, 3), "type": "dynamic", "axis": (0, 1)},
        },
        {
            "quantizer_name": "*gdn_w_quantizer",
            "cfg": {"num_bits": (4, 3), "type": "dynamic", "axis": (0, 1, 2)},
        },
    ],
    "algorithm": None,
    "linear_attention": [
        {
            "module_name": "decoder.layers.*.self_attention",
            "cfg": {
                "backend": "fla",
                "chunk_size": 64,
                "state": {"mode": "chunk", "block_v": 64, "quantize_initial": True},
                "solve": {"method": "exact"},
            },
        },
    ],
}
model = mtq.quantize(model, config)
```

To select INT8 state, replace only the state quantizer's `cfg` with:

```python
{"num_bits": 8, "unsigned": False, "narrow_range": True,
 "type": "dynamic", "axis": (0, 1), "pass_through_bwd": True}
```

The reusable recipe unit is `configs/ptq/units/gdn_state_int8_dynamic`.
For KDA, use the same attributes with `*kda_state_quantizer`. Existing FP8
recipes and checkpoints retain their format.

Dynamic scaling requires no calibration loop. Continue with the framework's normal
forward, backward, and optimizer steps. Enable projection quantization through separate
recipe entries if needed. The standard projection presets keep the GDN numerical handles
disabled; append the GDN units after standard exclusions when composing recipes.

Execution policies use last-match precedence: each match replaces the complete policy,
including defaults for omitted fields. There is no nested merge. The resolved policy is
reported at conversion and is available on each converted module as
`linear_attention_config`. Quantizer enables/formats and resolved policies are saved
through the ModelOpt checkpoint APIs, including changes made after conversion.

Without an explicit execution policy, enabled GDN recipe units use the defaults above.
When both GDN handles are disabled, the adapter calls the original training kernel.

## Numerical contract

`W` is an activation, **not a learned weight**. The W quantizer sees `[B,T,Hv,Dk]`
and reduces over `Dk`, using one scale per token/head. Forward's quantized W is saved
for backward, avoiding a second quantizer invocation and ensuring identical values.
This costs one saved W activation when that site is enabled.

The state handle specifies a fused operation rather than calling `TensorQuantizer` on a
materialized state. Each sequence/head uses one dynamic scale per `[Dk, block_v]` tile;
`block_v` can be 16, 32, 64, or 128. Partial value tiles are valid. The scale is
`amax / 448` for E4M3 or `amax / 127` for INT8, with scale one for an all-zero tile.
INT8 uses signed codes in `[-127, 127]`, zero point zero, round-to-nearest-even,
and saturation. Scales are detached FP32 values and QDQ returns floating values. Rounding occurs on the initial state
and after each chunk's state update. The chunk's outputs use its incoming rounded state;
they are computed before rounding the outgoing state. Initial/final state gradients
propagate through these rounding events using identity STE.

The triangular solve remains exact. No approximate-inverse algorithm is selected or
silently enabled by this integration.

## References and verification

The CPU-only reference package is `modelopt.torch.quantization.linear_attention`:

| Function | Contract |
| --- | --- |
| `recurrent_delta_rule_reference` | Exact GDN scalar-decay or KDA per-key-decay recurrence |
| `chunk_gdn_reference` | Exact GDN triangular solve and chunk algebra; optional state/W QDQ |
| `state_fp8_qdq_reference` | Dynamic E4M3 state tiles with identity STE |
| `state_qdq_reference` | E4M3 or INT8 state tiles selected by `state_format` |

Use float64 for algebra/gradcheck and float32 for GPU comparison. References support
grouped value heads, packed sequences, nonzero initial states, tails, and either state
layout. They favor explicit arithmetic over speed and are not an efficient training backend.

```bash
PYTHONPATH=. pytest -q tests/unit/torch/quantization/test_linear_attention_reference.py \
  tests/unit/torch/quantization/plugins/test_gated_delta_net.py \
  tests/unit/torch/quantization/test_quantize_cpu.py
PYTHONPATH=. pytest -q \
  tests/gpu/torch/kernels/quantization/linear_attention/test_fla_chunk_gated_delta_rule.py
PYTHONPATH=. pytest -q \
  tests/gpu_megatron/torch/quantization/plugins/test_megatron_gated_delta_net.py
```

The GPU suite compares outputs, final states, and gradients for every differentiable
input against the declared surrogate. It also covers packed boundaries, state layout,
fused gate/beta activation, activation checkpointing, and unsupported-policy errors.
The Megatron tests exercise state-only, W-only, and combined QAT after a distributed
checkpoint round trip at TP=1 and TP=2, including an edited execution policy and an optimizer
step. The `gpu` and `gpu_megatron` Nox sessions install pinned FLA and TileLang dependencies.

Prior FP8 qualification covered these environments (the counts below predate INT8):

- RTX A6000/SM86: Python 3.12.8, Torch 2.9.1, Triton 3.5.1, FLA 0.5.1. The suite passes
  17 cases; 11 native state-FP8 cases and the Hopper-only capability test are skipped.
- H100/SM90 in NeMo 26.08: Python 3.12.3, Torch 2.13.0a0 (NV 26.6), Triton 3.7.0,
  FLA 0.5.1, TileLang 0.1.8, and TVM-FFI 0.1.9. The suite passes 20 cases, including all
  state scale tiles and BF16 state/W gradient comparisons against float32 references.
  Nine FP32 cases are outside the supported Hopper contract; its early rejection is tested.

In the same H100 environment, Megatron-Core 0.19.1 and Transformer Engine 2.18.0 pass all
eight integration cases: state-only, W-only, and combined QAT at TP=1 and TP=2, plus two
context-parallel rejection cases. The training cases verify changed GDN branch outputs,
disabled-path parity, distributed checkpoint restore, backward, and an optimizer step.
Both direct-forward and older split-forward Megatron layouts are handled by the adapter;
the runtime qualification above uses the direct-forward layout in 0.19.1.

No model-quality, QAT recovery, or throughput result is claimed.
