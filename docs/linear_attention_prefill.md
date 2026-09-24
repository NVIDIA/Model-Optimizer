# GDN prefill arithmetic for QAT

GDN training can expose all eight logical prefill matmuls through an explicit
`backend="matmul"` policy. Each operand uses ModelOpt `TensorQuantizer`; standard
PyTorch autograd propagates its identity straight-through gradient. The computation
batches chunk-local work and carries state between chunks without detaching it.
This backend materializes operands and the triangular inverse. Its training cost
must be measured separately from the fused FLA state/W path.

## Enable operand quantization

```python
import modelopt.torch.quantization as mtq

model = mtq.quantize(model, {
    "quant_cfg": [
        {"quantizer_name": "*", "enable": False},
        {"quantizer_name": "*linear_attn_sites.*",
         "cfg": {"num_bits": (4, 3), "type": "dynamic", "axis": (0, 1, 2)}},
        {"quantizer_name": "*gdn_w_quantizer",
         "cfg": {"num_bits": (4, 3), "type": "dynamic", "axis": (0, 1, 2)}},
    ],
    "algorithm": None,
    "linear_attention": [{
        "module_name": "decoder.layers.*.self_attention",
        "cfg": {"backend": "matmul"},
    }],
})
```

Each site has `lhs_quantizer` and `rhs_quantizer`. The sole exception is
`state_read.lhs_quantizer`: use the existing `gdn_w_quantizer` handle. It is not
registered twice. Standard projection recipes keep all these handles disabled.

Both operands are presented as four-dimensional row vectors: `[chunk_or_sequence,
head, row, reduction]`. The right operand is transposed after quantization.
E4M3 uses one dynamic scale per row. The exact transformed operands are:

| Site | Left operand | Right operand before transpose |
| --- | --- | --- |
| `key_interaction` | beta-scaled keys | keys |
| `wy_value` | triangular inverse | transposed beta-scaled values |
| `wy_key` | triangular inverse | transposed beta- and prefix-decay-scaled keys |
| `state_read` | WY keys (`gdn_w_quantizer`) | transposed incoming state |
| `state_update` | transposed end-decayed keys | transposed corrected values |
| `output_state` | scaled, prefix-decayed queries | transposed incoming state |
| `output_score` | scaled queries | keys |
| `output_value` | causal, decayed local scores | transposed corrected values |

Stored-state rounding remains separate from temporary state-operand quantization.
Enable `gdn_state_quantizer` separately to round the initial state and chunk writes.

For NVFP4 operands, replace the operand configuration with:

```python
{
    "num_bits": (2, 1),
    "type": "dynamic",
    "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)},
}
```

This uses E2M1 values, E4M3 block scales, and a dynamically computed tensor amax
for each quantizer invocation. Chunk-local sites batch all sequence chunks in one
invocation. State-dependent sites invoke once per chunk over the sequence batch.
These scale domains are part of the numerical contract. Chunk padding is zero;
state carries for completed packed sequences do not incur additional state writes.
NVFP4 support here covers operands, including WY keys, and does not change the
FP8 stored-state codec. Static scales and other quantizer transformations are rejected.

## Select arithmetic experiments independently

The optional `matmul` and `elementwise` policy fields operate even when all
quantizers are disabled:

```python
policy = {
    "backend": "matmul",
    "matmul": {
        "output_value": {"accumulator_dtype": "float16", "reduction_block": 16},
    },
    "elementwise": {"value_residual": "bfloat16"},
}
```

A matmul schedule computes partial products in the working dtype, traverses the
reduction dimension left to right in `reduction_block` elements, and rounds the
accumulator after each addition. Both schedule fields are required together.
`float32`, `float16`, and `bfloat16` are supported rounding formats. This schedule
is a numerical experiment; it does not assert instruction-level hardware behavior.

Elementwise sites are `gate_prefix` (after cumulative summation), `gate_exp`
(after each exponential), `value_residual`, `state_decay`, `state_add`, and
`output_add`. Rounding applies only at the named boundary. These operations and
operand QDQ use identity STE, preserving baseline backward arithmetic.

Working arithmetic is FP32 for FP16/BF16/FP32 input and FP64 for double input,
including inside an outer training-autocast context.
Matmuls respect PyTorch's global precision setting. Set
`torch.set_float32_matmul_precision("highest")` to exclude TF32 from comparisons;
the kernel tests and benchmark do this explicitly.
Outputs are cast to the query dtype; carried state keeps the working dtype. The
triangular solve remains exact. No gate clamps or automatic inverse approximation
are enabled. Gate activation and Q/K normalization follow FLA's formulas.

## Persistence and verification

The policy and enabled handles survive ModelOpt save/restore. Checkpoints from
before these new handles existed restore with the new handles disabled. Context
parallelism is rejected; qualification is limited to unchanged TP topology and
PP=1. All-disabled policies preserve the original framework kernel.

The numerical tests compare outputs, final states, and every differentiable input
gradient, including raw gate parameters, and exercise packed tails and state layout.
FP8 single-site cases use elementwise comparisons. Composed FP32 state/operand QDQ
uses the established GDN relative-L2 bounds (1% outputs/states, 2% gradients):
batched and unbatched GEMM roundoff can cross a later rounding threshold. NVFP4
checks the codec independently at each actual operand shape and compares the full
computation and gradients with the chunk oracle using the declared scale domains.
Activation-checkpoint recomputation must reproduce the same gradients.

Measure forward/backward time and peak extra tensor memory with:

```bash
PYTHONPATH=. python examples/llm_qat/linear_attention/benchmark_prefill.py \
  --batch 1 --length 1024 --heads 4 --dim 64 --repeats 20 \
  --output prefill-overhead.json
```

The script warms each variant, interleaves trial order, and reports paired wall-time
ratios against the exact FLA path. Its JSON records the GPU, package versions, shape,
source hashes, and all samples. These are training-emulation costs; no native
low-precision inference speedup or model-quality recovery is implied.
