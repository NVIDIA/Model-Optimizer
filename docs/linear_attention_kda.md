# KDA prefill arithmetic for QAT

ModelOpt recognizes FLA 0.5.1 `KimiDeltaAttention` layers during conversion and adds
`kda_state_quantizer`, `kda_w_quantizer`, and the eight `linear_attn_sites` operand
pairs. These handles start disabled. The existing projection recipes leave them
disabled; enable them explicitly with `backend="matmul"`.

The adapter reuses the layer's projections, convolution, normalization, and gate
parameters. Its numerical path uses the materialized differentiable prefill
backend. It does not patch process-wide FLA functions. When all numerical policies
are off, the original FLA forward runs unchanged.

## KDA site map and decay stability

For a chunk, let `G[i,d]` be the cumulative natural-log retention for key channel
`d`. KDA's interaction terms contain `exp(G[i,d] - G[j,d])`. The implementation
computes that causal difference directly. It never constructs `exp(-G[j,d])`;
strong forgetting can underflow a retention to zero without introducing the
infinite inverse factor of an unstable factorization. Future positions are
masked before the exponential and zeroed before operand quantization.

The shared prefill sites have the following KDA operands. Right operands below
are the row vectors presented to the quantizer before transposition.

| Site | Left operand | Right operand |
| --- | --- | --- |
| `key_interaction` | `beta[i] * k[i]` | `k[j] * exp(G[i] - G[j])`, `j <= i` |
| `wy_value` | exact triangular inverse | transposed beta-scaled values |
| `wy_key` | exact triangular inverse | transposed beta-scaled, prefix-decayed keys |
| `state_read` | WY keys, via `kda_w_quantizer` | transposed incoming state |
| `state_update` | transposed end-decayed keys | transposed corrected values |
| `output_state` | scaled, prefix-decayed queries | transposed incoming state |
| `output_score` | scaled `q[i]` | `k[j] * exp(G[i] - G[j])`, `j <= i` |
| `output_value` | causal local scores | transposed corrected values |

For the two interaction sites, query rows are grouped in fixed tiles of eight
and folded into the batch dimension. Quantizer inputs remain four-dimensional:
`[chunk_or_sequence_times_query_rows, head, row, reduction]`. FP8 scales are
per row. NVFP4 uses block-16 E2M1 values and E4M3 scales with one dynamic tensor
amax per invocation; the query tile of eight is therefore part of its scale
contract. Other sites use the [GDN prefill scale domains](linear_attention_prefill.md).

Chunk size is 64. Working arithmetic is FP32, including inside an outer BF16
training-autocast context; double inputs use FP64. Output returns to the query
dtype. Select `torch.set_float32_matmul_precision("highest")` to exclude TF32.
Operand and arithmetic rounding use identity STE; state carry is differentiable.
Exact triangular solve is the only supported solve in this delivery.

Raw gate activation follows the loaded FLA model: without `lower_bound`, it is
`-exp(A_log) * softplus(raw_gate + dt_bias)`; with a lower bound, it is
`lower_bound * sigmoid(exp(A_log) * (raw_gate + dt_bias))`. The latter is the
model's bounded gate formula, not a new clamp. No additional decay approximation
is enabled by this integration. The materialized path does not emulate the
optional FLA `safe_gate` TensorCore rounding. Numerical parity checks use FLA's
general computation (`safe_gate=False`) for both gate formulas.

## Enable KDA operand QDQ

```python
import modelopt.torch.quantization as mtq

model = mtq.quantize(model, {
    "quant_cfg": [
        {"quantizer_name": "*", "enable": False},
        {"quantizer_name": "*linear_attn_sites.*",
         "cfg": {"num_bits": (4, 3), "type": "dynamic", "axis": (0, 1, 2)}},
        {"quantizer_name": "*kda_w_quantizer",
         "cfg": {"num_bits": (4, 3), "type": "dynamic", "axis": (0, 1, 2)}},
    ],
    "algorithm": None,
    "linear_attention": [{
        "module_name": "model.layers.*.self_attn",
        "cfg": {"backend": "matmul"},
    }],
})
```

To emulate stored-state writes, enable `*kda_state_quantizer` with dynamic E4M3
or signed narrow-range INT8 and `axis=(0,1)`; see the
[state configuration](linear_attention_qat.md#configure-training). The policy's `state.block_v` specifies the value-column tile;
initial state and every chunk write are rounded. This is distinct from temporary
state-read operand QDQ. State-V-first layout, grouped value heads, packed tails,
and gradients through the initial state are supported.

`flash-linear-attention==0.5.1` and `fla-core==0.5.1` are required for the layer
adapter. Importing ModelOpt itself does not require either package. Qualification
covers a single-GPU FLA model. CP, FLA-specific intermediate/recompute flags, and
quantized recurrent inference are rejected. In evaluation mode FLA selects its
recurrent path for lengths at most 64; use longer prefill sequences for this
integration. For token-state training and decode/replay policies, see
[decode-aware QAT](linear_attention_decode.md).

## Reproducible model study

The [study example](../examples/llm_qat/linear_attention/README.md) uses a public,
pinned Arcee KDA checkpoint and WikiText-2 splits. It reports pre/post-training
held-out NLL and perplexity with per-block results, hashes, seeds, and fixed token
budgets. A successful training step verifies integration; it is not evidence of
model-quality recovery. The materialized backend's time and memory must be
measured separately from fused FLA.
