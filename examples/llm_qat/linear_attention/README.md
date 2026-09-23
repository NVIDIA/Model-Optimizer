# Quantization-Aware Training for Linear Attention

This example fine-tunes linear-attention models with differentiable recurrent-state
fake quantization. It supports GDN/KDA token-state writes, KDA decay approximation,
and ReplaySSM, including INT8 state with optional Hadamard rotation. The included
model-level training script uses FLA `KimiDeltaAttention` layers.

## Prerequisites

Install ModelOpt from this checkout using the [QAT setup instructions](../README.md#quick-start),
then install the example dependencies. Run the commands below from the repository root.

```bash
pip install -r examples/llm_qat/linear_attention/requirements.txt
```

The example requires a CUDA GPU, `fla-core==0.5.1`, and
`flash-linear-attention==0.5.1`. Stage a local model/tokenizer snapshot and separate
training and evaluation Parquet files with a `text` column. The pinned KDA model
and WikiText revisions are recorded in [decode_study_plan.json](decode_study_plan.json);
the script defaults to those revision identifiers, so the local inputs must match.

## Run a QAT example

The following command runs one training step with INT8 recurrent-state QDQ.
Each sequence has 64 exact prefix tokens; the remaining tokens use the decode
policy. This recipe enables state QDQ only in the decode suffix; prefix state
QDQ is disabled by default. Loss is computed on suffix labels.

```bash
python examples/llm_qat/linear_attention/train.py \
  --model /path/to/model-snapshot \
  --train-data /path/to/train.parquet \
  --eval-data /path/to/validation.parquet \
  --quant-config examples/llm_qat/linear_attention/configs/kda_decode_state_int8.json \
  --prefill-tokens 64 \
  --length 128 \
  --train-steps 1 \
  --eval-blocks 4 \
  --output tmp/linear_attention/state_int8.json
```

Add `--trust-remote-code` if the selected checkpoint requires its custom model code.
The script trains attention parameters with FP32 AdamW, BF16 autocast, and activation
checkpointing. Other model parameters remain frozen. It writes pre/post-training
NLL, gradient norms, timings, and input/source hashes to JSON; it does not save a
trained checkpoint. This short run checks training integration. Use the study plan
for matched quality comparisons before making accuracy-recovery claims.

### Select a recipe

Pass one of the following files to `--quant-config`:

| Recipe | Behavior |
| --- | --- |
| [kda_decode_exact.json](configs/kda_decode_exact.json) | Exact recurrence control with state QDQ disabled. |
| [kda_decode_state_int8.json](configs/kda_decode_state_int8.json) | INT8 state QDQ after every token. |
| [kda_decode_state_fp8.json](configs/kda_decode_state_fp8.json) | FP8 E4M3 state QDQ after every token. |
| [kda_decode_replay_int8.json](configs/kda_decode_replay_int8.json) | Replay with INT8 anchors and separate FP8 factor QDQ. |
| [kda_decode_replay_fp8.json](configs/kda_decode_replay_fp8.json) | Replay with FP8 anchors and factors. |
| [kda_decode_replay_int8_hadamard.json](configs/kda_decode_replay_int8_hadamard.json) | Replay with rotated INT8 anchors and factor QDQ disabled. |
| [kda_decode_decay_256.json](configs/kda_decode_decay_256.json), [1024](configs/kda_decode_decay_1024.json), [4096](configs/kda_decode_decay_4096.json) | FP8 state QDQ with different log-retention rounding grids. |

For an exact control, repeat the training command with `kda_decode_exact.json` and
`--output tmp/linear_attention/exact.json`, keeping model, data, seed, phase lengths,
and training settings identical. Compare the resulting records with:

```bash
python examples/llm_qat/linear_attention/compare_quality.py \
  --control tmp/linear_attention/exact.json \
  --candidate tmp/linear_attention/state_int8.json \
  --output tmp/linear_attention/comparison.json
```

The comparison reports paired block-NLL differences and bootstrap bounds.
INT8 and Hadamard model-quality qualification remains pending.

## Enable state quantization

State quantizers start disabled. The state recipe enables `*kda_state_quantizer`
with signed narrow-range INT8, dynamic scales, and `axis=(0, 1)`. Use
`*gdn_state_quantizer` for GDN. Select the corresponding FP8 recipe for E4M3.
The quantizer selects the format; the execution policy selects when state is
rounded. Setting `prefill_state_qdq=True` alone does not enable a quantizer.

Start with the token-state recipe and select one of the schedules below **before**
calling `mtq.quantize`:

```python
import json
from pathlib import Path

recipe = json.loads(
    Path("examples/llm_qat/linear_attention/configs/kda_decode_state_int8.json").read_text()
)
policy = recipe["linear_attention"][0]["cfg"]
decode = policy["decode"]
```

### Choose the prefill and decode boundaries

The table assumes the state quantizer is enabled and the default `"tile"` codec
is selected. Each row describes one configuration; prefix lengths are supplied
separately through `linear_attention_training_phase`.

| Desired state QDQ | `decode.mode` | `decode.prefill_state_qdq` | Where rounding occurs |
| --- | --- | --- | --- |
| Token decode only | `"token"` | `False` (default) | At the first nonempty decode handoff and after every suffix token. |
| Prefill and token decode | `"token"` | `True` | At prefix initialization, each prefix chunk write, decode handoff, and every suffix token. |
| Replay anchors only | `"replay"` | `False` | At decode handoff and each replay-window refresh. |
| Prefill and replay anchors | `"replay"` | `True` | At prefix initialization and chunk writes, then decode handoff and replay-window refreshes. |

To enable **prefill and token decode**:

```python
decode.update(mode="token", replay=None, prefill_state_qdq=True)
```

Set `prefill_state_qdq=False` for **token decode only**, which is what the supplied
`kda_decode_state_int8.json` and `kda_decode_state_fp8.json` recipes select.
Prefix state remains unquantized until it enters the decode path.

To enable **ReplaySSM anchor quantization** with an eight-token window:

```python
decode.update(
    mode="replay",
    prefill_state_qdq=False,
    replay={"window": 8, "factor_qdq": False, "encoding": "once"},
)
```

Set `prefill_state_qdq=True` to add prefix state QDQ to this replay configuration.
`factor_qdq=False` above isolates state/anchor quantization. Set it to `True` to
also quantize buffered keys and updates to FP8; the supplied replay recipes enable
that separate factor quantization unless stated otherwise.

`decode.quantize_initial=True` is the default: it quantizes the incoming state
once at the first nonempty decode handoff. Set it to `False` to skip that initial
rounding while keeping later token writes or anchor refreshes quantized. This
setting does not disable prefix state QDQ. With both phases enabled, prefix-final
rounding and decode-handoff rounding are separate configured events.

`chunk_size=64` counts **tokens per prefill chunk**. `state.block_v=64` counts
**value channels per state scale tile**. A replay `window=8` counts **suffix tokens
between anchor refreshes**. These settings have different purposes.

### Run only the desired phase

For a batch containing one sequence of `T` tokens, choose the context lengths as
follows; for larger batches, provide one length per sequence:

| Workload | Context argument | Required setting |
| --- | --- | --- |
| Prefill followed by decode | `[64]`, with `T > 64` | Select either prefix setting above. |
| Decode only | `[0]` | State quantizer enabled; token or replay policy. |
| Prefill only | `[T]` | `prefill_state_qdq=True`; the decode suffix is empty. |

An empty suffix creates no decode quantization event. The combined interface has
one state quantizer per layer: it does not offer a switch to quantize the prefix
while leaving a **nonempty** decode suffix unquantized. Disable the state quantizer
to disable both state and anchor QDQ; replay factor QDQ has its own toggle.

The `int8_hadamard32` codec applies to decode token writes or replay anchors only.
It requires `prefill_state_qdq=False`; use the tile codec to quantize prefix state.

To use a modified recipe with `train.py`, save it as JSON and pass that file to
`--quant-config`. `--prefill-tokens` sets the phase split; it does not enable state
quantization. The integration loop below applies the configured `recipe` directly.

### GDN chunk-only FLA training

For GDN's existing chunked FLA path, enable the GDN state quantizer and omit the
`linear_attention` execution-policy entry:

```python
gdn_recipe = {
    "quant_cfg": [
        {"quantizer_name": "*", "enable": False},
        {
            "quantizer_name": "*gdn_state_quantizer",
            "cfg": recipe["quant_cfg"][1]["cfg"],
        },
    ],
    "algorithm": None,
}
# Apply mtq.quantize(gdn_model, gdn_recipe), then use normal forward/backward.
```

This rounds the initial state and each 64-token chunk's final state, including a
partial final chunk. It needs no phase context and leaves W/projection quantizers
disabled. KDA uses the materialized backend and can use the all-prefix context
shown above.

For pure-prefill GDN/KDA training with the materialized backend, use
`backend="matmul"` with no `decode` policy and enable the state quantizer.
That path also rounds initial and chunk-final state and needs no phase context;
`decode.prefill_state_qdq` applies only when a decode policy is present.

## Integrate with a training loop

Apply the configured `recipe` above with `mtq.quantize`, then supply one prefix length per sequence.
Keep the phase context active through backward so activation-checkpoint
recomputation uses the same prefix/decode split.

```python
import torch

import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.linear_attention import linear_attention_training_phase

mtq.quantize(model, recipe)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
model.train()
optimizer.zero_grad(set_to_none=True)

# One sequence with 64 prefix tokens; ids and labels are on the model's device.
with linear_attention_training_phase(model, [64]):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = model(input_ids=ids, labels=labels, use_cache=False).loss
    loss.backward()
optimizer.step()
```

For GDN, select `*gdn_state_quantizer` in the recipe instead of
`*kda_state_quantizer`. State quantization is dynamic, so these recipes use
`algorithm=None` without a calibration pass. Replay factors have their own FP8
toggle.

Policies persist through ModelOpt save/restore. Per-batch prefix lengths are
runtime metadata and must be supplied for each workload. The context restores
previous lengths on exit and supports nesting. Concurrent forwards on the same
model instance with different phase contexts are unsupported.

## Numerical contract

For activated input tensors, a token computes `D = exp(g)`, `R = D * S`, `u = beta * (v - k^T R)`, and `W = R + k u^T`. GDN uses one log retention per head; KDA uses one per key channel. The output reads either the working state `W` or the stored state, according to `readout`. All operations use FP32 working values (FP64 is supported by the Torch reference).

- **Token mode:** fake-quantize `W` at every state write when the module state quantizer is enabled. The next token consumes that stored value.
- **Decay approximation:** an optional positive `decay_log_step` rounds log retention to the nearest grid point before exponentiation. Its derivative is an identity straight-through estimator (STE); no hidden clamping is applied. Prefix gates retain the prefill policy.
- **Replay mode:** maintain an anchor and ordered encoded rank-one updates. Encode each key per head and each update per value block, with optional FP8 E4M3 QDQ. Compute a new residual from the reconstructed current trajectory and the encoded key. At `window` updates, encode the current state as a new anchor and clear the buffer. Anchor QDQ follows the state quantizer toggle; factor QDQ has its own toggle.
- **Initial state:** `quantize_initial` applies once at the first nonempty decode handoff. Continuation and empty calls do not create extra writes. Prefix state QDQ is independently selected by `prefill_state_qdq`.

State scale grouping is one dynamic scale for each `[Dk, block_v]` head tile. Update scales span one value block; key scales span the complete key vector. Zero tiles use scale one. Codec scales are detached, and rounded values have identity STE gradients. State codecs support E4M3 nearest-even rounding (including subnormals) and signed
symmetric INT8. INT8 uses FP32 `amax / 127` scales, zero point zero, codes in
`[-127, 127]`, nearest-even rounding, and saturation. Replay key/update factors
retain their independent E4M3 or identity format. Key reductions use a fixed pairwise FP32 tree; this avoids changing recurrence rounding merely by changing the reduction schedule.

The public numerical APIs `recurrent_decode_reference` and `recurrent_decode` accept `state_format="fp8_e4m3"` (default) or `"int8"` and one aligned `[T,H,D]` sequence and return outputs plus a `LinearAttentionCarry`. It retains anchor values/scales, buffered key/update values/scales, log retentions, position, cursor, and a policy signature. Returned tensors preserve gradients through prefix handoff, refreshes, and continuation. Restart with a new carry to reset; incompatible policy, state format, or state shape is rejected.

The Torch reference can explicitly re-encode stored entries using `encoding="reencode"`. It re-encodes the already stored values and does not regenerate factors from a teacher trajectory. The fused implementation supports encode-once replay. Carrying its reconstructed state incrementally applies the same ordered transitions as replaying the fixed entries; it does not measure serving-time reconstruction cost.

## INT8 state with Hadamard rotation

Set `decode.state_codec="int8_hadamard32"` with an INT8 state quantizer to enable
32-point orthonormal Sylvester Hadamard transforms along the **value** dimension.
This codec supports both token writes and replay anchors. The default `"tile"`
codec retains the existing FP8/INT8 behavior above.

```python
# Execution policy; pair with the INT8 quantizer attributes in the example above.
policy = {
    "backend": "matmul",
    "state": {"block_v": 64},
    "decode": {
        "mode": "replay",
        "implementation": "triton",
        "state_codec": "int8_hadamard32",
        "readout": "working",
        "replay": {"window": 8, "factor_qdq": False, "encoding": "once"},
    },
}
```

For key-first state `S`, the internal state is `S @ H`, with a block-diagonal
Hadamard matrix `H`. Inputs `v` and replay update vectors use the same basis;
queries, keys, and scalar/channel decay gates keep their original coordinates.
Outputs are transformed back. The first nonempty decode call transforms the
incoming state once; continuation retains the rotated anchor and updates without
another rotation or initial-state quantization. `carry.value_basis` records the
basis. `carry.reconstruct()` returns the **original** basis; passing
`original_basis=False` exposes the internal basis.

The state codec uses one scale per **key channel and 32 contiguous values**, even
when the execution tile `state.block_v` is 64 or 128. For each group, compute
`scale=max(amax/127, 6e-8)` in FP32, divide by that scale, round half ties away from
zero, and saturate codes to `[-127,127]`. Store the scale in FP16 and reconstruct
with that FP16 scale. Scale metadata has shape `[H,Dk,Dv/32]`. Rounding has
identity STE and scales are detached. Basis transforms use their actual linear
derivatives. Replay factor QDQ remains independently configurable; the example
disables it to isolate state/anchor quantization.

The Hadamard basis and INT8 scale/rounding conventions follow
[quantized-replayssm at 29c35508](https://github.com/mxinO/quantized-replayssm/blob/29c355086070d2c7973550d3670171af53433533/vllm/model_executor/layers/fla/ops/fused_recurrent_replayssm.py).
Model-quality evaluation of this ModelOpt implementation remains pending.

`Dv` must be divisible by 32, and `state.block_v` must be 32, 64, or 128. Prefix
computation remains in the original basis with the exact solve; this codec starts
at decode handoff and rejects `prefill_state_qdq=True`. An all-prefix or empty
sequence does not create a decode quantization event. The policy is saved and
restored with the model. A complete example is
[configs/kda_decode_replay_int8_hadamard.json](configs/kda_decode_replay_int8_hadamard.json).

## Training implementation and limits

The CUDA Triton implementation supports FP32 working inputs, key dimensions up to 128, scalar/channel gates, and value blocks 16/32/64/128 with masked tails. It saves a state every eight tokens and recomputes intermediate states in backward. This internal checkpoint interval does not change write cadence. Its custom backward supports first-order gradients; use the Torch reference for higher-order differentiation.

Direct prefill APIs accept `prefill_lengths`, grouped query/key heads, both state layouts, packed sequences, empty entries, and full/zero prefixes. Positive prefixes are evaluated together so prefill quantizer scale domains remain shared. FLA KDA layer integration requires 0.5.1 and `use_cache=False`; serving cache objects are rejected. The numerical carry API provides explicit continuation, but this change does not implement a serving cache or export format. Megatron GDN tensor parallelism is covered by dedicated tests; context parallelism remains rejected.

All encoded values remain floating tensors. This is training-time fake quantization and approximation, with no compressed storage or inference speed claim. Use [benchmark_decode.py](benchmark_decode.py) for matched-policy training overhead and `decode_study_plan.json` for the fixed model-quality protocol. The exact inverse remains selected in these studies; the earlier rejected Neumann candidate is not enabled.

See [the qualification study](decode_study.md) for the fixed
model/data protocol, quality comparisons, and measured training overhead.

## GDN prefill arithmetic for QAT

GDN training can expose all eight logical prefill matmuls through an explicit
`backend="matmul"` policy. Each operand uses ModelOpt `TensorQuantizer`; standard
PyTorch autograd propagates its identity straight-through gradient. The computation
batches chunk-local work and carries state between chunks without detaching it.
This backend materializes operands and the triangular inverse. Its training cost
must be measured separately from the fused FLA state/W path.

### Enable operand quantization

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

### Select arithmetic experiments independently

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

### Persistence and verification

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

## KDA prefill arithmetic for QAT

ModelOpt recognizes FLA 0.5.1 `KimiDeltaAttention` layers during conversion and adds
`kda_state_quantizer`, `kda_w_quantizer`, and the eight `linear_attn_sites` operand
pairs. These handles start disabled. The existing projection recipes leave them
disabled; enable them explicitly with `backend="matmul"`.

The adapter reuses the layer's projections, convolution, normalization, and gate
parameters. Its numerical path uses the materialized differentiable prefill
backend. It does not patch process-wide FLA functions. When all numerical policies
are off, the original FLA forward runs unchanged.

### KDA site map and decay stability

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
contract. Other sites use the [GDN prefill scale domains](#gdn-prefill-arithmetic-for-qat).

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

### Enable KDA operand QDQ

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
or signed narrow-range INT8 and `axis=(0,1)`. The policy's `state.block_v` specifies
the value-column tile; initial state and every chunk write are rounded. This is distinct from temporary
state-read operand QDQ. State-V-first layout, grouped value heads, packed tails,
and gradients through the initial state are supported.

`flash-linear-attention==0.5.1` and `fla-core==0.5.1` are required for the layer
adapter. Importing ModelOpt itself does not require either package. Qualification
covers a single-GPU FLA model. CP, FLA-specific intermediate/recompute flags, and
quantized recurrent inference are rejected. In evaluation mode FLA selects its
recurrent path for lengths at most 64; use longer prefill sequences for this
integration. For token-state training and decode/replay policies, see
[decode-aware QAT](#numerical-contract).

### Reproducible model study

The [study example](#prefill-model-study) uses a public,
pinned Arcee KDA checkpoint and WikiText-2 splits. It reports pre/post-training
held-out NLL and perplexity with per-block results, hashes, seeds, and fixed token
budgets. A successful training step verifies integration; it is not evidence of
model-quality recovery. The materialized backend's time and memory must be
measured separately from fused FLA.

## Prefill model study

The initial model is [arcee-ai/AFM-4.5B-Base-KDA-Only](https://huggingface.co/arcee-ai/AFM-4.5B-Base-KDA-Only)
at revision `01ad2e06ee4f1214193c17b69e09105a9b257e80`. Its implementation uses
FLA's `KimiDeltaAttention` directly. The dataset is
[Salesforce/wikitext](https://huggingface.co/datasets/Salesforce/wikitext), configuration
`wikitext-2-raw-v1`, revision `b08601e04326c79dfdd32d625aee71d232d685c3`.

Stage those exact revisions with model
weights, tokenizer, audited model code, and the three Parquet splits. Supply local
paths to the runner; it does not fetch weights or data. The checkpoint's custom
model code requires an explicit `--trust-remote-code` opt-in. Review its pinned
Python files before enabling that flag.

```bash
PYTHONPATH=. python examples/llm_qat/linear_attention/train.py \
  --model /path/to/arcee-kda \
  --train-data /path/to/wikitext-2-raw-v1/train-00000-of-00001.parquet \
  --eval-data /path/to/wikitext-2-raw-v1/validation-00000-of-00001.parquet \
  --eval-split validation --trust-remote-code \
  --quant-config examples/llm_qat/linear_attention/configs/kda_prefill_fp8.json \
  --length 128 --train-steps 1 --eval-blocks 4 --seed 2026 \
  --output fp8-smoke.json
```

The one-step command is an integration smoke test. For matched studies, run the
same workload from the same original checkpoint with each numerical configuration:

| Trial | Configuration |
| --- | --- |
| Exact fused FLA control | Omit `--quant-config` |
| Exact materialized control | `configs/kda_exact_matmul.json` |
| FP8 prefill operands | `configs/kda_prefill_fp8.json` |
| NVFP4 prefill operands | `configs/kda_prefill_nvfp4.json` |

The exact materialized control explicitly rounds the already-FP32 `output_add`
to FP32 to select the emulation path. It introduces no lower-precision rounding.
First establish agreement with the fused control before interpreting a candidate's
quality. Candidate `before` measurements isolate its untrained numerical change;
compare trained candidates against an equally trained exact control. Choose
settings using validation data, then run the fixed selected settings on the test
split. Never choose approximation degrees or training hyperparameters on test data.

### Protocol

- Join Parquet text rows with two newlines and tokenize without added special
  tokens. Use the first requested contiguous blocks, each containing `length+1`
  tokens and predicting `length` tokens. Blocks do not overlap; cache/state reset
  for each block. This protocol is for paired comparisons, not a claim of the
  checkpoint's published WikiText perplexity.
- Shuffle the training blocks with the recorded seed. Use the same seed, block
  count, token hashes, length, and learning rate across all trials.
- Train KDA attention parameters only. Keep them and AdamW moments in FP32, use
  BF16 autocast and activation checkpointing, and freeze other model parameters
  in BF16. Preserve the checkpoint's FP32 gate parameters when loading. AdamW has
  zero weight decay, default learning rate `1e-5`, and gradient clipping at 1.
- Record per-block NLL, predicted token counts, source/data/token hashes, package
  versions, training losses, finite-gradient checks, optimizer updates, step times,
  and peak allocated training memory. The output contains no generated answers or
  source document text.

A short matched study can screen numerical sensitivity. It does not establish
broad downstream-task quality, long-context equivalence, or native serving speed.

## Measure prefix and decode training cost

```bash
python examples/llm_qat/linear_attention/benchmark_decode.py \
  --attention kda --length 257 --prefill 64 --dim 128 --output decode-cost.json
```

The benchmark checks outputs, final states, and input gradients against the
matching Torch numerical policy before measuring forward/backward time and peak
memory. See the [recorded decode study](decode_study.md) for historical results.
The comparison measures training emulation, not inference or compressed-cache cost.

## State-only serving with vLLM

See the [vLLM example](../../vllm_serve/README.md#linear-attention-state-quantization)
for incoming-state QDQ before native prefill/decode. Its invocation boundaries
differ from the token-write and ReplaySSM policies used during training.
