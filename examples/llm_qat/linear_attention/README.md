# Quantization-Aware Training for Linear Attention

This example fine-tunes KDA attention parameters with recurrent-state fake
quantization, then saves a Hugging Face checkpoint with ModelOpt state. GDN/KDA
runtime support includes token writes, ReplaySSM, KDA decay approximation, and
FP8 or INT8 state QDQ. The INT8 recipes enable Hadamard rotation by default.

## Run the example

Install ModelOpt using the [QAT setup instructions](../README.md#quick-start),
then install the example dependencies. Run commands from the repository root.

```bash
pip install -r examples/llm_qat/linear_attention/requirements.txt
python examples/llm_qat/linear_attention/train.py \
  --model /path/to/local-kda-model \
  --train-data /path/to/train.parquet \
  --output /path/to/qat-checkpoint \
  --train-steps 1 --length 128 --prefill-tokens 64
```

Use a local model/tokenizer snapshot containing FLA `KimiDeltaAttention` layers,
a Parquet file with a `text` column, and a CUDA GPU. The example requires
`fla-core==0.5.1` and `flash-linear-attention==0.5.1`. If the local model requires
custom Python code, review it and explicitly pass `--trust-remote-code`.

The default [INT8 configuration](configs/kda_decode_state_int8.json) leaves the
64-token prefix state unquantized and applies INT8 QDQ in a 32-value Hadamard
basis during the decode suffix. Value dimensions must be divisible by 32.
Loss uses suffix labels. Only KDA attention parameters are trained, in FP32 under
BF16 autocast; other parameters are frozen in BF16. The output contains model
weights, tokenizer files, and ModelOpt quantizer/policy state. Call
`mto.enable_huggingface_checkpointing()` before reloading it with
`AutoModelForCausalLM.from_pretrained` to restore those quantizers and policies.
It is a floating fake-quantized training checkpoint, not a compressed serving model.
This short example does not measure model-quality recovery or performance.

## Enable state quantization

### What the execution policy controls

An **execution policy** is the layer's `LinearAttentionConfig`: the settings that
determine how the recurrence runs and when it applies quantization. Training
uses three separate inputs:

| Input | Purpose | Supplied through |
| --- | --- | --- |
| `TensorQuantizer` settings | Enable a quantization site and choose its numerical format, scales, and gradient behavior | Recipe `quant_cfg`, such as dynamic INT8 with identity STE |
| Execution policy | Choose token or replay mode, Hadamard rotation, handoff quantization, and the execution backend | Recipe `linear_attention` entries |
| Batch metadata | Specify where each sequence switches from prefill to decode | `linear_attention_training_phase(model, prefill_lengths)` |

The same INT8 quantizer settings can round the state after every token or round
a replay anchor every eight tokens. Those schedules produce different recurrent
states, so the policy must specify which computation training should emulate.

`LinearAttentionConfig` holds the overall backend, chunk size, and state settings.
Its optional `decode` field contains a `LinearAttentionDecodeConfig` for token or
replay mode, state codec, decay approximation, and replay settings. The decode
implementation uses Torch operations and autograd. This is one nested configuration: supplying a `decode`
dictionary in the recipe constructs the nested config automatically.

Each `linear_attention` entry uses `module_name` to select attention layers and
`cfg` to specify their policy. During `mtq.quantize`, ModelOpt stores that policy
as each matched layer's `linear_attention_config`. Quantizer settings and the
policy persist through ModelOpt save/restore; batch-specific prefill lengths
must be supplied again for each workload.

### Load a state recipe

State quantizers start disabled. The state recipe enables `*kda_state_quantizer`
with signed narrow-range INT8, dynamic scales, and `axis=(0, 1)`. Use
`*gdn_state_quantizer` for GDN. For E4M3, set the quantizer config to
`{"num_bits": [4, 3], "type": "dynamic", "axis": [0, 1]}` and set
`decode.state_codec="tile"`.
Setting `prefill_state_qdq=True` alone does not enable a quantizer.

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

For GDN, the complete recipe composes the INT8 quantizer unit with the Hadamard
execution policy:

```python
import modelopt.torch.quantization as mtq
from modelopt.recipe import load_recipe

gdn_recipe = load_recipe("general/ptq/gdn_state_int8_dynamic").quantize
mtq.quantize(gdn_model, gdn_recipe)
```

Use the phase context shown below for each forward/backward. Importing only
`configs/ptq/units/gdn_state_int8_dynamic` configures the quantizer without
selecting Hadamard. Both complete recipes use the Torch decode implementation.

### Why training needs prefill/decode boundaries

One training forward can simulate prompt prefill followed by recurrent decode.
The INT8 + Hadamard recipes apply different state quantization schedules to those
phases, so the workload must specify where each sequence switches to decode.
The training batch supplies all tokens; this simulates decode arithmetic without
running a text-generation loop. Total sequence length alone does not identify
the prompt portion.

For a 128-token sequence with `prefill_lengths=[96]`, the default token-state
recipe runs these steps:

1. **Prefill:** process tokens 0–95 with `prefill_state_qdq=False`, leaving state
   unquantized during the prefix.
2. **Handoff:** carry the resulting state into decode, rotate its value dimension
   into the Hadamard basis, and apply INT8 QDQ because `quantize_initial=True`.
3. **Decode:** process tokens 96–127 recurrently, applying INT8 QDQ after every
   state update. Later tokens consume the rounded state; outputs are transformed
   back to the original value basis.

The boundary is independent of `chunk_size=64`: this 96-token prefix contains a
full 64-token chunk and a partial 32-token chunk. The phase switches after token
95, not after each chunk.

Two 128-token sequences can use `prefill_lengths=[64, 96]` in the same batch:
their decode suffixes then contain 64 and 32 tokens, respectively. The next batch
can have different lengths while reusing the same quantization recipe. This is
why the execution policy is saved in `linear_attention_config`, while the lengths
are supplied per batch through `linear_attention_training_phase`. They are not
saved as part of the model's quantization policy. The context selects numerical
phases; the caller still supplies training labels and any loss masking.

The combined prefill/decode path requires explicit lengths, including `[0]` for
decode only or `[T]` for an all-prefix sequence of length `T`. The GDN chunk-only
FLA path described below applies its chunk schedule throughout and needs no
phase context. See the tables below for the corresponding quantization settings.

### Choose the prefill and decode boundaries

The table assumes the state quantizer is enabled. The INT8 recipes default to
`"int8_hadamard32"`; prefix state QDQ requires explicitly selecting `"tile"`.
Prefix lengths are supplied separately through `linear_attention_training_phase`.

| Desired state QDQ | `decode.state_codec` | `decode.mode` | `decode.prefill_state_qdq` | Where rounding occurs |
| --- | --- | --- | --- | --- |
| Token decode only (default) | `"int8_hadamard32"` | `"token"` | `False` | At the first nonempty decode handoff and after every suffix token. |
| Prefill and token decode | `"tile"` | `"token"` | `True` | At prefix initialization, each prefix chunk write, decode handoff, and every suffix token. |
| Replay anchors only | `"int8_hadamard32"` | `"replay"` | `False` | At decode handoff and each replay-window refresh. |
| Prefill and replay anchors | `"tile"` | `"replay"` | `True` | At prefix initialization and chunk writes, then decode handoff and replay-window refreshes. |

To enable **prefill and token decode**:

```python
decode.update(mode="token", replay=None, state_codec="tile", prefill_state_qdq=True)
```

For **token decode only**, keep the supplied INT8 recipe's defaults:
`state_codec="int8_hadamard32"` and `prefill_state_qdq=False`.
Prefix state remains unquantized until it enters the decode path.

To enable **ReplaySSM anchor quantization** with an eight-token window:

```python
decode.update(
    mode="replay",
    state_codec="int8_hadamard32",
    prefill_state_qdq=False,
    replay={"window": 8, "factor_qdq": False, "encoding": "once"},
)
```

Set `state_codec="tile"` and `prefill_state_qdq=True` to add prefix state QDQ
to this replay configuration, using unrotated INT8 for both phases.
`factor_qdq=False` above isolates state/anchor quantization. Set it to `True` to
also quantize buffered keys and updates to FP8.

`decode.quantize_initial=True` is the default: it quantizes the incoming state
once at the first nonempty decode handoff. Set it to `False` to skip that initial
rounding while keeping later token writes or anchor refreshes quantized. This
setting does not disable prefix state QDQ. With both phases enabled, prefix-final
rounding and decode-handoff rounding are separate configured events.

`chunk_size=64` counts **tokens per prefill chunk**. `state.block_v=64` counts
**value channels per execution tile**. The tile codec shares a scale across all
key channels and this value tile; Hadamard uses one scale per key channel and
32 values. A replay `window=8` counts **suffix tokens between anchor refreshes**.

### Run only the desired phase

For a batch containing one sequence of `T` tokens, choose the context lengths as
follows; for larger batches, provide one length per sequence:

| Workload | Context argument | Required setting |
| --- | --- | --- |
| Prefill followed by decode | `[64]`, with `T > 64` | Select either prefix setting above. |
| Decode only | `[0]` | State quantizer enabled; token or replay policy. |
| Prefill only | `[T]` | `state_codec="tile"`, `prefill_state_qdq=True`; the decode suffix is empty. |

An empty suffix creates no decode quantization event. The combined interface has
one state quantizer per layer: it does not offer a switch to quantize the prefix
while leaving a **nonempty** decode suffix unquantized. Disable the state quantizer
to disable both state and anchor QDQ; replay factor QDQ has its own toggle.

The `int8_hadamard32` codec applies to decode token writes or replay anchors only.
It requires `prefill_state_qdq=False`; use the tile codec to quantize prefix state.

Save a modified recipe as JSON and pass it to `train.py --quant-config`. `--prefill-tokens` sets the phase split; it does not enable state
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

This explicitly selects unrotated tile QDQ. It rounds the initial state and each
64-token chunk's final state, including a partial final chunk. It needs no phase
context and leaves W/projection quantizers
disabled. KDA uses the materialized backend and can use the all-prefix context
shown above.

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

# Two 128-token sequences: 64 and 96 prefill tokens, respectively.
# ids and labels have shape [2, 128] and are on the model's device.
with linear_attention_training_phase(model, [64, 96]):
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

## Replay, decay, and Hadamard options

Token mode rounds each recurrent-state write. Replay mode retains an anchor and
ordered key/update factors, rounding the anchor at each `replay.window` refresh.
`factor_qdq` independently enables FP8 QDQ for those factors. `readout="working"`
reads the state before its write quantization; `"stored"` reads it afterward.

For KDA decay approximation, set `decode["decay_log_step"] = 1 / 256` before
conversion. It rounds suffix log retention before exponentiation with identity
STE gradients. Prefix decay remains exact.

The default INT8 codec applies a 32-point orthonormal Hadamard transform along
the value axis, quantizes token states or replay anchors in that basis, and
transforms outputs back. Value dimensions must be divisible by 32; `state.block_v` must be 32, 64,
or 128. Scales group one key channel and 32 values, independent of execution tile
width. INT8 codes use half-away-from-zero rounding with FP16 stored scales;
the optional tile codec instead uses nearest-even rounding and FP32 scales.

## Training boundaries

The Torch implementation uses autograd through initial states, chunk handoff,
token writes, and replay refreshes. The decode recurrence runs token by token in
Python, so long training suffixes can be slow.

Prefill prefixes use exact chunk algebra with optional state QDQ. This example
has no prefill GEMM QDQ or approximate inverse. FLA KDA requires `use_cache=False`;
serving cache objects are rejected. ModelOpt saves execution policies, while
per-batch prefix lengths must be supplied again during training. Distributed
decode training and model-quality recovery require separate qualification.
