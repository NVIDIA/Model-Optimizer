# Quantization-Aware Training for Linear Attention

This example fine-tunes KDA attention parameters with recurrent-state fake
quantization, then saves a Hugging Face checkpoint with ModelOpt state. GDN/KDA
runtime support includes token writes, ReplaySSM, KDA decay approximation, and
FP8 or INT8 state QDQ with optional Hadamard rotation.

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
64-token prefix state unquantized and rounds state during the decode suffix.
Loss uses suffix labels. Only KDA attention parameters are trained, in FP32 under
BF16 autocast; other parameters are frozen in BF16. The output contains model
weights, tokenizer files, and ModelOpt quantizer/policy state. Call
`mto.enable_huggingface_checkpointing()` before reloading it with
`AutoModelForCausalLM.from_pretrained` to restore those quantizers and policies.
It is a floating fake-quantized training checkpoint, not a compressed serving model.
This short example does not measure model-quality recovery or performance.

## Enable state quantization

State quantizers start disabled. The state recipe enables `*kda_state_quantizer`
with signed narrow-range INT8, dynamic scales, and `axis=(0, 1)`. Use
`*gdn_state_quantizer` for GDN. For E4M3, set the quantizer config to `{"num_bits": [4, 3], "type": "dynamic", "axis": [0, 1]}`.
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

Set `prefill_state_qdq=False` for **token decode only**, which is the supplied INT8 recipe's default.
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
also quantize buffered keys and updates to FP8.

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

This rounds the initial state and each 64-token chunk's final state, including a
partial final chunk. It needs no phase context and leaves W/projection quantizers
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

## Replay, decay, and Hadamard options

Token mode rounds each recurrent-state write. Replay mode retains an anchor and
ordered key/update factors, rounding the anchor at each `replay.window` refresh.
`factor_qdq` independently enables FP8 QDQ for those factors. `readout="working"`
reads the state before its write quantization; `"stored"` reads it afterward.

For KDA decay approximation, set `decode["decay_log_step"] = 1 / 256` before
conversion. It rounds suffix log retention before exponentiation with identity
STE gradients. Prefix decay remains exact.

For INT8 Hadamard replay, modify the same recipe:

```python
decode.update(
    mode="replay", state_codec="int8_hadamard32", readout="working",
    prefill_state_qdq=False,
    replay={"window": 8, "factor_qdq": False, "encoding": "once"},
)
```

The codec applies a 32-point orthonormal Hadamard transform along the value axis,
quantizes token states or replay anchors in that basis, and transforms outputs
back. Value dimensions must be divisible by 32; `state.block_v` must be 32, 64,
or 128. Scales group one key channel and 32 values, independent of execution tile
width. INT8 codes use half-away-from-zero rounding with FP16 stored scales;
the default tile codec instead uses nearest-even rounding and FP32 scales.

## Training boundaries

The Torch and Triton implementations support first-order QAT gradients through
initial states, chunk handoff, token writes, and replay refreshes. Triton supports
key dimensions up to 128 and value blocks 16/32/64/128, with checkpointed backward.
Use the Torch implementation for higher-order differentiation.

Prefill prefixes use exact chunk algebra with optional state QDQ. This example
has no prefill GEMM QDQ or approximate inverse. FLA KDA requires `use_cache=False`;
serving cache objects are rejected. ModelOpt saves execution policies, while
per-batch prefix lengths must be supplied again during training. Distributed
decode training and model-quality recovery require separate qualification.
