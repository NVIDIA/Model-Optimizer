# Quantization-Aware Training and Distillation for Linear Attention

This example fine-tunes Megatron-Core GDN or KDA attention parameters with
recurrent-state fake quantization using Megatron Bridge's training loop (QAT), or
its distillation loop with an unquantized teacher (QAD). Bridge handles optimization,
gradient accumulation, logging, and checkpoints. Training adapters support Megatron
`GatedDeltaNet` and `KimiDeltaAttention`; FLA supplies kernels, not model-layer adapters.
Runtime support includes token writes, ReplaySSM, KDA decay approximation, and
FP8 or INT8 state QDQ. The INT8 recipe enables Hadamard rotation by default.

## Run the example

Use the [Megatron Bridge environment](../../megatron_bridge/README.md#pre-requisites)
and install the example dependencies. Run commands from the repository root.

KDA requires a Megatron revision exporting
`megatron.core.ssm.gated_delta_net.KimiDeltaAttention`. It is available on
[Megatron's dev branch](https://github.com/NVIDIA/Megatron-LM/tree/ac100f773f9d8fac918ce3e64698be4ed84fa428);
Megatron-Core 0.19.2 does not include it. Older versions can still use GDN.
Retain `fla-core==0.5.1` for the kernel dependency. ModelOpt registers no FLA
model-layer adapters; Megatron or Bridge may still install the full
`flash-linear-attention` package as their dependency. Use mutually compatible
Bridge/Core revisions: Bridge's pinned Core revision may differ from the KDA
`dev` revision. Native Megatron KDA training does not require Bridge.

```bash
pip install -r examples/llm_qat/linear_attention/requirements.txt
torchrun --standalone --nproc-per-node=1 examples/llm_qat/linear_attention/train.py \
  --model /path/to/local-model \
  --train-data /path/to/tokenized/train_text_document \
  --output /path/to/megatron-qat-checkpoint \
  --recipe general/ptq/linear_attention_state_int8_dynamic \
  --train-steps 1 --length 128 --prefill-tokens 64
```

`--recipe` accepts a built-in recipe name or a custom YAML path. The shared
INT8 state recipe shown above is the default.

Add `--teacher-model /path/to/unquantized-model` to run QAD. The teacher and
student must share the tokenizer vocabulary and output vocabulary dimensions.
The teacher stays frozen and unquantized. QAT uses next-token cross-entropy;
QAD uses Bridge's logits distillation loss with the language-model loss disabled.
Both losses are masked to the suffix after `--prefill-tokens`.

For multiple GPUs, use the same `--tp_size`, `--pp_size`, and `--ep_size` options
as [the Megatron Bridge example](../../megatron_bridge/distill.py). Bridge builds
the process groups, schedules forward/backward, and manages the distributed optimizer.
For example, run QAD with two-way tensor parallelism:

```bash
torchrun --standalone --nproc-per-node=2 examples/llm_qat/linear_attention/train.py \
  --model /path/to/local-model \
  --teacher-model /path/to/unquantized-model \
  --train-data /path/to/tokenized/train_text_document \
  --output /path/to/megatron-qad-checkpoint \
  --tp_size 2 --pp_size 1 --ep_size 1 \
  --global-batch-size 2 --train-steps 1 --length 128 --prefill-tokens 64
```

Student and teacher use the same topology. Sequence parallelism is enabled with
TP greater than one. With TP/PP/EP all set to one, additional ranks use data
parallelism; the model weights remain replicated on each GPU. The global batch
size must be divisible by the data-parallel size (microbatch size is one).
Choose a topology supported by the model's Bridge/Core implementation. This example
currently requires linear-attention layers on every pipeline stage. Context
parallelism stays fixed at one; GDN/KDA state QAT does not yet support CP.

Use a local HF model/tokenizer snapshot that your installed Megatron Bridge can
convert to a Megatron model containing GDN or KDA layers, a tokenized Megatron
`.bin`/`.idx` dataset pair, and CUDA GPUs. `--train-data` is the shared filename
prefix without either extension; use the model's tokenizer when
[preparing the data](../../megatron_bridge/README.md#data-preparation).
Bridge's architecture/checkpoint conversion support is a separate requirement
from ModelOpt's layer adapter; an arbitrary
FLA model checkpoint cannot be loaded through this example. For a model already
built in Megatron, use the training-loop integration below. If the local model
requires custom Python code, explicitly pass `--trust-remote-code`.

The shared [INT8 recipe](../../../modelopt_recipes/general/ptq/linear_attention_state_int8_dynamic.yaml) enables GDN and
KDA state quantizers. It leaves the 64-token prefix state unquantized and applies
INT8 QDQ in a 32-value Hadamard basis during the decode suffix. Value dimensions
must be divisible by 32. Only linear-attention parameters are trained, using
Bridge's BF16 mixed precision and optimizer; other model parameters are frozen.
Every dense sequence has the same fixed prefill boundary. The example attaches
the phase context to local linear-attention layers on each rank and retains it
throughout training so backward recomputation sees that boundary.
Packed sequences and variable per-sequence boundaries require their own batch integration.

Bridge saves model weights, optimizer/scheduler state, and ModelOpt state under
`<output>/checkpoints`. Reusing `--output` resumes from its latest checkpoint;
increase `--train-steps` to the desired total step count and retain the same model,
teacher, topology, dataset, and prefill boundary. A resumed checkpoint supplies the saved
quantization policy. `--global-batch-size` controls accumulation with microbatch size one.
Export to a supported serving model separately. The current vLLM
state-only plugin accepts its own boundary-QDQ recipe; it does not implement
the Hadamard or replay training policies.

This example uses Bridge's existing training workflows with state quantization.
The Hugging Face
`QATTrainer`/`QADTrainer` example in [the parent directory](../README.md#using-qattrainer-and-qadtrainer)
targets a different training framework. This example does not measure model-quality
recovery or performance.

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

State quantizers start disabled. The shared state recipe enables `*gdn_state_quantizer` and `*kda_state_quantizer`
with signed narrow-range INT8, dynamic scales, and `axis=(0, 1)`. Use
`*gdn_state_quantizer` or `*kda_state_quantizer` to select just one layer type. For E4M3, set the quantizer config to
`{"num_bits": [4, 3], "type": "dynamic", "axis": [0, 1]}` and set
`decode.state_codec="tile"`.
Setting `prefill_state_qdq=True` alone does not enable a quantizer.

Start with the token-state recipe and select one of the schedules below **before**
calling `mtq.quantize`:

```python
import modelopt.torch.quantization as mtq
from modelopt.recipe import load_recipe

recipe = load_recipe("general/ptq/linear_attention_state_int8_dynamic").quantize.model_dump()
policy = recipe["linear_attention"][0]["cfg"]
decode = policy["decode"]
```

The complete recipe composes both INT8 state quantizers with the Hadamard
execution policy and uses the Torch decode implementation. Use the phase context
shown below for each forward/backward. Importing only
`configs/ptq/units/linear_attention_state_int8_dynamic` configures the quantizers
without selecting Hadamard.

### Why training needs prefill/decode boundaries

One training forward can simulate prompt prefill followed by recurrent decode.
The INT8 + Hadamard recipe applies different state quantization schedules to those
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

The table assumes the state quantizer is enabled. The INT8 recipe defaults to
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

To customize the CLI, copy the shared YAML recipe, edit its `quantize` settings,
and pass `--recipe /path/to/recipe.yaml`. `--prefill-tokens` sets the phase split; it
does not enable state quantization. The integration loop below applies the
configured `recipe` directly.

### Megatron GDN with chunk-only FLA kernels

For Megatron GDN's existing chunked FLA kernel path, enable the GDN state quantizer and omit the
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

### Per-row blockwise INT8 through TensorQuantizer

Use the standard quantizer settings to select one dynamic scale per key row and
16, 32, or 64 consecutive value channels. For example, replace the state quantizer
configuration in `gdn_recipe` above with:

```python
gdn_recipe["quant_cfg"][1]["cfg"] = {
    "num_bits": 8,
    "type": "dynamic",
    "block_sizes": {-1: 32},
    "unsigned": False,
    "narrow_range": True,
    "pass_through_bwd": True,
}
```

The logical state is `[N, H, K, V]`; `-1` denotes V even when a kernel stores the
state transposed. Omit `axis` when using `block_sizes`. The outer `type="dynamic"`
uses ordinary INT8 with dynamic amax; a nested `block_sizes["type"]="dynamic"`
selects a different specialized ModelOpt path and is not supported here.

The GDN fused kernel applies the quantizer's grouping inside each execution tile.
It can enlarge `state.block_v` to fit a complete quantization group; changing the
execution tile does not change the scale groups. The Torch GDN/KDA path calls the
registered `TensorQuantizer` at the same configured state-write boundaries.
For KDA, use `*kda_state_quantizer` and the materialized backend. For token decode
or replay, explicitly select `state_codec="tile"`; the existing Hadamard recipe
keeps its fixed codec and cannot be combined with this `block_sizes` setting.

Configurations without `block_sizes` retain their existing `[K, block_v]` scale
grouping, or their fixed Hadamard codec. The default INT8 + Hadamard recipe is
unchanged. This remains floating-storage fake QDQ for QAT/QAD.

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
# ids and shifted next-token labels have shape [2, 128].
# model is a Megatron model.
positions = torch.arange(128, device=ids.device).expand_as(ids)
with linear_attention_training_phase(model, [64, 96]):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        losses = model(
            input_ids=ids, position_ids=positions, attention_mask=None, labels=labels
        )
        mask = torch.arange(128, device=ids.device)[None, :] >= torch.tensor(
            [64, 96], device=ids.device
        )[:, None]
        loss = (losses * mask).sum() / mask.sum()
    loss.backward()
optimizer.step()
```

The shared recipe selects both GDN and KDA state quantizers. State quantization is dynamic, so the recipe uses
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
has no prefill GEMM QDQ or approximate inverse. Use the Megatron training forward
without a serving inference context. ModelOpt saves execution policies, while
per-batch prefix lengths must be supplied again during training. Distributed
decode training and model-quality recovery require separate qualification.
