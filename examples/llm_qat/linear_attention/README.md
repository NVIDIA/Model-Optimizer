# Quantization-Aware Training and Distillation for Linear Attention

This example fine-tunes Megatron-Core GDN or KDA attention parameters with
recurrent-state fake quantization using Megatron Bridge's training loop (QAT), or
its distillation loop with an unquantized teacher (QAD). Bridge handles optimization,
gradient accumulation, logging, and checkpoints. Training adapters support Megatron
`GatedDeltaNet` and `KimiDeltaAttention`; FLA supplies kernels, not model-layer adapters.
State QAT uses serving forward kernels for a chunked prefix and a recurrent suffix,
with a differentiable Torch adjoint. The recipes select ordinary token-state INT8
or INT8 with Hadamard rotation and optional ReplaySSM. This workflow quantizes
recurrent state only.

## Training/serving mismatch and solution

### Why chunk-only state QAT can disagree with serving

GDN and KDA training normally process many tokens with chunked prefill kernels.
Serving processes the prompt with prefill, then updates the recurrent state for
each generated token. Applying state quantize/dequantize (QDQ) once per training
chunk does not reproduce the state consumed by each decode token.

Let `F_t` be the state update for token `t`, and `Q` be QDQ. Starting from the same
state `S`, even a two-token example gives different rounded states in general:

```text
QDQ only at the chunk boundary: Q(F_2(F_1(S)))
QDQ after each decode update:   Q(F_2(Q(F_1(S))))
```

The second decode update consumes the first update's rounded state. The chunked
computation consumes its unrounded state, so it follows a different trajectory.
A straight-through estimator (STE) changes the backward rule; it cannot repair
this forward mismatch.

Matching the QDQ boundaries is also insufficient if the arithmetic differs.
Prefill and recurrent kernels can use different BF16 casts, reduction orders,
normalization, and gate calculations. Small differences can cross INT8 rounding
thresholds and affect later tokens. ReplaySSM additionally reconstructs state
from a quantized checkpoint and a weighted BF16 key/update ring; an ordinary FP32
recurrence does not reproduce that arithmetic.

### Match each training phase to its serving phase

The solution is **a native chunked prefix followed by a serving-aligned recurrent
suffix**. `--prefill-tokens` selects the split in this example;
`linear_attention_training_phase(model, prefill_lengths)` supplies it to the
converted layers. All tokens still come from the training batch through teacher
forcing; no generation loop or running vLLM server is required.

```text
prompt tokens                   completion tokens
[native chunked prefill] -> QDQ handoff -> [native token/replay updates]
       training prefix                       training suffix
```

1. **Prefix:** use the selected serving prefill arithmetic. A fresh prefix has
   no internal state QDQ; native floating-point rounding still applies. A
   continuation prefix encodes its incoming nonzero state according to the cache
   policy. Keeping the prefix chunked matches serving prefill.
2. **Handoff:** encode the prefix's final state before the first suffix token.
   Ordinary state QDQ uses TensorQuantizer; the ReplaySSM profile uses its native
   INT8 + Hadamard checkpoint encoding. An empty suffix performs no handoff QDQ.
3. **Suffix:** use the native recurrent arithmetic and cache schedule. Token mode
   quantizes each next-token state; replay mode retains BF16 key/update entries
   and quantizes at checkpoint refreshes. Both read the current output from the
   working state before checkpoint rounding. KDA's per-key-channel decay follows
   the selected native implementation as well.

For the ordinary state-only serving plugin, QDQ happens **before** each native
decode call; training retains QDQ values **after** each update. These placements
give the next token the same input when the same deterministic QDQ is applied once:
if serving retains working state `W_t`, training retains `Q(W_t)`, and the next
serving call also consumes `Q(W_t)`. Raw stored tensors need not be identical;
compare the states consumed by the recurrence and the resulting outputs.

The native kernels supply forward values. ModelOpt attaches a differentiable
Torch computation for backward and uses identity STE through casts and QDQ.
The example applies QAT or QAD loss to the suffix, with gradients flowing through the
handoff into the prefix. This lets training observe the cache rounding used by
the selected serving policy.

Native-cache checks compare outputs, checkpoint values/scales, and replay entries;
training checks verify suffix-to-prefix gradients and optimizer updates. The
reproduced mismatch is resolved for the tested profiles and runtime settings.
See [State quantization alignment](STATE_QUANTIZATION.md) for the tests, earlier
failed approaches, and results. Exact kernel/cache matches do not establish
full-model quality recovery or training throughput. Match the serving version,
quantizer settings, initial state, and phase boundaries when evaluating; engine
scheduling and prompts split across multiple prefill calls still need integration
validation.

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
  --recipe general/ptq/linear_attention_state_int8_block32_dynamic \
  --train-steps 1 --length 128 --prefill-tokens 64
```

The default example recipe uses `backend="serving"` and
`precision="vllm_0_15"`, importing kernels directly from
[vLLM 0.15.1](https://github.com/vllm-project/vllm/tree/v0.15.1).
Install its optional dependency in an environment compatible with that release:

```bash
pip install -r examples/llm_qat/linear_attention/requirements-vllm.txt
```

This optional package supplies forward kernels;
training uses ModelOpt's Torch adjoint and does not launch a vLLM server. The example
requirements own the version pin. The ReplaySSM profile requires its compatible serving fork instead.
The block32 recipe
`general/ptq/linear_attention_state_int8_block32_dynamic` selects this profile;
the Hadamard recipe selects `precision="replayssm"`.

For both serving profiles, launch training and evaluation with
[`with_vllm_defaults.sh`](with_vllm_defaults.sh). It explicitly sets
`FLA_USE_FAST_OPS=0`, `USE_DEFAULT_FLA_NORM=0`, `FLA_GDN_FIX_BT=0`,
`FLA_USE_CUDA_GRAPH=0`, and `FLA_TRIL_PRECISION=ieee`, replacing inherited overrides
before Python imports vLLM. The library does not enforce these flags at import.
For example, prepend the wrapper to the training command above and select the
block32 recipe:

```bash
bash examples/llm_qat/linear_attention/with_vllm_defaults.sh \
  torchrun --standalone --nproc-per-node=1 examples/llm_qat/linear_attention/train.py \
  --model /path/to/local-model \
  --train-data /path/to/tokenized/train_text_document \
  --output /path/to/megatron-qat-checkpoint \
  --recipe general/ptq/linear_attention_state_int8_block32_dynamic \
  --train-steps 1 --length 128 --prefill-tokens 64

# Use the same launch settings for the native-kernel training checks.
bash examples/llm_qat/linear_attention/with_vllm_defaults.sh \
  python -m pytest tests/gpu_vllm/torch/quantization/test_linear_attention_training.py
```

Prefix the vLLM server or offline evaluation command with the same wrapper;
wrapping only a client of an already-running server does not configure its
workers. For multiple nodes, apply it to the worker launch on each node.
The wrapper prints the five settings to stderr; save that log with evaluation
results. Matching settings still requires numerical checks on each GPU/runtime
combination because Triton autotuning can select different configurations.

`--recipe` accepts a built-in recipe name or a custom YAML path. The public-vLLM block32 recipe shown above is the CLI default.

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
the Hadamard or replay policies; those target the separate quantized-ReplaySSM serving fork.

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
| Execution policy | Choose the native precision profile and checkpoint refresh window | Recipe `linear_attention` entries |
| Batch metadata | Specify where each sequence switches from prefill to decode | `linear_attention_training_phase(model, prefill_lengths)` |

`LinearAttentionConfig` is the single execution config for both phases:

```yaml
linear_attention:
  - module_name: "*"
    cfg:
      backend: serving
      precision: replayssm
      replay_window: 8
```

`backend="serving"` selects native forward arithmetic; `precision` selects the
serving implementation and its codec. `replay_window=1` (the default) refreshes
state every token. Values from 2 to 64 enable replay and require `replayssm`.
This profile includes Hadamard rotation; `vllm_0_15` uses ordinary state QDQ.
`state_block_v` is the legacy per-tile scale width, defaulting to 64; blockwise
grouping still comes from `TensorQuantizer.block_sizes`.

Both profiles fix working-state readout, QDQ at handoff, and no internal prefix
QDQ. These are native arithmetic requirements, not separate configuration knobs.
Use the flat schema above. Earlier unpublished nested `state`/`decode`/`replay`
configs and reference-training policies are unsupported.

Each `linear_attention` rule assigns the policy to matching modules. ModelOpt saves
it with the registered quantizers. Batch prefix lengths remain runtime metadata.

### Load a state recipe

State quantizers start disabled. The shared state recipe enables `*gdn_state_quantizer` and `*kda_state_quantizer`
with signed narrow-range INT8, dynamic scales, and `axis=(0, 1)`. Use
`*gdn_state_quantizer` or `*kda_state_quantizer` to select just one layer type. For E4M3, set the quantizer config to
`{"num_bits": [4, 3], "type": "dynamic", "axis": [0, 1]}` and set
`precision="vllm_0_15"`. The execution config alone does not enable a quantizer.

Start with the token-state recipe and select one of the schedules below **before**
calling `mtq.quantize`:

```python
import modelopt.torch.quantization as mtq
from modelopt.recipe import load_recipe

recipe = load_recipe("general/ptq/linear_attention_state_int8_dynamic").quantize.model_dump()
policy = recipe["linear_attention"][0]["cfg"]
```

The complete recipe composes both INT8 state quantizers with the Hadamard
execution policy and uses native forward kernels with a Torch adjoint. Use the phase context
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

1. **Prefill:** process tokens 0–95 with native chunked arithmetic, leaving state
   unquantized during the prefix.
2. **Handoff:** carry the resulting state into decode, rotate its value dimension
   into the Hadamard basis, and apply INT8 QDQ.
3. **Decode:** process tokens 96–127 recurrently, applying INT8 QDQ after every
   state update. Later tokens consume the rounded state; outputs are transformed
   back to the original value basis.

The boundary is independent of the fixed 64-token prefill chunk size: this 96-token prefix contains a
full 64-token chunk and a partial 32-token chunk. The phase switches after token
95, not after each chunk.

Two 128-token sequences can use `prefill_lengths=[64, 96]` in the same batch:
their decode suffixes then contain 64 and 32 tokens, respectively. The next batch
can have different lengths while reusing the same quantization recipe. This is
why the execution policy is saved in `linear_attention_config`, while the lengths
are supplied per batch through `linear_attention_training_phase`. They are not
saved as part of the model's quantization policy. The context selects numerical
phases; the caller still supplies training labels and any loss masking.

The combined path requires explicit lengths, including `[0]` for decode only or
`[T]` for an all-prefix sequence. An empty suffix causes no decode QDQ. For QAT/QAD,
leave a nonempty suffix and apply the loss there so training observes cache rounding.

### Select the serving policy

| Recipe / policy | State writes | Serving dependency |
| --- | --- | --- |
| `linear_attention_state_int8_block32_dynamic` | QDQ at handoff and every token write, one scale per key row and 32 value channels | Public vLLM 0.15.1 |
| `linear_attention_state_int8_dynamic` | INT8 + H32 at handoff and every token write | Compatible quantized-ReplaySSM fork, window 1 |
| Same Hadamard recipe with `replay_window=8` | INT8 + H32 at handoff and each window refresh; BF16 key/update ring between refreshes | Compatible quantized-ReplaySSM fork |

All three read the current token's output from the working state, before checkpoint
rounding. A fresh prefill has no internal state QDQ. Continuation prefill consumes
a state rounded according to the selected cache codec.

For Hadamard or ReplaySSM, use a vLLM environment providing
`vllm.model_executor.layers.fla.ops.fused_recurrent_replayssm`, including its KDA
vector-gate entry point. These kernels are not in public vLLM 0.15.1. The validation
report records the exact local fork revision and source hashes; it does not claim
that every published fork revision contains these interfaces. ModelOpt imports the
kernels and builds private cache buffers for autograd; it does not start a server.

Load the Hadamard recipe and optionally enable an eight-token replay window:

```python
recipe = load_recipe("general/ptq/linear_attention_state_int8_dynamic").quantize.model_dump()
recipe["linear_attention"][0]["cfg"]["replay_window"] = 8
```

The serving ReplaySSM profile stores BF16 key/update vectors once. Additional factor
FP8 QDQ, repeated factor encoding, stored-state readout, and internal-prefix QDQ are
unsupported. Mathematical reference implementations live only under `tests/`.

The fixed chunk size of 64 counts prefill tokens. `replay_window=8` counts suffix tokens
between checkpoint refreshes. H32 uses one scale per key channel and 32 value
channels, with FP16 scale metadata. The ordinary block32 recipe instead uses
standard `TensorQuantizer(block_sizes={-1: 32})` grouping.

To customize the CLI, copy a recipe YAML and pass `--recipe /path/to/recipe.yaml`.
`--prefill-tokens` sets the sequence split; it does not enable quantization.

### Migrating chunk-only state QAT

Chunk-only state QAT is retired. Enabling GDN or KDA state quantization without
`backend="serving"` fails during conversion. Select a serving recipe and supply
explicit prefix lengths; conversion does not infer a different QDQ schedule.

W-only QAT is also retired: enabled `gdn_w_quantizer` configurations or checkpoints
raise an error. The disabled handle remains loadable for checkpoint compatibility.
State QAT imports serving kernels and does not quantize WY operands.

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
`algorithm=None` without a calibration pass. The native replay profile retains
BF16 key/update entries without separate factor quantizers.

Policies persist through ModelOpt save/restore. Per-batch prefix lengths are
runtime metadata and must be supplied for each workload. The context restores
previous lengths on exit and supports nesting. Concurrent forwards on the same
model instance with different phase contexts are unsupported.

## Replay and Hadamard numerical behavior

In token mode, the imported ReplaySSM kernel runs with a one-token window. In replay
mode, it reads the quantized checkpoint plus the BF16 update ring and refreshes the
checkpoint at the configured window boundary. KDA applies decay on the key axis;
Hadamard rotates the value axis. Training retains the same checkpoint values,
scales, and ring entries, while its adjoint uses identity STE for casts and QDQ.

The persistent ModelOpt carry remains floating fake-QDQ data. Temporary native cache
buffers are allocated for forward evaluation, so this example makes no storage or
speed claim. Numerical checks and short QAT/QAD runs are reported separately from
model-quality evaluation in [the alignment report](STATE_QUANTIZATION.md).

## Training boundaries

The Torch implementation uses autograd through initial states, chunk handoff,
token writes, and replay refreshes. The decode recurrence runs token by token in
Python, so long training suffixes can be slow.

Serving-aligned prefixes call the imported prefill primitives without internal
state QDQ. This example has no prefill GEMM QDQ or approximate inverse. Use the Megatron training forward
without a serving inference context. ModelOpt saves execution policies, while
per-batch prefix lengths must be supplied again during training. Distributed
decode training and model-quality recovery require separate qualification.
