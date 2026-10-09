# Quantization-Aware Training and Distillation for Linear Attention

This example trains Megatron-Core GDN or KDA attention parameters with recurrent
state fake quantization. Megatron Bridge runs the optimizer, distributed training,
and checkpointing. QAT uses next-token cross-entropy; QAD uses a frozen,
unquantized teacher and Bridge's logits distillation loss.

Training combines native chunked prefill with a recurrent suffix that follows
the selected serving arithmetic and state quantization schedule. Loss is applied
to the suffix, with gradients flowing through the state handoff into the prefix.
See [State quantization alignment](STATE_QUANTIZATION.md) for the mismatch,
solution, and numerical validation. The example quantizes recurrent state only.

## Requirements

Use the [Megatron Bridge environment](../../megatron_bridge/README.md#pre-requisites),
CUDA GPUs, and mutually compatible Bridge/Core revisions. ModelOpt adapts
Megatron `GatedDeltaNet` and `KimiDeltaAttention`; FLA supplies their kernel
dependency. KDA requires a Core revision exporting `KimiDeltaAttention` and a
Bridge provider that can convert the chosen checkpoint. Layer support in
ModelOpt alone does not provide checkpoint conversion support in Bridge.

Prepare a local model/tokenizer snapshot and a Megatron `.bin`/`.idx` dataset
using that tokenizer. `--train-data` takes the shared filename prefix without an
extension; see [data preparation](../../megatron_bridge/README.md#data-preparation).
For a model already built in Megatron, use the Python API below.

For ordinary state QDQ, install the example dependencies from the repository root:

```bash
pip install -r examples/llm_qat/linear_attention/requirements.txt
pip install -r examples/llm_qat/linear_attention/requirements-vllm.txt
```

The example pins `fla-core==0.5.1` and public `vllm==0.15.1`. vLLM supplies forward
kernels; no running server is needed for training. INT8 + Hadamard and ReplaySSM
require a compatible quantized-ReplaySSM fork instead of public vLLM, including
its KDA vector-gate kernel when training KDA.

## Run QAT or QAD

```bash
bash examples/llm_qat/linear_attention/with_vllm_defaults.sh \
  torchrun --standalone --nproc-per-node=1 examples/llm_qat/linear_attention/train.py \
  --model /path/to/local-model \
  --train-data /path/to/tokenized/train_text_document \
  --output /path/to/megatron-qat-checkpoint \
  --recipe general/ptq/linear_attention_state_int8_block32_dynamic \
  --train-steps 1 --length 128 --prefill-tokens 64
```

Add `--teacher-model /path/to/unquantized-model` for QAD. Student and teacher must
have matching tokenizer vocabularies and output vocabulary dimensions. QAD uses
`kd_loss_alpha=1.0`, disabling the language-model loss. Both QAT and QAD mask loss
to positions after `--prefill-tokens`.

Only the student's linear-attention parameters are trainable; the remaining
parameters are frozen. Training uses BF16 mixed precision. Each dense sequence
has the same fixed prefix length in this CLI; packed data or variable boundaries
require a custom batch integration.

`with_vllm_defaults.sh` sets `FLA_USE_FAST_OPS=0`, `USE_DEFAULT_FLA_NORM=0`,
`FLA_GDN_FIX_BT=0`, `FLA_USE_CUDA_GRAPH=0`, and `FLA_TRIL_PRECISION=ieee` before
Python imports vLLM. Use the same runtime and settings for training and serving.
For evaluation, wrap the server or worker launch; wrapping a client does not
configure an already-running server. On multiple nodes, apply the wrapper to
workers on every node. The library does not enforce these settings at import.

## Select a state quantization recipe

`--recipe` accepts a built-in name or a custom YAML path. The complete recipes
under `general/ptq/` configure both GDN and KDA state quantizers and their
execution policy. They use dynamic quantization without a calibration pass.

| Recipe | State quantization | Runtime |
| --- | --- | --- |
| `linear_attention_state_int8_block32_dynamic` (default) | INT8 QDQ at handoff and every suffix token; one scale per key row and 32 value channels | Public vLLM |
| `linear_attention_state_int8_dynamic` | INT8 with Hadamard rotation over 32 value channels; checkpoint every suffix token | Compatible quantized-ReplaySSM fork |
| Same Hadamard recipe with `replay_window=8` | Checkpoint at handoff and every eight suffix tokens; BF16 key/update ring between refreshes | Compatible quantized-ReplaySSM fork |

Hadamard requires the value dimension to be divisible by 32. The state-only vLLM
fake-quant adapter targets ordinary TensorQuantizer QDQ; the Hadamard/replay
recipes target the separate native ReplaySSM implementation.

ModelOpt separates quantization settings from execution and batch metadata:

| Setting | Purpose | Where it belongs |
| --- | --- | --- |
| `TensorQuantizer` | Enable state QDQ; select format and scale grouping | Recipe `quant_cfg` |
| `LinearAttentionConfig` | Select native arithmetic and checkpoint frequency | Recipe `linear_attention` rules |
| Prefix lengths | Specify where each sequence switches to recurrence | `linear_attention_training_phase` at runtime |

For example, a complete recipe can select this execution policy:

```yaml
linear_attention:
  - module_name: "*"
    cfg:
      backend: serving
      precision: replayssm
      replay_window: 8
```

`precision="vllm"` selects the installed public vLLM arithmetic; `vllm_0_15` is
an accepted legacy spelling. `precision="replayssm"` selects native INT8/Hadamard
checkpoints. `replay_window=1` refreshes every token; values from 2 to 64 require
ReplaySSM. The prefill chunk size of 64 is independent of this suffix window.

`TensorQuantizer.block_sizes` owns blockwise scale grouping. `state_block_v`
controls the execution tile width and legacy per-tile grouping; it does not
replace `block_sizes`. An execution policy alone does not enable a quantizer.
Importing only a recipe unit configures the quantizers without its full policy.

To change the Hadamard checkpoint window in Python:

```python
from modelopt.recipe import load_recipe

cfg = load_recipe("general/ptq/linear_attention_state_int8_dynamic").quantize.model_dump()
cfg["linear_attention"][0]["cfg"]["replay_window"] = 8
```

For the CLI, put the modified recipe in a YAML file and pass its path to
`--recipe`. State QAT requires `backend="serving"` and explicit prefix lengths.
Keep the legacy `gdn_w_quantizer` disabled; this workflow does not quantize W.

## Supply prefill/decode boundaries in a training loop

A 128-token sequence with prefix length 96 runs tokens 0–95 through native
chunked prefill, encodes the handoff, then runs tokens 96–127 recurrently. The
prefix contains a full 64-token chunk and a partial 32-token chunk; the phase
switch happens at token 96, independently of the chunk boundaries.

The CLI supplies one fixed length for every sequence. The API accepts different
lengths per sequence, such as `[64, 96]`. These are batch metadata, so they are not
saved as part of the quantization policy. Keep the context active through
backward when activation checkpointing recomputes the forward.

```python
import torch

import modelopt.torch.quantization as mtq
from modelopt.recipe import load_recipe
from modelopt.torch.quantization.linear_attention import linear_attention_training_phase

# model is an initialized Megatron model; ids and shifted labels have shape [2, 128].
cfg = load_recipe("general/ptq/linear_attention_state_int8_block32_dynamic").quantize
model = mtq.quantize(model, cfg)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
model.train()
optimizer.zero_grad(set_to_none=True)
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

`[0]` means recurrence only; `[T]` means an all-prefix sequence and performs no
suffix handoff QDQ. QAT/QAD needs a nonempty suffix to train against state
rounding. The context selects execution phases; the caller still supplies
labels and loss masking. It restores the previous lengths on exit.
Concurrent forwards with different phase contexts on the same model are unsupported.

## Checkpointing and multiple GPUs

Bridge saves model, optimizer/scheduler, and ModelOpt state under
`<output>/checkpoints`. Reusing `--output` resumes from the latest checkpoint;
increase `--train-steps` to the desired total step count. Retain the model,
teacher, topology, dataset, and prefix boundary. The restored checkpoint supplies
its saved quantization policy; prefix lengths must still be supplied at runtime.
Export to a serving model is a separate step.

Use `--tp_size`, `--pp_size`, and `--ep_size` as in the
[Megatron Bridge example](../../megatron_bridge/distill.py). Student and teacher
use the same topology. Sequence parallelism is enabled when TP exceeds one;
additional ranks use data parallelism. `--global-batch-size` controls accumulation
with microbatch size one and must be divisible by the data-parallel size.

Choose a topology supported by the model's Bridge/Core provider. This example
requires linear-attention layers in each local pipeline chunk. Context parallelism
is fixed at one. The available topology options do not imply multi-GPU qualification.

## Validation and limitations

The minimal example tests run one GDN QAT step and one QAD step, with shared
compilation setup. They check student weight updates and a frozen, unquantized
QAD teacher. Run them in the matching Bridge/vLLM environment:

```bash
bash examples/llm_qat/linear_attention/with_vllm_defaults.sh \
  python -m pytest tests/examples/megatron_bridge/test_linear_attention.py
```

GDN Bridge workflow checks and KDA Megatron layer checks are distinct: the tested
Bridge provider supports GDN, while a KDA trainer needs a compatible provider.
Native-cache and gradient checks are summarized in
[State quantization alignment](STATE_QUANTIZATION.md#validation-results).

The recurrent suffix uses Python orchestration and a Torch adjoint, so long
suffixes can be slow. Kernel/cache agreement does not establish pretrained-model
quality recovery, full serving-engine equivalence, or training speed. Prefill
GEMM quantization and approximate inverse are outside this example.
