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
CUDA GPUs, and mutually compatible Bridge/Core revisions. GDN state QAT requires
Core with `_compute_gates` and either `forward_pre_attn_and_core_attn` or
`_forward_compute`; Core 0.19.2 is
unsupported, including the Core shipped in `nemo:26.08`. ModelOpt adapts
Megatron `GatedDeltaNet` and `KimiDeltaAttention`; FLA supplies their kernel
dependency. KDA requires a Core revision exporting `KimiDeltaAttention` and a
Bridge provider that can convert the chosen checkpoint. Layer support in
ModelOpt alone does not provide checkpoint conversion support in Bridge.

The single-GPU GDN QAT/QAD checkpoint tests passed with this source pair:

| Dependency | Tested revision/version |
| --- | --- |
| Megatron Bridge | `056b78e0a96fb0f5b50a82d09d1ec456b073d1e0` |
| Megatron Core (Bridge's pinned submodule) | `e998be072d22ff57b4b49899819a1a153f4f3dc9` |
| PyTorch / Transformer Engine | `2.13.0` / `2.16.1` |
| vLLM / FLA | `0.30.0` / `0.5.1` |
| Transformers / NVIDIA Resiliency Extension | `5.15.0` / `0.7.0` |

Use Bridge's installation instructions for the selected revision. A Core source
checkout also needs its dataset extension built: run
`python setup.py build_ext --inplace` from the Core repository root. The tested
Core revision supplies GDN but not `KimiDeltaAttention`; this pair does not
qualify KDA Bridge training.

Prepare a local model/tokenizer snapshot and a Megatron `.bin`/`.idx` dataset
using that tokenizer. `--train-data` takes the shared filename prefix without an
extension; see [data preparation](../../megatron_bridge/README.md#data-preparation).
For a model already built in Megatron, use the Python API below.

For ordinary state QDQ, install the example dependencies from the repository root:

```bash
pip install -r examples/llm_qat/linear_attention/requirements.txt
pip install -r examples/llm_qat/linear_attention/requirements-vllm.txt
```

The example pins `fla-core==0.5.1` and public `vllm==0.30.0`. vLLM supplies forward
kernels; no running server is needed for training.

## Use prompt boundaries from data

For prompt-completion data, pass a directory containing `training.jsonl` records
with `input` and `output` fields. `input` is the full rendered prompt, including
chat-role delimiters and any earlier turns; `output` is the selected assistant
continuation. Bridge tokenization supplies `context_lengths` and `token_count`.
The phase uses those prompt lengths and excludes right padding using the valid
input length (`token_count - 1`, accounting for the next-token label shift).
Loss masks remain dataset-owned; their values are not used to guess lengths.

```bash
bash examples/llm_qat/linear_attention/with_vllm_defaults.sh \
  torchrun --standalone --nproc-per-node=1 examples/llm_qat/linear_attention/train.py \
  --model /path/to/local-model --sft-data /path/to/prompt-completion-data \
  --output /path/to/qat-checkpoint --length 1024 --train-steps 10
```

Add `--teacher-model` for QAD. Omit `--prefill-tokens` to use each record's
prompt boundary. This data-driven CLI currently supports TP/DP with PP=CP=1;
fixed token splits below retain the existing PP support. Multi-turn conversations
should be rendered into prompt/target pairs, selecting the assistant turn in
preprocessing rather than scanning arbitrary token IDs during training.

A custom packed collator can use `run_training(..., prefill_tokens=None)` and
supply per-segment `prefill_lengths`, `sequence_lengths` and actual packed
storage boundaries (`cu_seqlens_q_padded` when present). Padded boundaries alone
do not identify real tokens. The example does not provide a chat packer.

## Run QAT or QAD with a fixed split

```bash
bash examples/llm_qat/linear_attention/with_vllm_defaults.sh \
  torchrun --standalone --nproc-per-node=1 examples/llm_qat/linear_attention/train.py \
  --model /path/to/local-model \
  --train-data /path/to/tokenized/train_text_document \
  --output /path/to/megatron-qat-checkpoint \
  --recipe general/ptq/linear_attention_state_int8_block32_dynamic \
  --train-steps 1 --length 128 --prefill-tokens 64
```

For QAD, use a separate output directory and add an unquantized teacher:

```bash
bash examples/llm_qat/linear_attention/with_vllm_defaults.sh \
  torchrun --standalone --nproc-per-node=1 examples/llm_qat/linear_attention/train.py \
  --model /path/to/local-model \
  --teacher-model /path/to/unquantized-model \
  --train-data /path/to/tokenized/train_text_document \
  --output /path/to/megatron-qad-checkpoint \
  --recipe general/ptq/linear_attention_state_int8_block32_dynamic \
  --train-steps 1 --length 128 --prefill-tokens 64
```

Student and teacher must have matching tokenizer vocabularies and output
vocabulary dimensions. QAD uses
`kd_loss_alpha=1.0`, disabling the language-model loss. Both modes use Bridge
`pretrain()`; QAD wraps the student in Bridge's `DistillationProvider`. Both fixed-split modes
mask loss to positions after `--prefill-tokens`.

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

ModelOpt separates quantization settings from execution and batch metadata:

| Setting | Purpose | Where it belongs |
| --- | --- | --- |
| `TensorQuantizer` | Enable state QDQ; select format and scale grouping | Recipe `quant_cfg` |
| `LinearAttentionConfig` | Select native arithmetic | Recipe `linear_attention` rules |
| Prefix lengths | Specify where each sequence switches to recurrence | `linear_attention_training_phase` at runtime |

For example, a complete recipe can select this execution policy:

```yaml
linear_attention:
  - module_name: "*"
    cfg:
      backend: serving
      precision: vllm
```

`precision="vllm"` selects vLLM 0.30.0's Triton GDN kernels and standalone
KDA kernels. For GDN serving, select `gdn_prefill_backend="triton"` and enable
`VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE=1`. Mixed prefill/decode
batches are outside this profile. Standalone KDA has no model-level serving
guarantee; Kimi-Linear/Kimi-K3 on vLLM 0.30 requires `precision="vllm_kimi_k3"`,
both KDA backends set to `triton`, and an unbounded softplus gate.

`TensorQuantizer.block_sizes` owns blockwise scale grouping. `state_block_v`
controls the execution tile width and legacy per-tile grouping; it does not
replace `block_sizes`. An execution policy alone does not enable a quantizer.
Importing only a recipe unit configures the quantizers without its full policy.

For a custom recipe, pass its YAML path to `--recipe`. State QAT requires
`backend="serving"` and explicit prefix lengths.

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

For a single-rank, state-only checkpoint, use Bridge to export the floating-point
weights and ModelOpt to save the quantization metadata separately:

```python
from pathlib import Path

import torch
import modelopt.torch.opt as mto
from modelopt.torch.utils import distributed as dist
from modelopt.torch.utils.plugins.mbridge import (
    load_mbridge_model_from_hf,
    load_modelopt_megatron_checkpoint,
)

dist.setup()  # Run with torchrun --nproc-per-node=1.
bridge, _, models, student, _ = load_mbridge_model_from_hf(
    hf_model_name_or_path="/path/to/original-hf-model",
    provider_overrides={"gradient_accumulation_fusion": False},
    load_weights=False,
)
load_modelopt_megatron_checkpoint(models, "/path/to/output/checkpoints/iter_0000002")
export_dir = Path("/path/to/hf-export")
bridge.hf_pretrained.save_artifacts(export_dir)
# Perform normalization convention changes in FP32 before writing HF weights.
student.float()
bridge.save_hf_weights([student], export_dir)
torch.save(mto.modelopt_state(student), export_dir / "megatron_modelopt_state.pt")

# Reimport HF weights into Megatron, then restore the saved state quantizers and policy.
_, _, _, restored, _ = load_mbridge_model_from_hf(
    hf_model_name_or_path=str(export_dir),
    provider_overrides={"gradient_accumulation_fusion": False},
    init_model_parallel=False,
)
mto.restore_from_modelopt_state(
    restored, modelopt_state_path=export_dir / "megatron_modelopt_state.pt"
)
dist.cleanup()
```

This example exports FP32 weights. GDN's output norm stores a zero-centered
weight in Megatron and a weight centered on one in HF; doing that conversion in
BF16 can erase small training updates. FP32 preserves the effective normalization
multiplier. Load these weights in Transformers with `dtype=torch.float32`;
forcing BF16 can introduce rounding again.

The sidecar uses ModelOpt's existing metadata format with **Megatron module names**.
Keep it with the exported weights. Transformers can load those weights but does
not apply this state-QAT policy; vLLM needs its own module-name mapping and state
adapter. HF export also omits optimizer/scheduler state: use the original Bridge
checkpoint to resume training. This export example covers dynamic state-only
quantization, not quantized projection weights or a packed recurrent cache.

Use `--tp_size`, `--pp_size`, and `--ep_size` as in the
[Megatron Bridge example](../../megatron_bridge/distill.py). Student and teacher
use the same topology. Sequence parallelism is enabled when TP exceeds one;
additional ranks use data parallelism. `--global-batch-size` controls accumulation
with microbatch size one and must be divisible by the data-parallel size.

Choose a topology supported by the model's Bridge/Core provider. This example
requires linear-attention layers in each local pipeline chunk. Context parallelism
is fixed at one. State QAT supports selective GDN recomputation on compatible Core
revisions, not full-layer recomputation. The available topology options do not
imply multi-GPU qualification.

## Validation and limitations

The two example tests exercise GDN QAT and QAD with shared compilation setup. Each
saves a checkpoint, checks exact weight and policy restoration, resumes for one
step, and exports/reloads HF weights with the separate Megatron ModelOpt metadata.
The HF round trip checks all effective parameters, logits, and attention gradients.
The QAD test also checks that the teacher stays frozen and unquantized.
Both tests passed on an RTX A6000 with the dependency versions above. They use a
tiny random Qwen3.5 GDN model and mock data: one optimizer step, then one resumed
step. This validates the workflow, not pretrained-model quality recovery. Run:

```bash
bash examples/llm_qat/linear_attention/with_vllm_defaults.sh \
  python -m pytest tests/examples/megatron_bridge/test_linear_attention.py
```

The CLI entry point was also checked in-process with local HF weights and a
`.bin`/`.idx` dataset; both QAT and QAD saved a checkpoint.

GDN Bridge workflow tests and KDA Megatron layer tests are distinct. The example
tests use a GDN provider; a KDA trainer needs a compatible provider.
Native-cache and gradient checks are summarized in
[State quantization alignment](STATE_QUANTIZATION.md#validation-results).

The recurrent suffix uses Python orchestration and an explicit recurrence
backward, so long suffixes can be slow. Each prefix and suffix token runs one
native forward. Kernel/cache agreement does not establish pretrained-model
quality recovery, full serving-engine equivalence, or training speed. Prefill
GEMM quantization and approximate inverse are outside this example.

## Optional ReplaySSM profile

See [ReplaySSM state QAT](REPLAYSSM.md) for the optional INT8/Hadamard fork,
its checkpoint window, and the proposed two-precision extension.

## State-only serving with vLLM

The [vLLM example](../../vllm_serve/README.md#linear-attention-state-quantization)
uses the same state QDQ helper for plain FP8/INT8 policies. Match the runtime,
quantizer grouping, and scheduler prefill boundaries to the training setup.
