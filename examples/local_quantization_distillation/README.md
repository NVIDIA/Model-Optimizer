# Local quantization distillation with Transformers

Bypass repairs an existing ModelOpt PTQ model using an original teacher. A
teacher forward supplies each selected unit's input and target output; the
student replays that input and immediately backpropagates its own valid-token
MSE. Units train independently during the same pass, without storing a corpus of
hidden states or backpropagating through neighboring units. Prompts contribute
alongside other valid tokens.

`TransformersLocalDistillationAdapter` supports Hugging Face text models and
standalone Torch modules. Default selection covers quantized text MLP/MoE units.
Use exact, disjoint `units` paths or semantic `unit_types` to include attention,
GDN, Mamba or whole decoder blocks. Normalization and residual paths outside the
selected module are outside its objective.
Repair skips the unused vocabulary projection when the head is outside selected
units, while preserving the normal model/root lifecycle. An explicitly selected
head receives all-token targets. Active head calibration and GPTQ/AWQ collectors
also receive all-token inputs; frozen initialization keeps the short-logit path.

Use the original teacher and an independently loaded or copied student with
matching architecture, weights, dtype and placement before PTQ. Packed FP8/FP4
checkpoint bytes need their supported reconstruction path; casting bytes to
BF16 does not reconstruct their weights or scales.

## Environment and supported models

The broad family baseline is Transformers **5.23.0.dev0**, pinned to
[`084b1fd1b9983dca2ce12441aec4478371b9fe8f`](https://github.com/huggingface/transformers/tree/084b1fd1b9983dca2ce12441aec4478371b9fe8f).
Use the complete matching package and dependencies. Native scoped CP was
verified with Torch 2.14 and requires its module-wrapper attention API. Follow
your environment and cluster setup instructions before installing or launching.

[Model coverage](MODEL_COVERAGE.md) describes upstream families, differentiable
text extensions, checkpoint-loading limits and the evidence behind them.
[Advanced configuration](ADVANCED.md) covers native topology, persistence and
custom families.

## GPTQ → Bypass → held-out teacher KLD

Assume `teacher`, re-iterable `calibration_batches`, unique `training_batches`,
independent `validation_batches`, and `diagnostic_batches` already exist on the
intended device. Keep all four conversation splits disjoint. This example uses
ordinary public 4-bit weights; choose a PTQ/repair format and shape supported by
your backend.

```python
import copy
import torch
import modelopt.torch.distill as mtd
import modelopt.torch.quantization as mtq
from modelopt.torch.distill.plugins.transformers import (
    TransformersLocalDistillationAdapter,
    transformers_family,
)
from examples.local_quantization_distillation.evaluate import evaluate_pair

teacher.eval().requires_grad_(False)
student = copy.deepcopy(teacher)
adapter = TransformersLocalDistillationAdapter()
repair = mtd.LocalDistillationConfig(
    unit_types=("mlp", "moe"), mode="scale_only",
    updates=512, sequence_length=32768,
    global_batch_size=8, micro_batch_size=1,
    validation_interval=64,
)
patterns = transformers_family(teacher).weight_patterns(teacher, repair)
if not patterns:
    raise ValueError("No selected projection weights; inspect the family and units.")
recipe = {
    "quant_cfg": [
        {"quantizer_name": "*", "enable": False},
        *[
            {"quantizer_name": pattern, "enable": True,
             "cfg": {"num_bits": 4, "axis": 0}}
            for pattern in patterns
        ],
    ],
    "algorithm": {"method": "gptq", "perc_damp": 0.01,
                  "block_size": 128, "fused": False},
}

@torch.no_grad()
def calibrate(model):
    for batch in calibration_batches:
        adapter.teacher_microbatch(model, [], batch, lambda *_: None)

mtq.quantize(student, recipe, forward_loop=calibrate)
before = evaluate_pair(teacher, student, diagnostic_batches, adapter=adapter)
result = mtd.local_distill(
    student, teacher, training_batches, repair,
    validation_dataloader=validation_batches, adapter=adapter,
)
after = evaluate_pair(teacher, result.student, diagnostic_batches, adapter=adapter)
print(before["kld_teacher_candidate"], after["kld_teacher_candidate"])
print(after["projection_modes"])
```

Family-generated patterns exclude routers from PTQ. GPTQ must see the real
inputs to each selected projection, including routed expert assignments; use a
quantization-supported experts implementation. Never copy GPTQ-modified weights
back into the teacher.

Scale-only repair freezes element codes, signs and lookup catalogs and updates
legal native scale fields through FP32 log-scale masters. Finalization removes
temporary repair state. Enabled activation quantizers retain their calibrated
forward by default. Unsupported format/layout combinations fail preflight.
`updates=0` inspects selected units but does not exercise training-time native
field validation.

Defaults include Adagrad, scale LR 0.01, 5% warmup, cosine decay, gradient clipping
at 1 and final-state selection. These are starting values, not an optimal recipe
for every model. Optional `best_validation_per_unit` selection uses local MSE;
select PTQ initializers and assess whole-model fidelity separately using held-out
**full-vocabulary KL(teacher || candidate)**. Surrogate-corpus PPL is secondary.

## GLM Flash MoE IQ2_S variation

For the original GLM-5.3-Flash teacher and matching student, use these assignments
in the example above **before calibration and PTQ**. Family-generated patterns
select the MoE projections and leave routers unchanged; no backend-specific
expert-bank names are needed.

```python
repair = mtd.LocalDistillationConfig(
    unit_types=("moe",), mode="scale_only",
    updates=512, sequence_length=32768,
    global_batch_size=8, micro_batch_size=1,
)
patterns = transformers_family(teacher).weight_patterns(teacher, repair)
recipe["quant_cfg"] = [
    {"quantizer_name": "*", "enable": False},
    *[
        {"quantizer_name": pattern, "enable": True,
         "cfg": {"num_bits": "iq2_s", "backend": "ggml"}}
        for pattern in patterns
    ],
]
recipe["algorithm"] = {
    "method": "gptq", "block_size": 256, "importance_weighted": True,
    "perc_damp": 0.01, "fused": False,
}
```

Prepare this variation with `--sequence-length 32768` and
`--calibration-sequence-length 4096`. The repair budget is 512 updates of eight
unique 32K packs (134,217,728 training tokens), with 128 calibration, validation
and diagnostic packs each by default. These settings are starting values, not a
new full-model quality result for this integration.

## Prepare unique packed data

The preparation tool uses Transformers tokenization and NumPy document packing.
Run from the repository root, using persistent paths and pinned source revisions:

```bash
python -m examples.local_quantization_distillation.prepare_data \
  --tokenizer /path/to/teacher \
  --output /path/to/new-packed-corpus \
  --sequence-length 32768 --calibration-sequence-length 4096 --updates 512 --global-batch-size 8
```

The default source is Nemotron Post-Training Dataset v2 with stem/chat/math/code
splits. Preparation resolves dataset/tokenizer revisions, applies the tokenizer's
chat template, deduplicates rendered conversations and assigns each conversation
to one split by stable hash. Long conversations may span packs; source token
intervals never repeat or cross splits. Exhausting unique data is an error.
The manifest records source revisions, hashes, provenance and actual budgets.

Load deterministic update batches with the helper; counts must divide exactly
across the global batch and unique data lanes:

```python
from examples.local_quantization_distillation.prepare_data import PackedBatchSequence

training_batches = PackedBatchSequence(
    "/path/to/new-packed-corpus", "train",
    global_batch_size=8, device=device,
)
```

Use `calibration`, `validation` and `diagnostic` for the other partitions. In a
distributed run, also pass the actual `dp_size` and `dp_rank`; CP/TP/EP peers
consume the same sequence for their unique data lane. There is no epoch wrap or
sampler padding. Match the prepared context to `repair.sequence_length`.

Batches carry `_packed_seq_ids` (`0` padding, positive document IDs), reset
`position_ids` and `valid_mask`. Ordinary causal families can use block-causal
attention masks. Recurrent, compressed and sparse families use independent
document forwards to isolate convolution, scan, pooling and indexer state.
A 4D mask alone cannot isolate those operations. Supported Qwen hybrid and
Nemotron H families opt in to grouping up to eight independent documents within
the original token budget, preserving MRoPE positions, labels and valid counts.
Calibration, custom masks, unknown metadata and context-coupled activation
statistics retain serial document forwards. Packed multimodal payloads still
need a family-owned implementation.

## Whole-model diagnostics

`evaluate_pair(teacher, candidate, batches, adapter=adapter, token_chunk=128,
vocabulary_chunk=8192)` computes full-vocabulary KLD from captured final-head
inputs, preserving the actual task and packed-document boundaries. It requires
**PP=1** and full-context batches. CP diagnostics replicate context and count
unique data lanes once.
All ranks must advance through matching outer batch schedules; the adapter
coordinates exhaustion/errors and rejects unequal schedules. Slice distinct
samples only over the declared unique data lanes.

Labels are already shifted next-token targets. Prepared packs provide them;
without labels, KL is still computed and PPL is unavailable. KL covers every
valid input position, including prompts and document-final tokens; NLL/PPL use
valid within-document targets. `projection_modes` reports `chunked_linear` for
vocabulary-sliced ordinary Linear heads, or `chunked_head` for custom, quantized
or TP heads whose real module forward runs on token chunks. These paths avoid
full-sequence vocabulary tensors; a logits-only model without an exposed head
reports `full_logits`. Final hidden states remain resident. Muse/Mistral family
post-projection transforms are preserved. Heads with context-coupled activation
statistics require a specialized diagnostic path and are refused by this helper.
Global module forward hooks are also refused; module-local head hooks are preserved.

Compare every initializer and repaired candidate on the same held-out split.
Record tokenizer/model revisions and `kl_tokens`/`next_token_targets` alongside
KLD. Recheck a deployment-converted artifact because export may fuse, resmooth or
requantize weights. An improvement on this diagnostic corpus does not establish
downstream benchmark quality.

## Distributed execution and checkpoints

The caller owns process launch, meshes and matching teacher/student placement.
Native TP/EP/FSDP uses HF `TransformersDeviceMesh`; EP overlaps dense FSDP/TP
ranks rather than adding an independent axis. Calibrate PTQ first, then call
`adapter.parallelize` on both models. The native plans and code/scale blocks
must support the requested physical partitions.

Full-causal half-precision SDPA can use scoped native CP. Packed, recurrent,
compressed, FP32, custom-mask or incompatible activation-statistic paths use
`replicated_full_context`: every CP rank retains the full forward/replay inputs,
with disjoint valid-token masks and gradient reduction. This fallback saves
**neither forward compute nor activation memory**. Inspect
`execution.context_parallel_modes`; a CP axis alone does not demonstrate savings.

With `output_dir`, all ranks publish a topology-native Torch DCP model checkpoint.
Restore with `adapter.restore(matching_distributed_ptq_skeleton, directory)` on
all ranks. Optimizer/scheduler/RNG/data-position resume instead uses
`checkpoint_dir` and `resume_from`. Neither checkpoint is a topology-changing
HF/GGUF deployment export. [Advanced configuration](ADVANCED.md) gives the mesh
and resume contracts.

## Benchmark and validation

The Transformers-only benchmark supports `qwen3` and `moonlight` random-weight
prefixes. Run it only inside an existing allocation. For four local GPUs, this
example uses dense FSDP4 and overlapping EP2:

```bash
python -m torch.distributed.run --nproc_per_node=4 \
  -m examples.local_quantization_distillation.benchmark_transformers \
  --family qwen3 --format nvfp4 --dp 4 --ep 2 \
  --layers 4 --batch 4 --seq-len 2048 --warmup 5 --steps 8 \
  --output /path/to/results/qwen3.json
```

Create the output parent directory first. The benchmark fetches configuration
metadata and initializes random weights, without downloading pretrained weights
or writing model checkpoints. `WORLD_SIZE = pp * cp * dp * tp`; EP divides
`dp * tp` and must be a multiple of TP. `--parameter-sharding replicated` is a
pure-DP control. Native FSDP+PP and dispatch-EP+PP are refused. Use
`--extend-context` explicitly when exceeding the source's configured range; this
makes no long-context quality claim.

[Validation](VALIDATION.md) separates executed checks from externally launched
GPU tests. [Results](RESULTS.md) distinguishes compatibility evidence from older
Transformers prefix timing measurements. [Review map](REVIEW_MAP.md) lists source
ownership.
