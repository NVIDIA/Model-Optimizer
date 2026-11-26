# Experimental NeMo AutoModel NVFP4 expert adapter

This example adds fixed-scale W4A4 fake quantization to NeMo AutoModel's native
`GroupedExperts` for QAD. It trains floating-point master weights with a
straight-through gradient and preserves published FP8 block scales and FP32
weight/input multipliers, including across module dtype casts.

The adapter uses ModelOpt's NVFP4 packing/dequantization reference. Its output
matches that reference with the supplied scales; it is a correctness-oriented
Python expert loop, not a fused training kernel. The forward pass chunks packing
work for large tensors; activations and quantized weights needed for backward
remain live. Export packs one complete projection at a time.

## Requirements and scope

Use a CUDA PyTorch environment with ModelOpt installed from this repository and
[NeMo AutoModel](https://github.com/NVIDIA-NeMo/Automodel) installed separately.
The integration targets AutoModel commit
`16670959b86f61dc662605c0df8e7de774b71aac`. AutoModel is an optional Apache-2.0
dependency; this example imports its implementation rather than vendoring it.

Supported: materialized BF16/FP16/FP32 master weights, bias-free unclamped SwiGLU,
expert dimensions divisible by 16, routing weights applied after the down
projection, torch expert loop, torch dispatcher, and expert parallel size 1.
Attach the adapter before FSDP2 wrapping, optimizer creation, or module hooks.

Grouped GEMM, Transformer Engine, expert parallelism, meta initialization, scale
learning, automatic calibration, and `torch.compile` are outside this example.
The adapter is not registered with `mtq.quantize` or `mto.restore`. Save and load
its ordinary PyTorch state, rebuilding the adapter before loading.

This is an expert-module integration, not a complete model-training or export
pipeline. It does not convert attention, shared experts, vision, PLE, or MTP;
it does not establish full-model accuracy improvement or serving compatibility.

## Attach to native experts

Configure AutoModel with its torch experts and dispatcher, and
`moe_overrides={"apply_router_weight_after_down": True}`. Load the floating-point
student weights first. For one expert module:

```python
from examples.llm_qat.nemo_automodel.nvfp4 import NVFP4GroupedExperts

# hf_scales is a tensor mapping read from the published safetensors checkpoint.
# Keys: "<HF module path>.<expert>.{gate,up,down}_proj.<scale name>"
# Scale names: weight_scale [out, in/16] FP8 E4M3,
#              weight_scale_2 and input_scale scalar FP32.
layer.mlp.experts = NVFP4GroupedExperts(
    layer.mlp.experts,
    hf_scales,
    prefix="model.language_model.layers.0.mlp.experts",
)
```

Each projection keeps its own input multiplier. The adapter shares the original
Parameter objects and clones the scales. The caller must ensure the supplied
scales correspond to the same master-weight checkpoint and projection ordering.

Wrap the resulting model in FSDP2 and train against the BF16 teacher with your
distillation loop. Retain FP32 master weights for small optimizer updates;
the native forward casts them to the activation dtype before quantization.
Save both model and optimizer state for exact resume. Construct the same
adapter and FSDP2 layout before restoring that state.

## Export one expert module

Gather full state collectively before exporting. Every rank must participate in
the gather; only rank 0 needs to pack and write the result:

```python
import torch
import torch.distributed as dist
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

from examples.llm_qat.nemo_automodel.nvfp4 import export_hf_experts

state = get_model_state_dict(
    model, options=StateDictOptions(full_state_dict=True, cpu_offload=True)
)
if dist.get_rank() == 0:
    expert_tensors = export_hf_experts(
        state,
        prefix="model.language_model.layers.0.mlp.experts",
        dtype=torch.bfloat16,  # Match the training activation dtype.
    )
```

The prefix must identify the adapter in the native state. Remap output keys if
the target HF module path differs. The exporter returns packed uint8 weights
and the original scales; it performs no collectives or device transfers.
Unrelated entries in the full-model state are ignored, including non-tensor
extra state and meta tensors outside the selected expert module.
Merge these tensors with the other model components and the appropriate
quantization configuration using a separately validated full-model exporter.

## Integration tests

With the optional AutoModel dependency installed, run from the repository root:

```bash
pytest tests/gpu/examples/llm_qat/test_nemo_automodel_nvfp4.py -v --durations=10
```

Tests cover packed parity, dtype preservation, routed and empty gradients,
rejection without modifying the source, and distributed FSDP2 Adam resume/export.
QDQ coverage includes the small and chunked paths, a partial final chunk,
contiguous and transposed tensors, fixed and dynamic block scales, and exact
straight-through gradients across all three supported compute dtypes.
The FSDP2 test uses all visible GPUs.

## Full-model experiment

A separate Qwen3.8-Flash-Next workspace experiment extended this adapter with a
streaming loader, mixed-format exporter, and serving-matched forward kernels with
recomputed backward passes. Those model-specific extensions are not included in
this example and are necessary to reproduce the following results.

In the initial pilot on 16 GB300 GPUs, BF16 and NVFP4 language-model parity passed
against vLLM 0.30 for five prompts of 256, 256, 500, 299, and 2048 tokens. All 97
captured boundaries matched exactly for each prompt. The serving comparison used TP1/EP1, isolated
uncached requests, chunked prefill, and pinned FlashInfer tactics. Zero-update
export/reload was exact; trained exports had maximum log-probability error below
6e-6 and top-1 agreement above 99.4%.

The subsequent run completed 100 QAD updates on 64 GB300 GPUs with a 16,384-token
context cap and global batch size 512. Only routed expert master weights were
trained; published scales and other model components remained frozen. The
mixed-format exporter preserved all frozen tensors, including vision weights.
Zero-update identity, checkpoint restoration, and trained-export serving
readback passed. Continuation stopped at update 160 after held-out KL exceeded
the run's quality limit; training was not resumed.

Matched BF16/PTQ/update-100 evaluations completed for GPQA-Diamond, IFBench,
MMMU-Pro, SciCode, AA-LCR, and Terminal-Bench 2.1. These comparisons have not
established a statistically clear QAD accuracy improvement. Lower distillation
loss alone does not establish task recovery. Vision forward was not exercised
during training.

The streaming loader/exporter, serving-parity kernels and bounded backward,
and full-model training/data/resume runner remain separate follow-up work.
This standalone adapter does not reproduce that pipeline by itself.
