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
Merge these tensors with the other model components and the appropriate
quantization configuration using a separately validated full-model exporter.

## Integration tests

With the optional AutoModel dependency installed, run from the repository root:

```bash
pytest tests/gpu/examples/llm_qat/test_nemo_automodel_nvfp4.py -v --durations=10
```

Tests cover packed parity, dtype preservation, routed and empty gradients,
rejection without modifying the source, and distributed FSDP2 Adam resume/export.
The FSDP2 test uses all visible GPUs.
