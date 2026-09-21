# API guide: custom PyTorch to ComfyUI FP8 or NVFP4

The [`quantize.py`](./quantize.py) example loads a Hugging Face Diffusers pipeline and exports it with
`export_hf_checkpoint`. This note covers a different case: the checkpoint belongs to an arbitrary Python class,
which may contain several attached `torch.nn.Module` objects but is not itself an `nn.Module`, Hugging Face model,
or Diffusers pipeline.

Explicitly collect the components that should appear in the checkpoint into a temporary `nn.ModuleDict`. This is
only an export view; the original class does not need to change. Modules stored in ordinary Python lists or dicts
must also be added explicitly.

```python
import copy

import torch

import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq

owner = build_custom_object()
owner.load_checkpoint(torch.load("model.pt", map_location="cpu", weights_only=True))

modelopt_root = torch.nn.ModuleDict(
    {
        "denoiser": owner.denoiser,
        "text_encoder": owner.text_encoder,
        "vae": owner.vae,
    }
).cuda().eval()


def forward_loop(_):
    for sample in calibration_data:
        # Calling the owner's custom inference path is fine because it uses the same modules.
        # It must exercise every component selected for quantization.
        owner.run_calibration(sample)


quant_cfg = mtq.FP8_DEFAULT_CFG  # Or mtq.NVFP4_DEFAULT_CFG.
mtq.quantize(modelopt_root, copy.deepcopy(quant_cfg), forward_loop)

# Save a resumable ModelOpt checkpoint before the destructive deployment packing step.
mto.save(modelopt_root, "model_quantized_modelopt.pt")
```

This example quantizes the supported leaves in `modelopt_root` that match the preset. Narrow `quant_cfg` by
component prefix when only a subset should be quantized.

## Export without Hugging Face or Diffusers

There is currently no public one-call exporter for this shape of object. The minimal path is to call
`_process_quantized_modules` on the temporary root, hide ModelOpt quantizer state, adapt the tensor/config names to
the target ComfyUI loader, and write safetensors directly:

```python
import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from modelopt.torch.export.convert_hf_config import convert_hf_quant_config_format
from modelopt.torch.export.diffusers_utils import (
    build_layerwise_quant_metadata,
    hide_quantizers_from_state_dict,
    pad_nvfp4_weights,
    swizzle_nvfp4_scales,
)
from modelopt.torch.export.quant_utils import (
    get_quant_config,
    has_quantized_modules,
    sync_tied_input_amax,
)
from modelopt.torch.export.unified_export_hf import _process_quantized_modules


def export_comfyui_checkpoint(
    modelopt_root,
    output_path,
    adapt_for_loader,
    *,
    dtype=torch.bfloat16,
    padding_strategy=None,
    swizzle=False,
):
    if not has_quantized_modules(modelopt_root):
        raise ValueError("No ModelOpt-quantized modules found")

    modelopt_config = get_quant_config(modelopt_root)
    quant_config = convert_hf_quant_config_format(modelopt_config)
    quant_algo = quant_config.get("quant_algo")
    if quant_algo not in {"FP8", "NVFP4"}:
        raise ValueError(f"Expected FP8 or NVFP4, got {quant_algo!r}")

    sync_tied_input_amax(modelopt_root)
    _process_quantized_modules(modelopt_root, dtype)

    with hide_quantizers_from_state_dict(modelopt_root):
        state_dict = dict(modelopt_root.state_dict())

    # Rename/filter tensor keys, merge external tensors, and update any config layer names.
    state_dict, quant_config = adapt_for_loader(state_dict, quant_config)
    state_dict = {
        key: tensor.detach().to(device="cpu", copy=True).contiguous()
        for key, tensor in state_dict.items()
    }

    if quant_algo == "NVFP4":
        if padding_strategy is not None:
            pad_nvfp4_weights(state_dict, padding_strategy)
        if swizzle:
            swizzle_nvfp4_scales(state_dict)
    elif padding_strategy is not None or swizzle:
        raise ValueError("Padding and swizzling apply only to NVFP4")

    metadata = {
        "quantization_config": json.dumps(quant_config),
        "_quantization_metadata": build_layerwise_quant_metadata(state_dict, quant_config),
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(state_dict, str(output_path), metadata=metadata)
```

`adapt_for_loader` is model-specific: there is no universal ComfyUI key layout. Attached modules already included
in `modelopt_root` are present in its state dict; use the callback to merge only tensors outside that root. Preserve
the `.weight_scale` and `.weight_scale_2` suffixes, or build equivalent layer metadata yourself. If keys are renamed,
also rename layer references in `quant_config`, such as `ignore` or `quantized_layers`.

FP8 is the simpler case: it has `.weight`, `.weight_scale`, and usually `.input_scale`, but no
`.weight_scale_2`, padding, or scale swizzling. NVFP4 adds those optional runtime-specific steps.

Despite their source filenames, the calls above do not construct or require a Hugging Face or Diffusers object;
Diffusers imports in `diffusers_utils.py` are optional. This minimal helper targets `FP8_DEFAULT_CFG` and
`NVFP4_DEFAULT_CFG` on ModelOpt-supported quantizable leaves such as `nn.Linear`. A custom weight-bearing leaf needs
a ModelOpt conversion/export plugin. These helpers are private and version-sensitive, and packing mutates the
selected modules in place.
