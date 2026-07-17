# API guide: custom PyTorch to ComfyUI FP8 or NVFP4

The [`quantize.py`](./quantize.py) example loads a Hugging Face Diffusers pipeline and exports it with
`export_hf_checkpoint`. This note instead assumes that the model remains a plain PyTorch `torch.nn.Module`: it
does not inherit from Diffusers and its source checkpoint is a regular torch state dict.

For a custom checkpoint, first instantiate its exact model class, load the state dict, and calibrate it:

```python
import copy

import torch

import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq

model = build_model()
state_dict = torch.load("model.pt", map_location="cpu", weights_only=True)
model.load_state_dict(state_dict, strict=True)
model.cuda().eval()


def forward_loop(model):
    for sample in calibration_data:
        model(**sample)


quant_cfg = mtq.NVFP4_DEFAULT_CFG  # Use mtq.FP8_DEFAULT_CFG for FP8.
mtq.quantize(model, copy.deepcopy(quant_cfg), forward_loop)

# Optional: save a resumable ModelOpt checkpoint before deployment packing.
mto.save(model, "model_quantized_modelopt.pt")
```

## Which export API to call

| Model object | API |
| --- | --- |
| A Diffusers `ModelMixin` or `DiffusionPipeline` | Public `modelopt.torch.export.export_hf_checkpoint` |
| A plain `torch.nn.Module` | Internal `_export_diffusers_checkpoint` shown below |

The internal Diffusers exporter already accepts a plain `nn.Module` and writes packed deployment tensors to
`model.safetensors`. This is an intermediate packed checkpoint; the adapter in the next section is what makes it
compatible with a particular ComfyUI loader.

```python
from pathlib import Path

from modelopt.torch.export.unified_export_hf import _export_diffusers_checkpoint

_export_diffusers_checkpoint(
    pipe=model,
    dtype=torch.bfloat16,
    export_dir=Path("packed_model"),
    components=None,
    max_shard_size="1000GB",  # Keep the deployment checkpoint in one file.
)
```

This is a private, version-sensitive API. It currently requires the `diffusers` package to be installed, although
the model itself remains a plain `nn.Module`. It packs recognized quantized modules in place, so call it on a
disposable model instance after `mto.save`. To avoid the Diffusers dependency, build the same staging file by
calling `_process_quantized_modules` and `hide_quantizers_from_state_dict` directly.

For FP8, the export call is unchanged. FP8 does not use the NVFP4 `weight_scale_2`, padding, or scale-swizzle
steps, so only the loader-specific key mapping, optional component merge, and metadata steps below remain.

## Make the packed file ComfyUI-specific

There is no universal ComfyUI key layout. After packing, add a small adapter for the exact ComfyUI loader or
custom node that will consume the model:

1. Rename every tensor key to the loader's expected namespace. Rename `.weight`, `.weight_scale`,
   `.input_scale`, bias, and NVFP4 `.weight_scale_2` keys consistently.
2. If the loader expects a full checkpoint, merge the required VAE, text encoder, or other tensors from a base
   safetensors file.
3. For NVFP4 only, optionally call `pad_nvfp4_weights`, followed by `swizzle_nvfp4_scales`, when the target
   runtime expects the cuBLAS/comfy_kitchen scale layout. Skip both for FP8.
4. Read `quantization_config` from the generated `packed_model/config.json`. After the final key mapping, call
   `build_layerwise_quant_metadata` and write both values into the single-file safetensors header.

These post-processing helpers live in
[`modelopt/torch/export/diffusers_utils.py`](../../../modelopt/torch/export/diffusers_utils.py). The existing LTX-2
implementation, `_merge_ltx2`, is a useful example, but its `model.diffusion_model.` prefix and component merge
rules are LTX-2-specific and should not be copied blindly.

No Diffusers wrapper is required for this path. Until ModelOpt exposes a public plain-module deployment exporter,
pin the ModelOpt revision when using `_export_diffusers_checkpoint` or promote that path to a public API.
