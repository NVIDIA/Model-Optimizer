# Extending ModelOpt PTQ to Unsupported Models

When a model architecture is not in `MODEL_NAME_TO_TYPE` (in `modelopt/torch/export/model_utils.py`), a custom PTQ script is needed. This document describes the general patterns.

## Identifying What Needs Customization

Inspect the model's modeling code (in its HuggingFace `modeling_*.py`) and ask:

1. **Are all quantizable weights in `nn.Linear` layers?** If yes, `mtq.quantize()` auto-instruments them. If the model uses raw `nn.Parameter` tensors (common in MoE expert implementations), manual quantizer injection is needed.

2. **Is it a VLM?** If yes, extract the language model backbone and only quantize that. The vision tower and projector should be left in BF16.

3. **Is the checkpoint in FP8?** If yes, dequantize to BF16 before NVFP4 PTQ. Check `config.json` for `"quant_method": "fp8"`.

4. **Does it have MoE routing?** If yes, during calibration, sparse routing means some experts may never see data. A calibration wrapper that forces all-expert routing is needed.

## Pattern 1: Custom Module with TensorQuantizer

For modules that use raw `nn.Parameter` + `F.linear()` instead of `nn.Linear`, inject `TensorQuantizer` modules and apply them in the forward pass.

```python
from modelopt.torch.quantization.nn import TensorQuantizer

class QuantCustomModule(OriginalModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._setup()

    def _setup(self):
        # One pair per projection
        self.proj_a_input_quantizer = TensorQuantizer()
        self.proj_a_weight_quantizer = TensorQuantizer()
        self.proj_b_input_quantizer = TensorQuantizer()
        self.proj_b_weight_quantizer = TensorQuantizer()

    def forward(self, x, ...):
        # Apply quantizers around F.linear calls
        q_x = self.proj_a_input_quantizer(x)
        q_w = self.proj_a_weight_quantizer(self.weight_a)
        out = F.linear(q_x, q_w)
        # ... continue with proj_b ...
```

**Rules:**

- Method MUST be named `_setup` (ModelOpt's `mtq.register()` asserts this)
- Quantizer names MUST end with `_input_quantizer` or `_weight_quantizer` for wildcard matching
- The `__init__` must call `super().__init__()` then `self._setup()`

## Pattern 2: MoE Calibration Wrapper

MoE models route tokens to a subset of experts (top-k). During calibration, experts that receive no tokens won't have their quantization scales calibrated. Fix this with a wrapper that temporarily routes all tokens to all experts:

```python
class CalibMoE(OriginalMoE):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._setup()

    def _setup(self):
        self._original_top_k = self.top_k

    def forward(self, hidden_states):
        # First pass: all experts get calibration data
        self.top_k = self.num_experts
        super().forward(hidden_states)
        # Second pass: normal routing for actual output
        self.top_k = self._original_top_k
        return super().forward(hidden_states)
```

Adjust attribute names (`top_k`, `num_experts`, `topk_group`, `n_group`, etc.) to match the model's implementation. Read the model's MoE source code to find the correct names.

## Pattern 3: Registering with ModelOpt

Register all custom classes BEFORE calling `mtq.quantize()`:

```python
import modelopt.torch.quantization as mtq

mtq.register(original_cls=OriginalModule, quantized_cls=QuantCustomModule)
mtq.register(original_cls=OriginalMoE, quantized_cls=CalibMoE)
```

`mtq.register()` tells ModelOpt to replace all instances of `original_cls` with `quantized_cls` during quantization. The replacement class must be a subclass of the original.

## Pattern 4: VLM Language Model Extraction

For multimodal models, only quantize the language model backbone:

```python
from modelopt.torch.export.model_utils import get_language_model_from_vl, is_multimodal_model

if is_multimodal_model(model):
    lineage = get_language_model_from_vl(model)
    language_model = lineage[-1]

    # Disable quantization for non-language modules
    disabled_cfg = {"quant_cfg": {"default": {"enable": False}}, "algorithm": "max"}
    memo = set(lineage)
    for ancestor in lineage[:-1]:
        for _, child in ancestor.named_children():
            if child not in memo:
                mtq.quantize(child, disabled_cfg, forward_loop=None)
                memo.add(child)

    # Now quantize only language_model
    language_model = mtq.quantize(language_model, quant_cfg, forward_loop=forward_loop)
```

Also add safety overrides to the config:

```python
quant_cfg["quant_cfg"]["*vision*"] = {"enable": False}
quant_cfg["quant_cfg"]["*multi_modal_projector*"] = {"enable": False}
```

## Pattern 5: FP8 Checkpoint Dequantization

### Standard nn.Linear weights

HuggingFace handles these automatically with `dequantize=True`:

```python
from transformers.utils.quantization_config import FineGrainedFP8Config

model = AutoModel.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    quantization_config=FineGrainedFP8Config(dequantize=True),
)
```

### Non-standard parameter names (e.g., 3D expert weights)

HF's `WeightConverter` uses source patterns `["weight$", "weight_scale_inv", "activation_scale"]`. Parameters with names like `gate_up_proj`, `down_proj`, `w1`, `w2`, `w3` won't match these patterns and will remain in FP8 after loading. Dequantize them manually:

```python
def dequantize_fp8_params(model, param_names=("gate_up_proj", "down_proj")):
    """Dequantize remaining FP8 parameters that HF's WeightConverter missed."""
    count = 0
    for name, module in model.named_modules():
        for param_name in param_names:
            param = getattr(module, param_name, None)
            if not isinstance(param, torch.nn.Parameter) or param.dtype != torch.float8_e4m3fn:
                continue
            scale = getattr(module, f"{param_name}_scale_inv", None)
            if scale is None:
                param.data = param.data.to(torch.bfloat16)
            elif scale.dim() == 1:
                # Per-tensor scale
                param.data = param.data.to(torch.bfloat16) * scale.data[:, None, None].to(torch.bfloat16)
            elif scale.dim() == 3:
                # Per-block scale: reshape, broadcast, multiply
                w = param.data
                s = scale.data
                block_m = w.shape[-2] // s.shape[-2]
                block_n = w.shape[-1] // s.shape[-1]
                reshaped = w.to(torch.bfloat16).reshape(-1, s.shape[-2], block_m, s.shape[-1], block_n)
                scaled = reshaped * s.to(torch.bfloat16).unsqueeze(-1).unsqueeze(2)
                param.data = scaled.reshape(w.shape)
            else:
                param.data = param.data.to(torch.bfloat16)
            count += 1
    if count:
        print(f"Dequantized {count} FP8 parameters to BF16.")
```

Adapt `param_names` to match the model's actual parameter naming convention. Inspect the model's `modeling_*.py` and `config.json` to find the right names.

## Pattern 6: Custom Quantization Config

When stock configs don't match the model's module naming:

```python
import copy
import modelopt.torch.quantization as mtq

# Start from a stock config
cfg = copy.deepcopy(mtq.NVFP4_MLP_ONLY_CFG)

# Add patterns for custom module names
cfg["quant_cfg"]["*custom_experts*weight_quantizer"] = {
    "num_bits": (2, 1),
    "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)},
    "enable": True,
}
cfg["quant_cfg"]["*custom_experts*input_quantizer"] = {
    "num_bits": (2, 1),
    "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)},
    "enable": True,
}

# Verify wildcards target the right modules
# After quantization, always run:
mtq.print_quant_summary(model)
```

## General Custom PTQ Script Structure

```python
import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq
from modelopt.torch.export import export_hf_checkpoint

mto.enable_huggingface_checkpointing()

# 1. Load model (with FP8 dequant if needed)
model = load_and_dequantize(model_path)

# 2. Register monkey-patched modules
mtq.register(original_cls=..., quantized_cls=...)

# 3. Calibrate and quantize
dataloader = get_dataset_dataloader(dataset_name=["cnn_dailymail"], tokenizer=tokenizer, ...)
def forward_loop(model):
    for batch in dataloader:
        model(**batch)

model = mtq.quantize(model, quant_cfg, forward_loop=forward_loop)
mtq.print_quant_summary(model)

# 4. Export
export_hf_checkpoint(model, export_dir=output_path)
tokenizer.save_pretrained(output_path)
```

## Debugging Tips

- **Smoke test first**: Run with `--calib_size 4` to verify the pipeline end-to-end before full calibration
- **Check quantizer summary**: `mtq.print_quant_summary(model)` shows which quantizers are enabled/disabled
- **Inspect dtypes**: After loading, iterate `model.named_parameters()` and check for unexpected FP8 tensors
- **Watch for silent disabling**: A misconfigured wildcard pattern can silently disable quantizers — always verify the summary
