# Handling Unlisted Models

The model is not in the verified support table (`examples/llm_ptq/README.md`). This does NOT mean it won't work — ModelOpt auto-detects standard HF modules (linear layers, attention, MoE blocks with `gate`+`experts`). Many unlisted models work with `hf_ptq.py` out of the box.

Follow the investigation steps below to determine if `hf_ptq.py` works or if patches are needed.

## Step A — Locate the model source

**Is it a HuggingFace checkpoint?** Check for `config.json`. If present, try loading:

```bash
python -c "
from transformers import AutoConfig
cfg = AutoConfig.from_pretrained('<ckpt_path>')
print(type(cfg).__name__)
"
```

- **Succeeds** → transformers knows the architecture. Find the source file:

  ```bash
  python -c "
  import importlib, inspect
  from transformers import AutoConfig
  cfg = AutoConfig.from_pretrained('<ckpt_path>')
  mod_name = 'transformers.models.' + cfg.model_type.replace('-', '_')
  mod = importlib.import_module(mod_name + '.modeling_' + cfg.model_type.replace('-', '_'))
  print(inspect.getfile(mod))
  "
  ```

  Read the modeling file and proceed to Step B.

- **Raises `ValueError` / `OSError` (unknown architecture)** → not in the installed transformers. Determine why:

  1. **Search the working directory** for the class — a local fork or custom modeling file may already be present. If found, add its path to `sys.path`.

  2. **Check the transformers `main` branch** (not yet released):

     ```bash
     git clone --depth 1 https://github.com/huggingface/transformers.git /tmp/transformers-main --quiet
     grep -r "class <ArchName>" /tmp/transformers-main/src/transformers/models/
     ```

     - **Found** → install from that clone: `pip install /tmp/transformers-main --quiet`, then re-run `AutoConfig.from_pretrained()`.
     - **Not found** → ask the user: *"The checkpoint uses `<ArchName>` which isn't in released or main-branch transformers. Do you have a private fork or custom modeling code?"*

- **No `config.json`** → not a standard HF checkpoint. List the directory for README or `.py` files. If nothing useful, ask the user for the modeling code.

## Step B — Is the checkpoint already FP8-quantized?

Check `config.json` for `"quantization_config"` or scan weight files for `*_scale_inv*` tensors. If found, the model must be dequantized before re-quantizing. HuggingFace's `WeightConverter` only handles standard `weight` / `weight_scale_inv` names and will silently miss non-standard parameter names (e.g., 3D expert tensors in MoE layers). See **Pattern 5** below.

## Step C — Determine what custom patches are needed

Read the model source to identify how weights are stored. **If all linear layers are plain `nn.Linear`, no custom code is needed** — ModelOpt quantizes them automatically.

**For HuggingFace models**, check `modelopt/torch/quantization/plugins/huggingface.py` first — it already registers patches for common non-standard modules (`Llama4TextExperts`, `FP8Linear`, `FalconLinear`, `Conv1D`, `Qwen3_5MoeExperts`, etc.). If your model's non-standard class is already registered there, no extra code is needed.

Custom patches are required when:

- **Fused/batched expert weights** — experts stored as a single parameter (e.g., 3D `[num_experts, in, out]`) rather than separate `nn.Linear` modules → Pattern 1 + 2
- **Self-defined weight parameters** (`nn.Parameter` used directly instead of `nn.Linear`) — common in non-HF or research models → Pattern 1 + 3
- **VLM structure** (vision encoder that should be excluded) → Pattern 4
- **FP8 checkpoint** that needs dequantization before re-quantizing → Pattern 5

## Step D — Check weight names against ModelOpt's config patterns

Scan actual parameter names in the checkpoint and compare them against the wildcard patterns in the chosen quant config (`modelopt/torch/quantization/config.py`). If a module has a weight with a non-standard name (e.g., `gate_up_proj` instead of `gate_proj`/`up_proj`, or `experts.w1` instead of `experts.*.w1`), the wildcard will silently miss it.

```python
import json
idx = json.load(open('<ckpt_path>/model.safetensors.index.json'))
import re
names = set(re.sub(r'\.\d+\.', '.N.', k) for k in idx['weight_map'])
for n in sorted(names): print(n)
```

Compare against the `enable`/`disable` patterns in the config. Add custom overrides using Pattern 6 if needed. Always verify with `mtq.print_quant_summary(model)` after quantization.

## Step E — Run and iterate

After Steps A-D:

- **No patches needed** (all standard modules) → run `hf_ptq.py` with a smoke test (`--calib_size 4`). If it succeeds, proceed with full calibration. If it fails, read the error and revisit Steps C/D.
- **Patches needed** → write a custom script using the patterns below, then smoke test it. Debug failures iteratively — quantization errors often reveal additional modules that need patching.

---

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

## Pattern 2: MoE Models

MoE models route tokens to a subset of experts (top-k). During calibration, experts that receive no tokens won't have their quantization scales calibrated.

**In most cases, ModelOpt handles this automatically.** The HuggingFace plugin auto-detects MoE blocks with the standard `gate` + `experts` pattern (`register_sparse_moe_on_the_fly`) and registers `_QuantSparseMoe`, which:

- Syncs input quantizer amax across all experts after calibration (`layer_sync_moe_local_experts_amax`)
- Runs weight-only calibration for experts that received no tokens (`sync_moe_expert_amax`)

**When you still need custom work:**

- **Fused expert weights** (e.g., Qwen3.5's `Qwen3_5MoeExperts`, Step3.5's `MoELinear`): These store all expert weights in a single tensor `[num_experts, out, in]` instead of separate `nn.Linear` per expert. Use Pattern 1 to create a `QuantModule` that expands fused weights into per-expert modules with their own quantizers.

- **Non-standard MoE structure** (no `gate`/`experts` attributes): Auto-detection won't find it. Either add `layer_sync_moe_local_experts_amax` to your custom class, or call `sync_moe_expert_amax` manually after quantization:

```python
from modelopt.torch.quantization.utils import sync_moe_expert_amax

mtq.quantize(model, config, forward_loop)
for name, module in model.named_modules():
    if hasattr(module, 'experts'):  # adjust to match the model
        sync_moe_expert_amax(module.experts)
```

## Pattern 3: Registering with ModelOpt

Register all custom classes BEFORE calling `mtq.quantize()`:

```python
import modelopt.torch.quantization as mtq

mtq.register(original_cls=OriginalModule, quantized_cls=QuantCustomModule)
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
