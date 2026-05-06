# Handling Unlisted Models

The model is not in the verified support table (`examples/llm_ptq/README.md`). This does NOT mean it won't work — ModelOpt auto-detects standard HF modules (linear layers, attention, MoE blocks with `gate`+`experts`). Many unlisted models work with `hf_ptq.py` out of the box.

Follow the investigation steps below to determine if `hf_ptq.py` works or if patches are needed.

## Step A — Download the model and locate the source

**Download first.** Follow `skills/common/workspace-management.md` to set up local and remote workspaces, sync ModelOpt source, and download the model on the target machine. This avoids downloading twice and gives access to README, custom modeling code, and tokenizer config.

After download, inspect the model files on the target machine (use `remote_run` if remote):

1. **Read `README.md`** — often lists required transformers versions, dependencies, or `trust_remote_code` requirements
2. **Check for `modeling_*.py` or `tokenization_*.py`** — custom code shipped with the model. If found, **always use `--trust_remote_code`** with `hf_ptq.py`, and `trust_remote_code=True` in any custom scripts. Without it, `AutoConfig`, `AutoTokenizer`, and `AutoModel` will fail to resolve custom classes.

Write custom scripts locally (in `./workspaces/<model>/scripts/`), then sync to remote before running.

**Check transformers compatibility** (on the target machine):

First, if README or `config.json` specifies a required transformers version, check if installed version satisfies it. If not, upgrade: `pip install -U "transformers>=<required_version>"`.

Then try loading:

```bash
python -c "
from transformers import AutoConfig
cfg = AutoConfig.from_pretrained('<workspace>/model', trust_remote_code=True)
print(type(cfg).__name__)
"
```

- **Succeeds** → transformers knows the architecture. Find the source file:

  ```bash
  python -c "
  import importlib, inspect
  from transformers import AutoConfig
  cfg = AutoConfig.from_pretrained('<workspace>/model', trust_remote_code=True)
  mod_name = 'transformers.models.' + cfg.model_type.replace('-', '_')
  mod = importlib.import_module(mod_name + '.modeling_' + cfg.model_type.replace('-', '_'))
  print(inspect.getfile(mod))
  "
  ```

  Read the modeling file and proceed to Step B.

- **Raises `ValueError` / `OSError` (unknown architecture)** → not in the installed transformers. Try `pip install -U transformers` first. If still not found, check the `main` branch:

     ```bash
     git clone --depth 1 https://github.com/huggingface/transformers.git /tmp/transformers-main --quiet
     grep -r "class <ArchName>" /tmp/transformers-main/src/transformers/models/
     ```

     - **Found** → `pip install /tmp/transformers-main`, then re-run `AutoConfig`.
     - **Not found** → ask the user: *"The checkpoint uses `<ArchName>` which isn't in released or main-branch transformers. Do you have a private fork or custom modeling code?"*

- **No `config.json`** → not a standard HF checkpoint. List the directory for README or `.py` files. If nothing useful, ask the user for the modeling code.

## Step B — Is the checkpoint already FP8-quantized?

Check `config.json` for `"quantization_config"` with `"quant_method": "fp8"`, or scan weight files for `*_scale_inv*` tensors. If the model uses standard `FP8Linear` modules (2D weights with `weight` + `weight_scale_inv`), ModelOpt's `_QuantFP8Linear` plugin handles them automatically — no manual dequantization needed. The plugin keeps weights in FP8 and dequantizes lazily during calibration, which is memory-efficient.

Manual dequantization is only needed for **non-standard parameter names** (e.g., 3D expert tensors in MoE layers) that the plugin doesn't cover. See **Pattern 5** below.

## Step C — Determine what custom patches are needed

Read the model source to identify how weights are stored. **If all linear layers are plain `nn.Linear`, no custom code is needed** — ModelOpt quantizes them automatically.

**For HuggingFace models**, check `modelopt/torch/quantization/plugins/huggingface.py` first — it already registers patches for common non-standard modules (`Llama4TextExperts`, `FP8Linear`, `FalconLinear`, `Conv1D`, `Qwen3_5MoeExperts`, etc.). If your model's non-standard class is already registered there, no extra code is needed.

Custom patches are required when:

- **Fused/batched expert weights** — experts stored as a single parameter (e.g., 3D `[num_experts, in, out]`) rather than separate `nn.Linear` modules → Pattern 1 + 3
- **Self-defined weight parameters** (`nn.Parameter` used directly instead of `nn.Linear`) — common in non-HF or research models → Pattern 1 + 3
- **VLM structure** (vision encoder that should be excluded) → Pattern 4
- **FP8 checkpoint with non-standard parameter names** (standard `FP8Linear` is handled automatically by the `_QuantFP8Linear` plugin) → Pattern 5

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
- **Patches needed** → patch ModelOpt directly using the patterns below (add `QuantModule` in `modelopt/torch/quantization/plugins/huggingface.py`, update `modelopt/torch/export/` if needed), then run `hf_ptq.py` with a smoke test. This is preferred over writing a standalone script because it reuses all existing `hf_ptq.py` logic. Debug failures iteratively — quantization errors often reveal additional modules that need patching.

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

**Most MoE models are auto-detected** — ModelOpt handles two common patterns automatically:

- **transformers >= 5.0**: Unified fused experts (`gate_up_proj` + `down_proj` 3D tensors) → auto-detected by `register_fused_experts_on_the_fly`, handled by `_QuantFusedExperts`. Covers Mixtral, Qwen, DeepSeek, Jamba, OlMoE, etc.
- **transformers < 5.0**: Sequential per-expert `nn.Linear` with `gate` + `experts` → auto-detected by `register_sparse_moe_on_the_fly`.

**Custom MoE** (non-standard layout not matching auto-detection) requires patching. Find the closest pattern in the plugin (`modelopt/torch/quantization/plugins/huggingface.py`):

| MoE design | Strategy | Plugin example |
| --- | --- | --- |
| Fused weights + `torch.bmm` | Add `TensorQuantizer` around bmm | `_QuantLlama4TextExperts` |
| Fused weights + functional interception | Intercept matmul ops | `_QuantGptOssExperts` |
| Fused 2D weights (experts stacked in rows) | Two-level expansion | `_QuantDbrxExpertGLU` |
| Fused weights + `forward(x, expert_id)` | Expand + reconstruct on export | `_QuantMoELinear` (Step3.5) |

For the full guide, see `examples/llm_ptq/moe.md`.

**Critical: always check the weight layout.** `nn.Linear` expects `(out_features, in_features)` — the last dimension must be `in_features`. If the fused tensor is `(num_experts, in_dim, out_dim)`, you must transpose (`.T`) when copying. Getting this wrong silently corrupts quantization scales. Inspect the original forward pass to determine which dimension is which.

For non-standard MoE structures (no `gate`/`experts` attributes), auto-detection won't find the outer block. Call `sync_moe_expert_amax` manually after quantization:

```python
from modelopt.torch.quantization.utils import sync_moe_expert_amax

mtq.quantize(model, config, forward_loop)
for name, module in model.named_modules():
    if hasattr(module, 'experts'):  # adjust to match the model
        sync_moe_expert_amax(module.experts)
```

## Pattern 3: Registering with ModelOpt

**When patching the plugin directly** (preferred): Use `QuantModuleRegistry.register` in `modelopt/torch/quantization/plugins/huggingface.py`, following existing examples:

```python
from modelopt.torch.quantization.nn import QuantModuleRegistry

# Static registration (class available at import time):
QuantModuleRegistry.register({OriginalModule: "hf.OriginalModule"})(QuantCustomModule)

# Dynamic registration (trust_remote_code, class only available at runtime):
def register_my_model_on_the_fly(model):
    for module in model.modules():
        if type(module).__name__ == "OriginalModule":
            mod_type = type(module)
            if QuantModuleRegistry.get(mod_type) is None:
                QuantModuleRegistry.register({mod_type: f"hf.{mod_type.__name__}"})(QuantCustomModule)
            break
```

**When writing a standalone script** (fallback): Use `mtq.register()`:

```python
import modelopt.torch.quantization as mtq
mtq.register(original_cls=OriginalModule, quantized_cls=QuantCustomModule)
```

Both methods replace all instances of `original_cls` with `quantized_cls` during quantization. The replacement class must be a subclass of the original.

## Pattern 4: VLM Language Model Extraction

**Note**: `hf_ptq.py` already handles VLMs automatically via `extract_and_prepare_language_model_from_vl()`. It detects multimodal models, extracts the language backbone, and disables quantization for vision/projector modules. This works for most VLMs (tested with Mistral3/Devstral, Nemotron VL, Llama VL, etc.) — try `hf_ptq.py` first before writing custom VLM handling.

For custom scripts or when `hf_ptq.py` doesn't handle the VLM correctly, only quantize the language model backbone:

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

**Known VLM export issue**: The export step (`requantize_resmooth_fused_llm_layers` in `unified_export_hf.py`) may try to run a dummy forward pass on the full VLM instead of the language model backbone. This currently only handles Nemotron VLMs. If hit, patch the export to use `is_multimodal_model()` for the VLM check instead of model-specific string matching.

## Pattern 5: FP8 Checkpoint Handling

### Standard FP8Linear modules (preferred — no action needed)

ModelOpt's `_QuantFP8Linear` plugin (`modelopt/torch/quantization/plugins/huggingface.py`) automatically handles HuggingFace `FP8Linear` modules. It:

1. Keeps weights **compact in FP8** in GPU memory during calibration
2. **Dequantizes lazily** on-the-fly during calibration forward passes via `weight_dequant()`
3. Has `unpack_weight()` for full dequantization at export time

This is registered automatically for `transformers.integrations.finegrained_fp8.FP8Linear`. It requires **Triton** to be installed (used internally for FP8 dequantization kernels). Just load the model normally — no `FineGrainedFP8Config(dequantize=True)` needed:

```python
model = AutoModel.from_pretrained(model_path, device_map="auto", torch_dtype="auto")
# FP8Linear modules stay in FP8 → _QuantFP8Linear handles dequant during calibration
```

**Do NOT use `FineGrainedFP8Config(dequantize=True)`** — it expands the entire model to BF16 upfront, wasting ~2x GPU memory. The plugin approach is both more memory-efficient and simpler.

### Non-standard parameter names (e.g., 3D expert weights)

The `_QuantFP8Linear` plugin only handles standard 2D `FP8Linear` modules with `weight` + `weight_scale_inv`. Parameters with non-standard names (e.g., `gate_up_proj`, `down_proj`, `w1`/`w2`/`w3` in fused MoE experts) won't be covered. For these, dequantize manually after loading:

```python
def dequantize_fp8_params(model, param_names=("gate_up_proj", "down_proj")):
    """Dequantize remaining FP8 parameters that the plugin doesn't cover."""
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
                param.data = param.data.to(torch.bfloat16) * scale.data[:, None, None].to(torch.bfloat16)
            elif scale.dim() == 3:
                w = param.data
                s = scale.data
                assert w.shape[-2] % s.shape[-2] == 0 and w.shape[-1] % s.shape[-1] == 0, (
                    f"Incompatible FP8 scale shape: weight={tuple(w.shape)}, scale={tuple(s.shape)}")
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

## Fallback: Custom PTQ Script

Only if patching ModelOpt is not feasible (e.g., the model is not a standard transformer and `hf_ptq.py` fundamentally won't work):

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
- **Validate quantization pattern after export**: Run the validation script from SKILL.md Step 5 on the exported checkpoint. It checks every linear layer is either quantized (has scale params) or explicitly excluded. Layers that are neither were silently skipped — common for models with non-standard naming (e.g., Gemma4 `experts.*` missed by `*mlp*` patterns). This causes deployment failures when the framework tries to load BF16 weights as quantized
- **Read pip errors carefully**: `ResolutionImpossible` means dependency conflict (try `--no-deps`), NOT network failure. Check for `Connection refused`/`Name resolution failed` before concluding network is down
