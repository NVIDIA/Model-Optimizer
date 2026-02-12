# Puzzletron DeciLM: MBridge and NeMo Workflows

This document explains both MBridge and NeMo workflows for Puzzletron DeciLM models.

## Implementation Comparison

**Which was easier to implement?**

**MBridge was significantly easier** to implement for the following reasons:

### MBridge Implementation (Simpler)

- **~242 lines of code** in `puzzletron_decilm_bridge.py`
- **No field name conversions**: Just serializes entire HF config as JSON (`hf_config.to_json_string()`)
- **Reuses existing infrastructure**: Uses `LlamaNemotronHeterogeneousProvider` - no custom provider needed
- **Simple weight mappings**: Uses same wildcard patterns as `LlamaNemotronBridge` (QKV/MLP concatenation)
- **No custom model class**: Works directly with MCore's `GPTModel`
- **Key insight**: MBridge stores HF config as-is in JSON, letting MCore parse it later

### NeMo Implementation (More Complex)

- **~967 lines** in `llama_nemotron.py` + **~377+ lines** in `llama_nemotron_utils.py`
- **Field name conversions required**: Converts each `block_configs[i]` from HF → NeMo format:
  - `n_heads_in_group` → `num_query_groups`
  - `intermediate_size` → `ffn_hidden_size`
  - `ffn` → `mlp`
  - etc.
- **Custom model class**: Requires `PuzzletronLlamaNemotronModel` wrapper
- **Complex weight mappings**: Per-layer mappings for heterogeneous layers (Mamba, MoE, attention)
- **Helper functions**: `convert_attention_config_from_cfg_object()`, `convert_mlp_config_from_cfg_object()`, etc.

**Conclusion**: MBridge's approach of serializing the HF config as JSON and letting MCore handle parsing is much simpler than NeMo's approach of converting field names upfront.

## 1. MBridge/MCore Heterogeneous Model Support

**Question**: Does MBridge/MCore support heterogeneous models at inference?

**Answer**: Yes! MBridge can create heterogeneous `GPTModel` instances in MCore, and MCore supports inference of such models.

**How it works**:

1. **Bridge creates heterogeneous provider**: `PuzzletronDeciLMBridge.provider_bridge()` creates `LlamaNemotronHeterogeneousProvider` with:
   - `heterogeneous_layers_config_encoded_json`: JSON string containing HF config (with `block_configs`)
   - `transformer_layer_spec = heterogeneous_layer_spec`: Callable that returns MCore layer specs

2. **MCore builds heterogeneous model**: When MBridge instantiates the MCore `GPTModel`:
   - `LlamaNemotronHeterogeneousProvider.transformer_layer_spec` calls `get_gpt_heterogeneous_layer_spec()` from MCore
   - This function parses `heterogeneous_layers_config_encoded_json` and creates `ModuleSpec` objects per layer
   - `GPTModel.__init__()` uses these specs to instantiate PyTorch modules (different types per layer)

3. **Inference**: The built MCore `GPTModel` instance handles forward passes:
   - Each layer executes according to its architecture (Mamba, MoE, attention, etc.)
   - MCore's heterogeneous support is built into `GPTModel` - no special wrapper needed

**Key difference from NeMo**:

- **NeMo**: Uses `PuzzletronLlamaNemotronModel` wrapper class (extends NeMo's `GPTModel`)
- **MBridge/MCore**: Works directly with MCore's `GPTModel` (no wrapper needed)
- Both use the same underlying MCore heterogeneous layer building mechanism

## 2. HF → MBridge Conversion

**Script**: `puzzletron_decilm_bridge.py` (bridge implementation)

**Process**:

- Loads HF checkpoint (`DeciLMForCausalLM` + `DeciLMConfig` with `block_configs`)
- `PuzzletronDeciLMBridge.provider_bridge()` performs conversion:
  - **Config Conversion**: `DeciLMConfig` → `LlamaNemotronHeterogeneousProvider`
    - Extracts global config values (e.g., `num_layers`, `hidden_size`, `num_attention_heads`)
    - Extracts placeholder values from first valid block (e.g., `num_query_groups`, `ffn_hidden_size`)
    - Serializes entire HF config to `heterogeneous_layers_config_encoded_json` (JSON string in **HF format**)
    - Creates `LlamaNemotronHeterogeneousProvider` with heterogeneous layer support
  - **Weight Conversion**: Uses `mapping_registry()` to define HF → Megatron weight mappings
    - Standard mappings (embedding, output, layernorm, etc.)
    - Special mappings for QKV concatenation and gated MLP concatenation
- `AutoBridge.import_ckpt()` converts and saves as MBridge checkpoint:
  - Saves config and weights in `iter_XXXXXXX/` directory structure
  - Result: MBridge checkpoint ready for loading with `load_megatron_model()`

**Example**:

```python
from megatron.bridge import AutoBridge

# Convert HF checkpoint to MBridge format
AutoBridge.import_ckpt(
    hf_model_path="/path/to/decilm/checkpoint",
    megatron_path="/path/to/mbridge/checkpoint",
    dtype="bfloat16",
    trust_remote_code=True,  # Required for DeciLM models
)
```

## 3. MBridge Model Loading

**When**: Loading MBridge checkpoint for inference/serving

**Process**:

- `load_megatron_model()` loads the MBridge checkpoint:
  - Loads config from checkpoint metadata
  - Creates `LlamaNemotronHeterogeneousProvider` with `heterogeneous_layers_config_encoded_json`
  - Initializes model parallelism (TP/PP) from config or `mp_overrides`
  - `provider.finalize()` parses JSON config into per-layer parameters
  - `GPTModel.__init__()` uses `transformer_layer_spec` to build heterogeneous layers:
    - Parses `heterogeneous_layers_config_encoded_json` → creates `ModuleSpec` objects per layer
    - Instantiates PyTorch modules (Mamba, MoE, standard attention, etc.) based on specs
- Result: MCore `GPTModel` instance with heterogeneous architecture

**Key**:

- `GPTModel` is the model class used at inference (no wrapper needed)
- Specs are used only during model instantiation, not during forward passes

**Example**:

```python
from megatron.bridge.training.model_load_save import load_megatron_model

# Load MBridge checkpoint
# NOTE: Path must point to the iter_XXXXXXX/ directory, not the parent
model = load_megatron_model(
    checkpoint_path="/path/to/mbridge/checkpoint/iter_0000000",
    mp_overrides={
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 1,
    },
)
model = [m.cuda() for m in model]
for m in model:
    m.eval()
```

## 4. MBridge Inference

**When**: Forward passes during inference/serving

**Process**:

- Model instance is MCore `GPTModel` (loaded from MBridge checkpoint)
- Uses the built PyTorch modules (already instantiated from step 3)
- Specs are **not used** - only the actual PyTorch modules handle forward passes
- Each layer executes according to its architecture (Mamba, MoE, attention, etc.)

**Example**:

```python
import torch

# Prepare inputs
tokens = torch.tensor([[1, 2, 3, 4]])  # Token IDs
position_ids = torch.arange(tokens.shape[1]).unsqueeze(0)
attention_mask = torch.ones_like(tokens)

# Forward pass
with torch.no_grad():
    logits = model[0](
        tokens.cuda(),
        position_ids.cuda(),
        attention_mask.cuda(),
    )
```

---

## 5. HF → NeMo Conversion

**Script**: `convert_puzzletron_hf_to_nemo_with_api.py`

**Process**:

- Loads HF checkpoint (`DeciLMForCausalLM` + `DeciLMConfig` with `block_configs`)
- `PuzzletronHFLlamaNemotronImporter` performs two conversions:
  - **Config Conversion**: `DeciLMConfig` → `PuzzletronNemotronModelConfig`
    - Converts each `block_configs[i]` from HF field names → MCore field names:
      - `n_heads_in_group` → `num_query_groups`
      - `intermediate_size` → `ffn_hidden_size`
      - etc.
    - Serializes to `heterogeneous_layers_config_encoded_json` (JSON string in **MCore format**)
    - Saves in `context/model.yaml` as part of `PuzzletronNemotronModelConfig`
  - **Weight Conversion**: Maps HF weight names → NeMo weight names (with QKV/MLP concatenation)
    - Saves weights in `weights/` directory
- Result: NeMo checkpoint with config in `context/model.yaml` and weights in `weights/`

## 6. NeMo Model Loading

**When**: Loading NeMo checkpoint for inference/serving

**Process**:

- NeMo loads the checkpoint as `PuzzletronLlamaNemotronModel` (the model class)
- Loads `PuzzletronNemotronModelConfig` from `context/model.yaml`
  - Contains `heterogeneous_layers_config_encoded_json` (JSON string in **MCore format**)
  - Contains `transformer_layer_spec = heterogeneous_layer_spec_puzzletron` (callable)
- `PuzzletronLlamaNemotronModel.__init__()` calls `GPTModel.__init__()` (parent class)
- `puzzletron_layer_specs.py` converts config → MCore specs:
  - Parses `heterogeneous_layers_config_encoded_json` (MCore format) → creates `PuzzletronTransformerBlockConfig` dataclasses
  - `get_gpt_heterogeneous_layer_spec_puzzletron()` → calls `get_layer_spec_for_layer()` for each block
  - Returns `TransformerBlockSubmodules` containing `ModuleSpec` objects
- `GPTModel.__init__()` uses specs to instantiate PyTorch modules (Mamba, MoE, standard attention, etc.)

**Key**:

- `PuzzletronLlamaNemotronModel` is the model class used at inference (wraps NeMo's `GPTModel`)
- Specs are used only during model instantiation, not during forward passes

## 7. NeMo Inference

**When**: Forward passes during inference/serving

**Process**:

- Model instance is `PuzzletronLlamaNemotronModel` (loaded from NeMo checkpoint)
- Uses the built PyTorch modules (already instantiated from step 6)
- Specs are **not used** - only the actual PyTorch modules handle forward passes
- Each layer executes according to its architecture (Mamba, MoE, attention, etc.)
