# Puzzletron DeciLM: HF → NeMo → Inference

## 1. HF → NeMo Conversion

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

## 2. NeMo Model Loading

**When**: Loading NeMo checkpoint for inference/serving

**Process**:

- Loads `PuzzletronNemotronModelConfig` from `context/model.yaml`
  - Contains `heterogeneous_layers_config_encoded_json` (JSON string in **MCore format**)
  - Contains `transformer_layer_spec = heterogeneous_layer_spec_puzzletron` (callable)
- `puzzletron_layer_specs.py` converts config → MCore specs:
  - Parses `heterogeneous_layers_config_encoded_json` (MCore format) → creates `PuzzletronTransformerBlockConfig` dataclasses
  - `get_gpt_heterogeneous_layer_spec_puzzletron()` → calls `get_layer_spec_for_layer()` for each block
  - Returns `TransformerBlockSubmodules` containing `ModuleSpec` objects
- `GPTModel.__init__()` uses specs to instantiate PyTorch modules (Mamba, MoE, standard attention, etc.)

**Key**: Specs are used only during model instantiation, not during inference.

## 3. Actual Inference

**When**: Forward passes during inference/serving

**Process**:

- Uses the built PyTorch modules (already instantiated from step 2)
- Specs are **not used** - only the actual PyTorch modules handle forward passes
- Each layer executes according to its architecture (Mamba, MoE, attention, etc.)
