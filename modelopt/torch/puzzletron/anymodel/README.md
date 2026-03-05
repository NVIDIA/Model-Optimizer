# AnyModel Guide

This guide explains how to add support for new models in the compress pipeline.

## Convert model

Convert a HuggingFace model to Puzzletron format.

Step 1: Create Model Descriptor

Extend `ModelDescriptor` and implement `layer_name_predicates()` to define regex patterns for grouping weights into subblocks (embeddings, lm_head, block_N_ffn, block_N_attention).

Key points:

- Find weight names on the model's HuggingFace page → click "Files info" to see the safetensors structure with all tensor names (example: [Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct?show_file_info=model.safetensors.index.json))

See example: [llama_model_descriptor.py](models/llama/llama_model_descriptor.py)

Step 2: Create Converter

Extend `Converter` and implement `create_block_configs_from_main_config()` to create per-layer BlockConfigs from the HuggingFace config.

Key points:

- Import correct HuggingFace config class (e.g., `MistralConfig`, `LlamaConfig`, `Qwen2Config`). Find it in the transformers source: `github.com/huggingface/transformers/tree/main/src/transformers/models/<model_type>/configuration_<model_type>.py`

See example: [llama_converter.py](models/llama/llama_converter.py)

Step 3: Create `models/<model_name>/__init__.py`

Export descriptor and converter classes:

```python
from models.<model_name>.<model_name>_model_descriptor import MyModelDescriptor
from models.<model_name>.<model_name>_converter import MyConverter
```

Step 4: Register in `models/__init__.py`

Add import to trigger factory registration:

```python
from models.<model_name> import *
```

## Usage

```python
from scripts.convert_any_model import convert_model

convert_model(
    input_dir="path/to/hf_checkpoint",
    output_dir="path/to/puzzletron_checkpoint",
    converter="model_name",
)
```

## Compress model

Run pruning and compression on a Puzzletron model.

Step 1: Implement ModelDescriptor methods for compression

Add to your `ModelDescriptor`:

- `decoder_layer_cls()` - return the decoder layer class(es) to patch for heterogeneous config support
- `block_config_to_layer_overrides()` - map BlockConfig to layer override dict (see [details](#implementing-block_config_to_layer_overrides))
- `init_rotary_embedding()` - reinitialize rotary embeddings after model loading (see [details](#implementing-init_rotary_embedding))
- `input_embedding_name()` - return the name of the input embedding layer (see [details](#implementing-path-based-methods))
- `output_embedding_name()` - return the name of the output embedding layer (see [details](#implementing-path-based-methods))
- `layer_block_name()` - return the name pattern for decoder layers (see [details](#implementing-path-based-methods))
- `final_norm_name()` - return the name of the final normalization layer (see [details](#implementing-path-based-methods))
- `attn_no_op_post_init()` - replace attention sublayers with no-op modules
- `mlp_no_op_post_init()` - replace MLP sublayers with no-op modules

Step 2: Create FFN Layer Descriptor

Extend `FFNIntermediateLayerDescriptor` to define model-specific paths for FFN pruning hooks (`down_proj_name`, `ffn_prefix_name`, `linear_weight_names`). Derive values from your model's weight names in `layer_name_predicates()`.

See example: [llama_model_descriptor.py](models/llama/llama_model_descriptor.py) → `LlamaFFNIntermediateLayerDescriptor`

Step 3: Configure YAML files

Update the main model config YAML:

- Set `descriptor` to match the name used in `@ModelDescriptorFactory.register_decorator("your_model_name")`
- See example: [llama_3_1_8b_instruct.yaml](../../../../tests/gpu/torch/puzzletron/resources/configs/llama_3_1_8b_instruct/llama_3_1_8b_instruct.yaml)

Update pruning YAML files (`ffn_pruning.yaml`, `expert_pruning.yaml`, etc.):

- Set `pruning_mixin._target_` to the appropriate mixin class
- Set `layer_descriptor._target_` to your layer descriptor class
- Set `hook_class` to the activation hook for scoring
- Set `target_layer` in `activation_hooks_kwargs` to the layer name for hook attachment
- See examples in [configs/llama_3_1_8b_instruct/pruning/](../../../../tests/gpu/torch/puzzletron/resources/configs/llama_3_1_8b_instruct/pruning/)

## End-to-end example

See [test_compress_model.py](../../../../tests/gpu/torch/puzzletron/test_compress.py) for a complete example that runs both convert and compression steps.

---

## Advanced Topics

## Pruning Configuration

### Pruning YAML Structure

Each pruning type has a YAML config with these key fields:

```yaml
pruning_mixin:
  _target_: pruning.<type>_pruning_mixin.<MixinClass>
  layer_descriptor:
    _target_: models.<model>.<descriptor_class>

hook_class: ${get_object:utils.activation_hooks.hooks.<HookClass>}
activation_hooks_kwargs:
  method: <method_name>
  target_layer: "<layer.name>"  # e.g., "mlp.down_proj", "self_attn.o_proj"
```

| Field | Description |
|-------|-------------|
| `pruning_mixin._target_` | Mixin class that orchestrates this pruning type |
| `layer_descriptor._target_` | Model-specific class defining layer paths for hooks |
| `hook_class` | Activation hook class for importance scoring |
| `target_layer` | Layer name (relative to decoder block) where hooks attach |

### Adding a New Hook Class

1. **Implement the hook** in `modelopt/torch/nas/plugins/megatron_hooks/base_hooks.py`:
   - Extend an existing hook base class (e.g., `RemoveExpertsIndependentHook`)
   - Implement required methods (e.g., `get_router_logits_and_routed_experts`)

2. **Register the hook** in the appropriate pruning mixin's `supported_hooks()`:

   For FFN pruning (`pruning/ffn_intermediate_pruning_mixin.py`):

   ```python
   def supported_hooks(self) -> List[Type[ActivationsHook]]:
       return [IndependentChannelContributionHook, IterativeChannelContributionHook, YourNewHook]
   ```

   For expert removal (`pruning/expert_removal_pruning_mixin.py`):

   ```python
   def supported_hooks(self) -> List[Type[ActivationsHook]]:
       return [RankedChoiceVotingHook, ..., YourNewHook]
   ```

3. **Reference in YAML**:

   ```yaml
   hook_class: ${get_object:utils.activation_hooks.hooks.YourNewHook}
   ```

### Pruning Types Reference

| Type | Mixin | Example Hooks |
|------|-------|---------------|
| FFN intermediate | [`FFNIntermediatePruningMixIn`](../pruning/ffn_intermediate_pruning_mixin.py) | [`IterativeChannelContributionHook`](../../../nas/plugins/megatron_hooks/base_hooks.py), [`IndependentChannelContributionHook`](../../../nas/plugins/megatron_hooks/base_hooks.py) |
| Expert removal | [`ExpertRemovalPruningMixIn`](../pruning/expert_removal_pruning_mixin.py) | [`NemotronHRemoveExpertsIndependentHook`](../../../nas/plugins/megatron_hooks/base_hooks.py), [`Qwen3VLRemoveExpertsIndependentHook`](../../../nas/plugins/megatron_hooks/base_hooks.py) |
| KV heads | [`KVHeadsPruningMixIn`](../pruning/kv_heads_pruning_mixin.py) | [`IndependentKvHeadContributionHook`](../../../nas/plugins/megatron_hooks/base_hooks.py) |

## Implementing `block_config_to_layer_overrides`

Maps Puzzletron's [`BlockConfig`](../decilm/deci_lm_hf_code/block_config.py) fields to HuggingFace config attribute names. Only override attributes that change during pruning:

| BlockConfig Field | HuggingFace Attribute (check `config.json`) |
|-------------------|---------------------------------------------|
| `attention.num_key_value_heads` | `num_key_value_heads` |
| `ffn.intermediate_size` | `intermediate_size` |
| `ffn.moe.num_local_experts` | `num_experts` or `n_routed_experts` (model-specific) |
| `ffn.moe.expert_intermediate_dim` | `moe_intermediate_size` |

**Tip**: Check the model's `config.json` for exact attribute names - they vary between models.

See examples: [qwen3_vl](models/qwen3_vl/qwen3_vl_model_descriptor.py), [nemotron_h](models/nemotron_h/nemotron_h_model_descriptor.py)

---

## Implementing path-based methods

These methods return paths derived from the model's weight names:

- `input_embedding_name()`, `output_embedding_name()`, `layer_block_name()`, `final_norm_name()`

Find them on the model's HuggingFace page → "Files info" → safetensors structure (example: [Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct?show_file_info=model.safetensors.index.json)).

See example: [llama_model_descriptor.py](models/llama/llama_model_descriptor.py)

---

## Implementing `init_rotary_embedding`

Rotary embeddings are computed modules (not saved weights). After model sharding, they need re-initialization on the correct device/dtype.

Look in `github.com/huggingface/transformers/tree/main/src/transformers/models/<model_type>/modeling_<model_type>.py` for:

- `class.*Rotary` — the rotary embedding class name and constructor arguments
- `self.rotary_emb` — the attribute path

See example: [llama_model_descriptor.py](models/llama/llama_model_descriptor.py)
