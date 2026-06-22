---
name: puzzletron
description: "End-to-end workflow for model pruning and MIP-based optimization. Commands: mip, all, add-model. Usage: /puzzletron <command> [args]"
license: Apache-2.0
---

# Puzzletron

## Routing

**STEP 1 — Check args before doing anything else. This is MANDATORY.**

- If args are **empty**, output the block below verbatim and **STOP immediately. Do NOT proceed to any command.**
- If the first word of args does **not exactly match** `mip`, `all`, or `add-model`, output the block below verbatim and **STOP immediately. Do NOT proceed to any command.**

---

**Puzzletron** — end-to-end workflow for model pruning and MIP-based optimization.

Available commands:
- `mip <nproc_per_node>` — Run the MIP step (nproc_per_node: number of GPUs per node)
- `mip progress` — Show live MIP progress with timing summary
- `mip losses` — Show teacher vs. compressed model accuracy for the MIP solution
- `all <nproc_per_node>` — Run the full Puzzletron pipeline (nproc_per_node: number of GPUs per node)
- `all progress` — Show live full pipeline progress with timing summary
- `add-model <hf_model_path>` — Implement descriptor, converter, and configs for an unsupported model

Usage: `/puzzletron <command> [args]`

---

**STEP 2 — Only if the first word of args exactly matches a command name, execute it. Never reach this step if args were empty.**

## Command: all

Parse `nproc_per_node` from args using either positional or flag syntax:
- Positional: second word is a number, e.g. `all 2`
- Flag: `--nproc_per_node <value>` anywhere in args, e.g. `all --nproc_per_node 2`

- If the second word is exactly `progress`, execute the **all progress** sub-command below.
- If no `nproc_per_node` value can be found, ask the user: "Please provide the number of GPUs per node (nproc_per_node)." and **STOP**.
- If the value does not match `^[0-9]+$`, ask the user: "nproc_per_node must be a positive integer." and **STOP**.
- Otherwise use the parsed value and run the full pipeline.

### all \<nproc_per_node\>

Run the following Bash command, substituting `<nproc_per_node>` with the parsed value:

```bash
set -o pipefail && export PYTHONPATH=$PYTHONPATH:/workspace/Model-Optimizer && \
torchrun --nproc_per_node <nproc_per_node> examples/puzzletron/main.py \
  --config examples/puzzletron/configs/llama-3_1-8B_pruneffn_memory/llama-3_1-8B_pruneffn_memory.yaml \
  2>&1 | tee ./log.txt | grep "Puzzletron Progress"
```

Stream output to the user as it arrives. When the command finishes, report the exit code.

### all progress

Run the following Bash command. Present the output to the user wrapped in a fenced code block (``` ... ```).

```bash
python3 .agents/skills/puzzletron/all_progress.py
```

## Command: mip

Parse `nproc_per_node` from args using either positional or flag syntax:
- Positional: second word is a number, e.g. `mip 2`
- Flag: `--nproc_per_node <value>` anywhere in args, e.g. `mip --nproc_per_node 2`

- If the second word is exactly `progress`, execute the **mip progress** sub-command below.
- If the second word is exactly `losses`, execute the **mip losses** sub-command below.
- If no `nproc_per_node` value can be found, ask the user: "Please provide the number of GPUs per node (nproc_per_node)." and **STOP**.
- If the value does not match `^[0-9]+$`, ask the user: "nproc_per_node must be a positive integer." and **STOP**.
- Otherwise use the parsed value and run the MIP step.

### mip \<nproc_per_node\>

Run the following Bash command, substituting `<nproc_per_node>` with the parsed value:

```bash
set -o pipefail && export PYTHONPATH=$PYTHONPATH:/workspace/Model-Optimizer && \
torchrun --nproc_per_node <nproc_per_node> examples/puzzletron/main.py \
  --config examples/puzzletron/configs/llama-3_1-8B_pruneffn_memory/llama-3_1-8B_pruneffn_memory.yaml \
  --mip-only 2>&1 | tee ./log.txt | grep "Puzzletron Progress"
```

Stream output to the user as it arrives. When the command finishes, report the exit code.

### mip progress

Run the following Bash command. Present the output to the user wrapped in a fenced code block (``` ... ```).

```bash
python3 .agents/skills/puzzletron/mip_progress.py
```

### mip losses

Run the following Bash command. Present the output to the user wrapped in a fenced code block (``` ... ```).

```bash
python3 .agents/skills/puzzletron/mip_losses.py
```

## Command: add-model

Parse `hf_model_path` from args (the second word). If missing, ask: "Please provide the HuggingFace model path (local or hub)." and **STOP**.

Then follow the steps below to implement full Puzzletron support for the model.

### Step 1 — Check if already supported

```bash
python3 -c "
import sys; sys.path.insert(0, '.')
from modelopt.torch.puzzletron.anymodel.model_descriptor import ModelDescriptorFactory
from transformers import AutoConfig
cfg = AutoConfig.from_pretrained('<hf_model_path>', trust_remote_code=True)
supported = cfg.model_type in ModelDescriptorFactory.CLASS_MAPPING
print(f'model_type: {cfg.model_type}')
print(f'already supported: {supported}')
"
```

If already supported, tell the user and **STOP**.

If `AutoConfig` raises an error about an unrecognised model type, the installed Transformers version is too old. Check the version, upgrade with `python3 -m pip install --upgrade transformers`, then re-run.

### Step 2 — Inspect the architecture

Run the following to understand what you are implementing:

```bash
python3 -c "
from transformers import AutoConfig
cfg = AutoConfig.from_pretrained('<hf_model_path>', trust_remote_code=True)
print(cfg)
# If it has a nested text_config, print that too
if hasattr(cfg, 'text_config'):
    print('--- text_config ---')
    print(cfg.text_config)
"
```

Key things to note:
- **`model_type`** — this becomes the registration key for both descriptor and converter.
- **Nested `text_config`** — VLMs (e.g. Qwen3.5) wrap language model params inside `config.text_config`. Use `config.text_config` wherever you need `num_hidden_layers`, `intermediate_size`, `num_key_value_heads`. The converter must save `text_config` (not the full VLM config) so downstream code can access these fields directly.
- **Hybrid attention** — check `cfg.text_config.layer_type_list` (or similar). If some layers are linear/recurrent and others are full attention, `attn_no_op_post_init` must branch on `decoder_layer.layer_type`.
- **MoE** — if `num_experts` > 1 the model uses a MoE FFN and is not currently supported by the FFN pruning path; skip FFN pruning for such models.
- **Weight name prefixes** — inspect the checkpoint index to understand the layout:

```bash
python3 -c "
import json, collections
idx = json.load(open('<hf_model_path>/model.safetensors.index.json'))
prefixes = collections.Counter()
for n in idx['weight_map']:
    prefixes['.'.join(n.split('.')[:3])] += 1
for p, c in sorted(prefixes.items()):
    print(f'{c:4d}  {p}')
"
```

If weight names use a prefix like `model.language_model.*` rather than `model.*`, the converter must implement `convert_weight_name` to remap them, and `get_weight_groups` must handle both the original checkpoint names (used during conversion) and the remapped names (used when saving pruned checkpoints). See the Qwen3_5 descriptor/converter for the reference implementation of this pattern.

### Step 3 — Create the files

Create the following files (use an existing descriptor as a reference — `qwen3_5` for VLMs with nested config and weight remapping, `llama` or `qwen2` for standard text-only models):

**`modelopt/torch/puzzletron/anymodel/models/<model_type>/__init__.py`**

```python
from .<model_type>_converter import *
from .<model_type>_model_descriptor import *
```

**`modelopt/torch/puzzletron/anymodel/models/<model_type>/<model_type>_model_descriptor.py`**

Must implement (inheriting from `ModelDescriptor`):
- `decoder_layer_cls()` → the HF decoder layer class
- `input_embedding_name()` → e.g. `"model.embed_tokens"`
- `output_embedding_name()` → e.g. `"lm_head"`
- `final_norm_name()` → e.g. `"model.norm"`
- `layer_block_name(index)` → e.g. `f"model.layers.{index}"`
- `block_config_to_layer_overrides(block_config)` → dict with `intermediate_size` and `num_key_value_heads`
- `attn_no_op_post_init(decoder_layer)` → replace attention + input norm with no-ops
- `mlp_no_op_post_init(decoder_layer)` → replace MLP + post-attention norm with no-ops
- `layer_name_predicates(num_layers)` → regex dict grouping weights into `embeddings`, `lm_head`, `block_N_ffn`, `block_N_attention`
- `init_rotary_embedding(model, runtime)` → re-initialise rotary embedding after subblock load

**Critical:** `layer_name_predicates` patterns must match the **converted** `model.*` names (not the original VLM checkpoint names). If the checkpoint uses a different prefix, override `get_weight_groups` to normalise names before matching and restore originals in the returned groups (so `param_to_file` lookups in `convert_model_weights` still work). See `Qwen3_5ModelDescriptor.get_weight_groups` for the reference pattern.

**`modelopt/torch/puzzletron/anymodel/models/<model_type>/<model_type>_converter.py`**

Must implement (inheriting from `Converter`):
- `create_block_configs_from_main_config(config)` → list of `BlockConfig`, one per layer
- `convert_configs_in_dirs(input_dir, output_dir)` → if the model has a nested `text_config`, save that instead of the full VLM config so `num_hidden_layers` is accessible at the top level
- `convert_weight_name(name)` → remap checkpoint weight names to converted model names (identity if no remapping needed)

**Register in `modelopt/torch/puzzletron/anymodel/models/__init__.py`** — gate behind the minimum Transformers version that introduced the model:

```python
if _Version(_transformers_version) >= _Version("X.Y.Z"):
    from .<model_type> import *
```

**Compression config** at `examples/puzzletron/configs/<model_type>-<size>_pruneffn_memory/`:
- Base YAML (`<model_type>.yaml`): `descriptor: <model_type>`, MIP constraints
- Main YAML (override): `input_hf_model_path`, `dataset_path`, `puzzle_dir` (use a **model-specific path** to avoid collisions with other models), `pruning.intermediate_size_list`
- Pruning YAML: points `layer_descriptor._target_` at the new `FFNIntermediateLayerDescriptor` subclass

Choose `intermediate_size_list` by scaling the Llama-3.1-8B ratios (~21%, 42%, 60%, 83% of teacher) to the new model's `intermediate_size`.

### Step 4 — Verify registration

```bash
python3 -c "
import sys; sys.path.insert(0, '.')
from modelopt.torch.puzzletron.anymodel.model_descriptor import ModelDescriptorFactory
from modelopt.torch.puzzletron.anymodel.converter import ConverterFactory
print('descriptor:', '<model_type>' in ModelDescriptorFactory.CLASS_MAPPING)
print('converter: ', '<model_type>' in ConverterFactory.CLASS_MAPPING)
"
```

Both must print `True`. If not, check the `__init__.py` import chain and the `@register_decorator` keys.

### Step 5 — Tell the user what was created

List the files created, confirm registration, and suggest running the pipeline:

```text
run puzzletron all for <model_name> on <N> GPUs
```
