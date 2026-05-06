# Post-Quantization Checkpoint Validation

Verify the exported checkpoint's quantization pattern matches the recipe used. Quantization config patterns may silently miss layers if the model uses non-standard naming — this only surfaces later as deployment failures when the serving framework tries to load unquantized weights as quantized.

## Expected quantization patterns by recipe

| Recipe (`--qformat`) | What should be quantized | What should be excluded |
|----------------------|-------------------------|------------------------|
| `nvfp4` | All linear layers | lm_head, routers, norms, embeddings |
| `nvfp4_mlp_only` | MLP layers (including MoE experts) | Attention layers, lm_head, routers |
| `nvfp4_experts_only` | MoE expert layers only | Dense MLP, attention, lm_head, routers |
| `nvfp4_omlp_only` | MLP + o_proj layers | Other attention layers, lm_head, routers |
| `fp8` | All linear layers | lm_head, norms, embeddings |
| `int4_awq` | All linear layers | lm_head, norms, embeddings |

## Validation script

Run against the exported checkpoint to check every linear layer is either quantized (has scale params) or explicitly excluded:

```bash
python3 -c "
import json, fnmatch

output = '<output_path>'
idx = json.load(open(f'{output}/model.safetensors.index.json'))
cfg = json.load(open(f'{output}/hf_quant_config.json'))
excludes = cfg['quantization']['exclude_modules']

all_keys = set(idx['weight_map'].keys())
# Identify linear weight params (skip norms, embeddings, scalars, scales)
skip_suffixes = ('_scale', '_scale_2', 'layernorm', 'layer_norm', 'norm.weight', 'embed', 'scalar')
linear_weights = sorted(k for k in all_keys
    if k.endswith('.weight') and not any(s in k.lower() for s in skip_suffixes))

# Check which have quantization scales
quantized, excluded, unexpected = [], [], []
for w in linear_weights:
    base = w.rsplit('.weight', 1)[0]
    has_scales = any(f'{base}.{s}' in all_keys for s in ['weight_scale', 'input_scale'])
    is_excluded = any(fnmatch.fnmatch(w, p) or fnmatch.fnmatch(base, p) for p in excludes)

    if has_scales:
        quantized.append(w)
    elif is_excluded:
        excluded.append(w)
    else:
        unexpected.append(w)

print(f'Quantized layers: {len(quantized)}')
print(f'Excluded layers (in exclude_modules): {len(excluded)}')
if unexpected:
    print(f'\nWARNING: {len(unexpected)} layers have NO scales and are NOT in exclude list:')
    # Group by module type for readability
    groups = {}
    for w in unexpected:
        parts = w.split('.')
        module_type = next((p for p in parts if p in
            ('self_attn', 'mlp', 'experts', 'router', 'lm_head', 'embed_tokens', 'vision_tower')), 'other')
        groups.setdefault(module_type, []).append(w)
    for mtype, weights in sorted(groups.items()):
        print(f'  {mtype}: {len(weights)} weights (e.g., {weights[0]})')
    print()
    print('These layers were silently skipped during quantization.')
    print('Likely cause: quantization config patterns did not match these module names.')
    print('This WILL cause deployment failures (framework loads them as quantized but they are BF16).')
    print('Fix: add missing patterns to the config, or add to exclude_modules if intentionally unquantized.')
else:
    print('\nAll layers are either quantized or explicitly excluded. Checkpoint is consistent.')
"
```

## Common pattern gaps

Layers silently skipped because the quantization config patterns don't match the model's naming:

| Model | Module path | Missed by pattern | Fix |
|-------|-------------|-------------------|-----|
| Gemma4 MoE | `layers.N.experts.*` | `*mlp*`, `*block_sparse_moe*` | Add `*.experts.*` (PR #1219) |
| Custom MoE | `layers.N.moe_block.experts.*` | `*mlp*` | Add matching pattern |
| VLM projector | `multi_modal_projector.*` | — | Usually excluded; verify |

## What to do when warnings appear

- **Layers should have been quantized** (e.g., MoE experts with `nvfp4_mlp_only`): the quantization config patterns missed them. Fix by adding the missing pattern to the config and re-running PTQ. Check if ModelOpt already has a plugin for the model in `modelopt/torch/quantization/plugins/huggingface.py`.

- **Layers are intentionally unquantized** (e.g., attention layers with `nvfp4_mlp_only`): they should be in the `exclude_modules` list but the export didn't add them. Add them manually to both `hf_quant_config.json` and `config.json` `quantization_config.ignore` in the checkpoint to prevent deployment failures.
