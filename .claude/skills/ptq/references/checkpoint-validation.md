# Post-Quantization Checkpoint Validation

Before treating an exported checkpoint as ready for deployment/evaluation, verify checkpoint size/bits, quantized-weight coverage, and metadata consistency. This is a gate, not a guideline: do not submit evals, start serving jobs, or mark the checkpoint ready until all required checks pass and the validation report is recorded.

## Required checks

1. The quantized checkpoint is smaller on disk than the baseline/source checkpoint and has lower estimated bits per weight. Record source size, output size, and output/source ratio. A partial-quantization recipe may not shrink every tensor, but it should still match the intended quantization coverage. If the size reduction is small or missing, explain why before proceeding.
2. The weights that were actually quantized match what the requested qformat/recipe/config targeted. Record layer precision counts grouped by actual/declarative precision, such as NVFP4, FP8, INT4, BF16/unquantized excluded, unexpected unquantized, and declaration mismatches. Quantization config patterns may silently miss layers if the model uses non-standard naming — this only surfaces later as deployment failures when the serving framework tries to load unquantized weights as quantized.
3. Metadata that should not change still matches the baseline/source model. Compare generation settings, tokenizer files, chat template, model architecture fields, max positions/context length, and special tokens; quantization should affect weights and quantization metadata, not silently change prompting or generation behavior. Record every diff and classify it as expected or blocking.

## Gate report

Before moving to deployment/evaluation, report a table in this shape:

| Check | Result |
| --- | --- |
| Size vs source | `<output> GB / <source> GB = <ratio>x`; PASS only if the ratio matches the recipe's compression intent |
| Layer precision counts | `<count> NVFP4 / <count> FP8 / <count> INT4 / <count> BF16-or-excluded / <count> unexpected / <count> declaration mismatches` |
| Metadata | `no unexpected diffs` or list exact diffs |

Stop instead of proceeding if:

- Output/source ratio is `>= 1.0` for a compression recipe, unless the user explicitly accepts the explanation.
- Any layer group intended to be quantized has zero or unexpectedly low coverage.
- Any layer has quantization metadata inconsistent with its declared precision.
- Prompting, tokenizer, generation, architecture, context-length, or special-token metadata changed unexpectedly.

## Expected quantization patterns by recipe

| Recipe (`--qformat`) | What should be quantized | What should be excluded |
|----------------------|-------------------------|------------------------|
| `nvfp4` | All linear layers | lm_head, routers, norms, embeddings |
| `nvfp4_mlp_only` | MLP layers (including MoE experts) | Attention layers, lm_head, routers |
| `nvfp4_experts_only` | MoE expert layers only | Dense MLP, attention, lm_head, routers |
| `nvfp4_omlp_only` | MLP + o_proj layers | Other attention layers, lm_head, routers |
| `fp8` | All linear layers | lm_head, norms, embeddings |
| `int4_awq` | All linear layers | lm_head, norms, embeddings |

## Size check

Compare only checkpoint weight files, not cache directories or eval artifacts:

```bash
python3 -c "
from pathlib import Path

source = Path('<source_checkpoint_path>')
output = Path('<output_path>')

def safetensor_bytes(path):
    files = list(path.glob('*.safetensors')) if path.is_dir() else [path]
    return sum(p.stat().st_size for p in files)

src = safetensor_bytes(source)
dst = safetensor_bytes(output)
ratio = dst / src if src else float('nan')
print(f'Source safetensors: {src / 1e9:.2f} GB')
print(f'Output safetensors: {dst / 1e9:.2f} GB')
print(f'Output/source ratio: {ratio:.2f}x')
"
```

Treat the ratio as the first-order bits-per-weight proxy unless you separately load tensors and compute exact parameter bit counts. For compression recipes, a ratio at or above `1.0x` is blocking unless the user explicitly accepts the explanation.

## Layer coverage and precision script

Run against the exported checkpoint to check every linear layer is either quantized with the expected precision or explicitly excluded. This handles both uniform `quant_algo` exports and mixed-precision `quantized_layers` exports:

```bash
python3 -c "
import collections, fnmatch, json, os

output = '<output_path>'
idx = json.load(open(os.path.join(output, 'model.safetensors.index.json')))
cfg = json.load(open(os.path.join(output, 'hf_quant_config.json')))
q = cfg.get('quantization', {})
excludes = q.get('exclude_modules', []) or q.get('ignore', [])
declared_layers = q.get('quantized_layers') or {}
uniform_algo = q.get('quant_algo')
if uniform_algo == 'MIXED_PRECISION':
    uniform_algo = None

all_keys = set(idx['weight_map'].keys())
# Identify linear weight params (skip norms, embeddings, scalars, scales)
skip_suffixes = ('_scale', '_scale_2', 'layernorm', 'layer_norm', 'norm.weight', 'embed', 'scalar')
linear_weights = sorted(k for k in all_keys
    if k.endswith('.weight') and not any(s in k.lower() for s in skip_suffixes))

def is_excluded(base, weight):
    return any(fnmatch.fnmatch(weight, p) or fnmatch.fnmatch(base, p) for p in excludes)

def declared_algo(base):
    if base in declared_layers:
        return declared_layers[base].get('quant_algo', 'DECLARED_UNKNOWN')
    if is_excluded(base, base + '.weight'):
        return 'BF16/EXCLUDED'
    if uniform_algo:
        return uniform_algo
    return 'UNDECLARED'

precision_counts = collections.Counter()
unexpected = []
mismatches = []
for w in linear_weights:
    base = w.rsplit('.weight', 1)[0]
    algo = declared_algo(base)
    has_scales = any(f'{base}.{s}' in all_keys for s in
                     ['weight_scale', 'weight_scale_2', 'input_scale', 'activation_scale', 'weight_scale_inv'])

    if has_scales and algo not in ('BF16/EXCLUDED', 'UNDECLARED'):
        precision_counts[algo] += 1
    elif has_scales and algo in ('BF16/EXCLUDED', 'UNDECLARED'):
        precision_counts['QUANTIZED_BUT_' + algo.replace('/', '_')] += 1
        mismatches.append((w, algo, 'has quantization scales'))
    elif not has_scales and algo == 'BF16/EXCLUDED':
        precision_counts['BF16/EXCLUDED'] += 1
    else:
        precision_counts['UNEXPECTED_UNQUANTIZED'] += 1
        unexpected.append((w, algo, 'no quantization scales'))

print('Layer precision counts:')
for name, count in sorted(precision_counts.items()):
    print(f'  {name}: {count}')
print(f'Unexpected unquantized layers: {len(unexpected)}')
print(f'Declaration mismatches: {len(mismatches)}')
if unexpected:
    print(f'\nWARNING: {len(unexpected)} layers have NO scales and are NOT in exclude list:')
    # Group by module type for readability
    groups = {}
    for w, algo, reason in unexpected:
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
if mismatches:
    print(f'\nWARNING: {len(mismatches)} layers have declaration/metadata mismatches:')
    for w, algo, reason in mismatches[:20]:
        print(f'  {w}: declared {algo}, {reason}')
    if len(mismatches) > 20:
        print(f'  ... {len(mismatches) - 20} more')
if not unexpected and not mismatches:
    print('\nAll layers are quantized at the declared precision or explicitly excluded.')
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
