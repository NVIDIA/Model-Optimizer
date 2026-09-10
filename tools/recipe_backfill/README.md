# Published-checkpoint recipe backfill

Tooling that keeps `modelopt_recipes/` honest about NVIDIA's released quantized
checkpoints: for every checkpoint NVIDIA publishes in the
[Inference Optimized Checkpoints][collection] collection, which recipe reproduces it, and
a check that it still does.

Scope is NVIDIA's **own** releases. The collection also carries partner-published
checkpoints (LGAI-EXAONE, stepfun-ai, thinkingmachines, black-forest-labs); the scan
still records them, but they are listed under `unmapped` in `recipe_map.json` instead of
getting recipes.

**Coverage is partial by design.** The backfill lands one source-model org per branch
(`shengliangx/batch-backfill-recipe-<org>`), so `recipe_map.json` on any given branch
covers only the orgs merged so far. `verify_recipes.py` prints the checkpoints that are
not covered yet; nothing fails on them.

The problem it solves: a recipe that claims to mirror a published checkpoint can drift
from it silently. A wildcard is widened, a new architecture reuses a leaf name, an
exclusion is dropped — and nothing fails until someone re-quantizes a 600 B model and
diffs the result. These tools turn that into a unit test that runs in a couple of
seconds with no GPU, no Hub access and no weights.

## Files

| File | What it is |
|---|---|
| `scan_collection.py` | **Online.** Reads the collection and, per checkpoint, its `config.json`, `hf_quant_config.json` and `model.safetensors.index.json`; writes the snapshot. |
| `published_checkpoints.json` | The snapshot: every quantizable module of every released checkpoint, mapped to the format it ships in. |
| `recipe_map.json` | Published checkpoint → the recipe that reproduces it, plus the checkpoints that deliberately have none. |
| `checkpoint_scan.py` | Shared helpers: module-map construction and the compact packed form both of the above use. |
| `verify_recipes.py` | **Offline.** Replays a recipe's `quant_cfg` over the snapshot's module names and diffs the result. |
| `render_index.py` | Regenerates `modelopt_recipes/published_checkpoints.md` from `recipe_map.json`. |
| `render_aliases.py` | Regenerates the thin checkpoint **alias** recipes under `modelopt_recipes/models/`. |

## Usage

```bash
# Check every mapped recipe against its checkpoint (this is what CI runs).
python tools/recipe_backfill/verify_recipes.py

# Why does one of them disagree?
python tools/recipe_backfill/verify_recipes.py --checkpoint Qwen3-Next --show-diff

# Refresh the snapshot after a new release lands in the collection.
# HF_TOKEN is needed for gated repos (the Llama-4 mirrors).
python tools/recipe_backfill/scan_collection.py \
    --out tools/recipe_backfill/published_checkpoints.json \
    --cache-dir ~/.cache/modelopt-recipe-backfill

# Refresh the generated index doc and the checkpoint aliases.
python tools/recipe_backfill/render_index.py
python tools/recipe_backfill/render_aliases.py
```

## Aliases

Every published checkpoint gets an entry under
`modelopt_recipes/models/<source org>/<source model>/ptq/`, so a release is findable at
its own hub path. When a portable recipe reproduces it unchanged — which is true of most
of them — that entry is a generated **alias**: a top-level `$import` of the recipe that
reproduces it, overriding only `metadata`. Nothing is duplicated, and editing the base
recipe changes every alias pointing at it.

The alias folder is keyed by the checkpoint's **source** model, read from the release's
`base_model` card field. When a release does not declare one, put a `source_model` on its
`recipe_map.json` entry.

## Adding a checkpoint

1. Re-run `scan_collection.py` so the new release is in the snapshot.
2. Run `verify_recipes.py` — the test will now report the checkpoint as unaccounted for.
3. Try an existing recipe first. Most releases are reproduced by a **general** recipe;
   a new file under `models/` is only warranted when the checkpoint genuinely deviates
   (see the "What belongs here" section of `modelopt_recipes/models/README.md`).
4. Add the mapping to `recipe_map.json` and re-run the verifier until it passes.
5. Re-run `render_index.py` and `render_aliases.py`. A checkpoint that needed a recipe of
   its own also wants a mention in `modelopt_recipes/ptq.md`, next to the other
   deviations; an aliased one is covered by the generated index.

## How a checkpoint's module map is derived

Three sources, because no single one is sufficient:

- **`hf_quant_config.json`** is authoritative for *which format* a module got. It is
  what the exporter wrote out of the quantizer states, and it is the only place that
  distinguishes per-tensor FP8 from block-scaled `FP8_PB_WO`, or W4A16 NVFP4 from W4A4.
- **The exported scale tensors** settle *whether* a module was quantized at all. Older
  exports ship an `exclude_modules` list that is not exhaustive — the `mlp.gate` routers
  are missing from several — and a quantized `nn.Linear` always carries a
  `weight_scale`, so its absence is proof. (The exception is `nn.Conv1d` in a
  linear-attention block, which the exporter does not scale even when its quantizer is
  on; there the config wins.)
- **The safetensors headers**, read with two HTTP range requests per shard, cover the
  oldest exports whose index lists only `.weight` and carries no scales at all. A packed
  low-precision weight is not BF16.

Two things the config cannot express are read from the weights directly:

- **Cast vs. calibrated KV cache.** `hf_quant_config.json` records only `FP8`.
  `use_constant_amax` pins amax to the E4M3 max, so a *cast* KV cache exports
  `k_scale == 1.0` exactly; a calibrated one essentially never does. The scanner samples
  one `k_scale` per checkpoint to tell them apart, which is what separates
  `kv_fp8_cast` from `kv_fp8`.
- **Layers the model class never builds.** A checkpoint declaring
  `num_hidden_layers: 61` with `num_nextn_predict_layers: 1` ships weights for
  `model.layers.61` — the MTP block — that the HF class does not instantiate. Those
  tensors are in the file but outside the quantized model, so they are dropped from the
  comparison rather than counted against the recipe.

## What verification actually compares

`verify_recipes.py` replays the recipe's `quant_cfg` the way
`modelopt.torch.quantization.conversion.set_quantizer_by_cfg` does — `fnmatch`
wildcards applied in list order, later entries overriding, `cfg` replacing attributes
wholesale while a bare `enable` only toggles — over the quantizer names the snapshot's
modules would carry, then compares the resulting per-module format against the published
one. Wildcards that cannot tell two modules apart collapse to one representative, so a
million-module checkpoint costs the same as a dense one.

Per-checkpoint escape hatches in `recipe_map.json`, all requiring an explicit reason:

- `quantizer_naming: fused_parameter` — the model keeps several weights as plain
  `nn.Parameter`s and registers one quantizer per weight on the parent
  (DeepSeek-V4's `Expert.w1/w2/w3` → `w1_weight_quantizer`).
- `ignore_modules` — modules whose published format is the *source* checkpoint's rather
  than PTQ output, e.g. a speculative-decoding block the recipe leaves alone that still
  carries the source model's native block-FP8 weights.
- `approximate` — a recipe that is a documented approximation of its release rather than
  an exact mirror. These report as `WARN` and `xfail`; the test also fails if one starts
  matching exactly, so the marker cannot go stale.

[collection]: https://huggingface.co/collections/nvidia/inference-optimized-checkpoints-with-model-optimizer
