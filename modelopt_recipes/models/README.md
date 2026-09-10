# Recipes for specific model-hub checkpoints

This folder holds model-optimization recipes (e.g. PTQ recipes) tuned for a
**specific published model instance** — one checkpoint released on a model hub
such as the [Hugging Face Hub](https://huggingface.co/),
[ModelScope](https://modelscope.cn/), or similar. Unlike
[`../huggingface/`](../huggingface/), which keys recipes by a transformers
`model_type` (an architecture shared by many checkpoints), an entry here is keyed
to **one checkpoint**.

Every checkpoint NVIDIA has published is listed here, in one of two forms: a
**mirror**, whose body reproduces a per-layer scheme no portable recipe can
express, or an **alias**, which records that a `general/` or
`huggingface/<model_type>/` recipe reproduces the release unchanged and imports
it wholesale. See [What belongs here](#what-belongs-here) for which to write, and
[`../published_checkpoints.md`](../published_checkpoints.md) for the full index.

## Folder structure

Each instance is keyed by its **model-hub path** — the same `<org>/<model_id>`
you pass to `from_pretrained(...)` or find in the hub URL. The on-disk path
mirrors the hub path exactly:

```text
modelopt_recipes/models/
  <org>/                       # hub namespace / organization, e.g. nvidia, mistralai
    <model_id>/                # hub model id, e.g. Nemotron-3-Nano-4B-BF16
      <task>/                  # optimization workflow, e.g. ptq
        <recipe>.yaml
        [<recipe>.<aux>.yaml]  # optional $import snippet helpers (see below)
        [README.md]            # optional; describes what's checkpoint-specific
```

For example, the recipe for the hub checkpoint `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16`
(`https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16`) lives at
`models/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16/ptq/`. Because the folder path *is* the
hub path, you can go straight from a checkpoint id to its recipe — and back —
with no lookup table.

`<task>` is the optimization workflow the recipe targets (e.g. `ptq` for
post-training quantization).

### Naming the `<org>/<model_id>` folders

Use the checkpoint's exact hub `<org>/<model_id>`, including casing. When the
same weights are published on more than one hub (e.g. the Hugging Face Hub and
ModelScope) under the same `<org>/<model_id>`, a single folder serves them all.
When a recipe was tuned against one **canonical / base** checkpoint but also
applies to its mirrors, key it by that base model's id.

## Choosing a recipe

Prefer the most specific entry that applies to your model:

1. **`models/<org>/<model_id>/`** — if there is an entry for your **exact**
   checkpoint. Use it to match a published quantized checkpoint: it either
   reproduces a validated, often per-component mixed-precision scheme for that
   release, or aliases the portable recipe that does.
2. **[`huggingface/<model_type>/`](../huggingface/)** — an architecture-level
   recipe that applies to every checkpoint of that `model_type`.
3. **[`general/`](../general/)** — model-agnostic recipes; a good starting point
   for any model without a more specific entry.

## Selecting a recipe at runtime

Use the path relative to `modelopt_recipes/`:

```text
--recipe models/<org>/<model_id>/<task>/<recipe>
```

or from Python:

```python
from modelopt.recipe import load_recipe

recipe = load_recipe("models/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16/ptq/nvfp4_w4a16")
```

## What belongs here

Two kinds of entry, and the difference matters when you add one.

### 1. Checkpoint mirrors — the recipe lives here

A recipe earns a **body** here only when it mirrors **one specific released (or
planned) checkpoint** — a hand-mapped, usually per-layer or per-component
precision scheme tuned to match that exact release. If the tuning generalizes to
every checkpoint of an architecture, it belongs under
[`../huggingface/<model_type>/`](../huggingface/) instead; if it is
model-agnostic, it belongs under [`../general/`](../general/). See
[`../ptq.md`](../ptq.md) for what each checkpoint mirror does and how it compares
to its general baseline.

### 2. Aliases — the recipe lives elsewhere, the *record* lives here

Most released checkpoints are reproduced by a portable recipe with no
checkpoint-specific changes at all. Those still get an entry here, so every
published checkpoint is findable at its own hub path — but the entry is a thin
**alias** that delegates its whole body to the recipe that reproduces it:

```yaml
imports:
  base: general/ptq/nvfp4_default-kv_fp8_cast

$import: base
metadata:
  description: >-
    meta-llama/Llama-3.1-8B-Instruct as published in nvidia/Llama-3.1-8B-Instruct-NVFP4.
    Alias for `general/ptq/nvfp4_default-kv_fp8_cast`, which reproduces this
    checkpoint unchanged.
```

A top-level `$import` brings in the whole imported recipe; the keys given
alongside it override the imported ones, so the alias supplies its own
`metadata` and inherits `quantize` — algorithm and every `quant_cfg` entry —
unchanged. Nothing is duplicated: editing the base recipe changes every alias
that points at it.

Note what the alias does *not* say. It has no `recipe_type`, because the kind is
whatever the imported recipe's kind is; a recipe states its kind in whichever of
these it likes, and the loader takes the first that answers:

1. a `# modelopt-schema:` comment naming its schema class,
2. `metadata.recipe_type`, or
3. the recipe it delegates to via a top-level `$import`.

Whatever a recipe does state has to be true. Declaring both a schema comment and
a `recipe_type` is fine as long as they agree, and the same holds across a
delegation: a recipe and the recipe it imports must be the same kind, since the
import takes over the whole body. Any disagreement is an error, not a preference.

The one thing that *is* required: **a recipe another file imports must carry the
schema comment**, because that is what `$import` resolution needs to validate the
imported payload. A recipe nothing imports needs nothing — so aliasing a recipe
for the first time means adding the comment to it in the same change.

Aliases are **generated**, not hand-written, from the checkpoint-to-recipe map:

```bash
python tools/recipe_backfill/render_aliases.py          # rewrite them
python tools/recipe_backfill/render_aliases.py --check   # verify they are current
```

so add the mapping to `tools/recipe_backfill/recipe_map.json` and re-render
rather than editing an alias file by hand. `tests/unit/recipe/test_published_checkpoint_recipes.py`
fails if a checked-in alias has drifted, and
[`../published_checkpoints.md`](../published_checkpoints.md) is the full index.

**Which one to write.** Start by assuming an alias: run
`tools/recipe_backfill/verify_recipes.py` against the candidate portable recipe.
Only when no portable recipe reproduces the release does it warrant a body of its
own here.

## Sharing content across recipes

When several recipes reuse the same body, extract it into a sibling **snippet**
file with a `# modelopt-schema:` header and `$import` it, keeping each recipe
wrapper thin. Name snippets so they are obviously not runnable recipes (e.g.
`<recipe>.<field>.yaml`), and reference them by their path relative to
`modelopt_recipes/`.

## Per-folder READMEs

Each `<task>/` folder may contain a short `README.md` describing exactly what is
checkpoint-specific — which layers deviate, the calibration used, and the
reference checkpoint it mirrors — so reviewers and users don't have to diff the
YAML against the generic presets to see the intent.
