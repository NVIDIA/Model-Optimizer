# Recipes for specific model-hub checkpoints

This folder holds model-optimization recipes (e.g. PTQ recipes) tuned for a
**specific published model instance** — one checkpoint released on a model hub
such as the [Hugging Face Hub](https://huggingface.co/),
[ModelScope](https://modelscope.cn/), or similar. Unlike
[`../huggingface/`](../huggingface/), which keys recipes by a transformers
`model_type` (an architecture shared by many checkpoints), a recipe here mirrors
**one checkpoint's** quantization scheme verbatim.

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
   checkpoint. It reproduces a validated, often per-component mixed-precision
   scheme for that release; use it to match a published quantized checkpoint.
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

A recipe earns a place here only when it mirrors **one specific released (or
planned) checkpoint** — a hand-mapped, usually per-layer or per-component
precision scheme tuned to match that exact release. If the tuning generalizes to
every checkpoint of an architecture, it belongs under
[`../huggingface/<model_type>/`](../huggingface/) instead; if it is
model-agnostic, it belongs under [`../general/`](../general/). See
[`../ptq.md`](../ptq.md) for what each checkpoint mirror does and how it compares
to its general baseline.

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
