# Model-specific recipes for Hugging Face models

This folder holds model-optimization recipes (e.g. PTQ recipes) whose
behavior is tied to a **specific Hugging Face model architecture or model instance**.

## Folder structure

Recipes are categorized by the Hugging Face `model_type` string — the
value of the top-level `model_type` field in the model's `config.json`
(or, for multimodal configs, the `text_config.model_type` of the inner
language model). Use the exact `model_type` as the directory name:

```text
modelopt_recipes/huggingface/
  <model_type>/
    <task>/
      <recipe>.yaml
      [<recipe>.<aux>.yaml]          # optional snippet helpers (see below)
      [README.md]                    # optional; describes what's model-specific
```

`<task>` is the model-optimization workflow the recipe targets (e.g.
`ptq` for post-training quantization).

Selecting a recipe at runtime uses the path relative to
`modelopt_recipes/`, e.g.
`--recipe huggingface/<model_type>/<task>/<recipe>`.

### Verifying a model's `model_type`

The authoritative source for a model's `model_type` is the released
checkpoint's `config.json` on the Hugging Face Hub, e.g.
`https://huggingface.co/<org>/<model>/raw/main/config.json`. The
`transformers` library's per-model `configuration_<name>.py` files also
hardcode the `model_type` string. Do not guess — confirm against one of
these sources before placing a recipe.

### Sharing content across recipes

When the same body is reused by multiple recipes — for example, one
recipe that applies to several `model_type`s, or several recipes that
share a sub-block — extract the reused portion into a sibling
**snippet** file with a `# modelopt-schema:` header and have each
recipe `$import` it. The recipe wrappers stay thin; the shared body
lives in one place.

Name snippet files so they are obviously not runnable recipes, e.g.
include the field name the snippet represents as a secondary suffix
(`<recipe>.<field>.yaml`). The snippet lives next to whichever recipe
is its natural canonical home; other importers reference it by the
same relative path under `modelopt_recipes/`.

### Per-family nested layout for specific model variants

If a recipe is tuned for one specific released model rather than every
checkpoint under a `model_type`, nest the model name as an extra level:

```text
<model_type>/
  <specific_model>/
    <task>/
      <recipe>.yaml
```

### Per-folder READMEs

Each `<task>/` folder may contain a short `README.md` describing exactly
what is model-specific about each recipe (the algorithm override, the
disabled-quantizer pattern, etc.) so reviewers and users do not have to
diff the YAML against the generic presets to see the intent.
