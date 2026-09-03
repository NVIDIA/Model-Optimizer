# Per-model specs

One package per HF `config.model_type`, holding everything modelopt knows about that
model. Importing `modelopt.torch.models` registers them all.

## Adding a model

Create `<model_type>/__init__.py`:

```python
from . import specs
```

and `<model_type>/specs.py`:

```python
from ..registry import register
from ..specs import ModelSpec, MoESpec, MoEVariant

register(
    ModelSpec(
        model_type="qwen3_moe",
        moe_spec=MoESpec(
            moe_variants=(
                MoEVariant(
                    block_names=("Qwen3MoeSparseMoeBlock",),
                    expert_linear_names=("gate_proj", "down_proj", "up_proj"),
                    gate_up_pair=("gate_proj", "up_proj"),
                    has_iterable_experts=True,
                ),
            )
        ),
    )
)
```

Then add it to the import list in `__init__.py`. Two tables in
`tests/unit/torch/models/` are checked for exhaustiveness and fail until you extend
them:

- `EXPECTED_MOE_VARIANTS` in `test_model_specs.py` — pins the spec's values.
- `IN_TRANSFORMERS` (or `UNCHECKABLE`, with a reason) in
  `test_specs_vs_transformers.py` — says whether the spec can be validated against
  the transformers definition.

The file is named for what it holds, not for who reads it: a model's spec is general
model data, and export is only its first consumer.

## Sections

A `ModelSpec` holds one attribute per section, each `None` unless the model fills it:

| Section | Holds | Leave `None` when |
|---|---|---|
| `moe_spec` (`MoESpec`) | MoE architecture facts — block classes, expert projection naming | the model is dense |
| `export_spec` (`ExportSpec`) | Per-model data of the unified HF export path | export needs nothing model-specific |

`None` means "the model has nothing to say about this", which stays distinct from a
filled-in-but-empty section. Only `model_type` is required.

## Fields

See [`specs.py`](specs.py) — each field is documented on its declaration, and those
docstrings are what the API reference renders. Specs hold data and trivial accessors
only; subsystem logic stays in the subsystem.
