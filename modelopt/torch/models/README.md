# Per-model specs

One package per HF `config.model_type`, holding that model's data for each subsystem
(currently `export.py`). Importing `modelopt.torch.models` registers them all.

## Adding a model

Create `<model_type>/__init__.py`:

```python
from . import export
```

and `<model_type>/export.py`:

```python
from ..registry import register
from ..specs import ModelSpec, MoEVariant

register(
    ModelSpec(
        model_type="qwen3_moe",
        moe_variants=(
            MoEVariant(
                block_names=("Qwen3MoeSparseMoeBlock",),
                expert_linear_names=("gate_proj", "down_proj", "up_proj"),
                gate_up_pair=("gate_proj", "up_proj"),
                has_iterable_experts=True,
            ),
        ),
    )
)
```

Then add it to the import list in `__init__.py`, and add a row to
`EXPECTED_MOE_VARIANTS` in `tests/unit/torch/export/test_model_specs.py` — an
exhaustive-table test fails otherwise.

Fill only the sections your model needs; every field is optional except `model_type`.

## Fields

See [`specs.py`](specs.py) — each field is documented on its declaration, and those
docstrings are what the API reference renders. Specs hold data and trivial accessors
only; subsystem logic stays in the subsystem.
