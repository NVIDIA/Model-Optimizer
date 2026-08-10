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

Then add it to the import list in `__init__.py` and a row to `EXPECTED_MOE_VARIANTS` in
`tests/unit/torch/export/test_model_specs.py` (an exhaustive-table test fails otherwise).

## Fields

| Field | Meaning |
|---|---|
| `model_type` | HF `config.model_type`. Unique across the registry. |
| `moe_variants` | MoE block layouts. Several when one model type materializes differently (e.g. Mixtral across transformers generations). |
| `block_names` | MoE block class names, matched case-insensitively and **exactly** against the module's MRO. |
| `expert_linear_names` | Expert projection names, e.g. `("gate_proj", "down_proj", "up_proj")`. |
| `has_iterable_experts` | True when experts are per-expert submodules `get_experts_list` can group; False for stacked/fused layouts. |
| `gate_up_pair` | The (gate, up) pair serving engines fuse into `gate_up_proj`. `None` if non-gated or already fused. |
| `pqs_fuse_rules` | AWQ `pre_quant_scale` fusion, as `(class_substrings, fuse_into, fuse_from)`. |
| `weight_plus_one_norm_names` | Norm classes storing `w - 1` (effective scale is `weight + 1`). |

Specs hold data and trivial accessors only — subsystem logic stays in the subsystem.
