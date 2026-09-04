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
`test_specs_vs_transformers.py` needs no edit: it reads which models to check, and
which to skip, from the registry.

The file is named for what it holds, not for who reads it: a model's spec is general
model data, and export is only its first consumer.

## Sections

A `ModelSpec` holds one attribute per section, each `None` unless the model fills it:

| Section | Holds | Leave `None` when |
|---|---|---|
| `moe_spec` (`MoESpec`) | MoE architecture facts — block classes, expert projection naming | the model is dense |
| `export_spec` (`ExportSpec`) | Per-model data of the unified HF export path | export needs nothing model-specific |

`None` means "the model has nothing to say about this", which stays distinct from a
filled-in-but-empty section.

Alongside the required `model_type`, two fields say where the classes this spec names
come from, and they must agree:

| Field | Meaning |
|---|---|
| `modeling_source` | `"transformers"` (default), or `"remote_code"` when the code ships with the checkpoint and needs `trust_remote_code=True` |
| `min_transformers_version` | Earliest `transformers` release whose definitions match this spec; `None` for a `remote_code` model |

Clamp `min_transformers_version` at the repo's minimum supported transformers (`tf_min`
in `noxfile.py`, currently `4.57`) — a model older than the floor records the floor,
since nothing older is installed or tested. A model added later records its own release.
The pair is what lets `test_specs_vs_transformers.py` assert rather than skip: it can
tell a model that is legitimately absent on an older transformers from a spec that no
longer matches reality.

## Fields

See [`specs.py`](specs.py) — each field is documented on its declaration, and those
docstrings are what the API reference renders. Specs hold data and trivial accessors
only; subsystem logic stays in the subsystem.
