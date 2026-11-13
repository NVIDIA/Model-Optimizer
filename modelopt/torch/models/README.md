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
from ..specs import ExportSpec, ModelSpec, MoESpec, register

__all__: list[str] = []

register(
    ModelSpec(
        model_type="qwen3_moe",
        min_transformers_version="5.5",
        # modelopt policy, not a model fact: this model is validated for grouped export.
        export_spec=ExportSpec(grouped_expert_export=True),
        moe_spec=MoESpec(
            block_names=("Qwen3MoeSparseMoeBlock",),
            expert_linear_names=("gate_proj", "down_proj", "up_proj"),
            gate_up_pair=("gate_proj", "up_proj"),
        ),
    )
)
```

Then add it to the import list in `__init__.py`. Two tables in
`tests/unit/torch/models/` are checked for exhaustiveness and fail until you extend
them:

- `EXPECTED_MOE_LAYOUTS` in `test_model_specs.py` — pins the spec's values.
`test_specs_vs_transformers.py` needs no edit: it reads which models to check, and
which to skip, from the registry.

A model type that is not a lowercase module name (`NemotronH_Nano_VL_V2`) gets a snake_case
package name (`nemotron_h_nano_vl_v2`), as transformers puts `EvollaModel` in `evolla`. No code
derives a package from a model type: the spec's `model_type` stays the exact HF string, and every
reference names the package explicitly.

The file is named for what it holds, not for who reads it: a model's spec is general
model data, and export is only its first consumer.

`__all__` is empty because these modules export nothing — they exist so that importing
them runs `register()`. Declaring it keeps the helper imports above out of any
star-import. A value shared between two model types is private and imported explicitly
(`gemma4_text` reuses `gemma4`'s `_GEMMA4_MOE_SPEC`); sub-model types of one family are
an intra-family detail, not a package API.

## PTQ modeling

Quantized-module wrappers and registrations that only one model needs (e.g. Llama4's fused
BMM experts, Step's expert-indexed `MoELinear`) live in `<model_type>/modeling_ptq.py`; generic
ones (fused/sequential MoE auto-detection, attention, `FP8Linear`) stay in
`modelopt/torch/quantization/plugins/huggingface.py`. Nothing imports a `modeling_ptq` up front:
before converting a model or finding its decoder layers (which loading with FSDP2 does first),
quantization imports those of its model types with `import_model_modules`, which finds them by
file name. A model's types are its root
`config.model_type` and those of the configs nested in it, so a VLM also gets its language
model's. A new `modeling_ptq` therefore needs no list entry, and the plugins name no model.
Leave the model's `__init__.py` importing only `specs`: `modeling_ptq` pulls in quantization and
transformers, which `import modelopt.torch.models` must not.

A `modeling_ptq.py` registers on import — static classes via `QuantModuleRegistry.register`,
imported from transformers at module scope; remote-code classes via a callback added to
`CUSTOM_MODEL_PLUGINS`; a KV-cache wrapper for attention outside transformers' attention
interface (T5's) by appending it to the plugin's `_QUANT_ATTENTION_WRAPPERS`, which tries it
after the interface wrapper and before the AST-based fallback — and exports nothing
(`__all__ = []`). Every package still has a `specs.py`, even when its spec only records
`model_type` and where the modeling code comes from. A model type that shares another type's
PTQ code (`step3p7` uses `step3p5`'s, `llama4_text` uses `llama4`'s) still gets a
`modeling_ptq.py` that imports the shared one, so loading it by model type works. Falcon's
original remote-code types (`refined_web`, `refined_web_model`) get only that, without a spec.

vLLM's model-specific wrappers follow the same pattern in `<model_type>/modeling_ptq_vllm.py`,
on top of the generic ones in the vLLM plugin modules
(`modelopt/torch/quantization/plugins/vllm*.py`), and a vLLM model's are imported the same way,
by the model type in its HF config. A package that only holds vLLM modeling (`glm5_next`) has
no `specs.py`.

## Sections

A `ModelSpec` holds one attribute per section, each `None` unless the model fills it:

| Section | Holds | Leave `None` when |
|---|---|---|
| `moe_spec` (`MoESpec`) | MoE architecture facts only — block classes, expert projection naming, which layout that naming describes | the model is dense |
| `export_spec` (`ExportSpec`) | Per-model data *and policy* of the unified HF export path, e.g. whether grouped expert export is validated for this model | export needs nothing model-specific |
| `ep_spec` (`EPSpec`) | Per-model data of the expert-parallel PTQ path: the `base_model_ep_plan` to attach and the config it goes on, the transformers release it needs, or why the model cannot be expert-parallel yet | the model is dense, or stock transformers already ships all of it |
| `ptq_spec` (`PTQSpec`) | Per-model facts of the Hugging Face PTQ flow (`examples/hf_ptq`): how it loads (device map, precision keyword), whether it is a VLM wrapper, why its generation preview is skipped or decoded as it is, why `--use_fsdp2` refuses it. Each field replaces a check on the model's name in the example; read it through `get_ptq_spec` | the flow's generic behavior fits the model |

`None` means "the model has nothing to say about this", which stays distinct from a
filled-in-but-empty section.

Alongside the required `model_type`, two fields say where the classes this spec names
come from, and they must agree:

| Field | Meaning |
|---|---|
| `modeling_source` | `"transformers"` (default), or `"remote_code"` when the code ships with the checkpoint and needs `trust_remote_code=True` |
| `min_transformers_version` | Earliest `transformers` release whose definitions match this spec; `None` for a `remote_code` model |

Clamp `min_transformers_version` at the repo's minimum supported transformers (`tf_min`
in `noxfile.py`, currently `5.5`) — a model older than the floor records the floor,
since nothing older is installed or tested. A model added later records its own release.
The pair is what lets `test_specs_vs_transformers.py` assert rather than skip: it can
tell a model that is legitimately absent on an older transformers from a spec that no
longer matches reality.

## Fields

See [`specs.py`](specs.py) — each field is documented on its declaration, and those
docstrings are what the API reference renders. Specs hold data and trivial accessors
only; subsystem logic stays in the subsystem.

## Model-specific behavior

Keep architecture-specific loading and calibration helpers alongside the model's
`specs.py` (for example, `nemotron_h/mtp.py`). The `hf.py` dispatcher loads these
helpers on demand; examples call its loading context and calibration hook.

A model type's helper modules, registered in `hf.py`'s `_PLUGINS` (one module or several), define
any of these hooks, all optional, and each lists exactly the ones it defines in `__all__`. The
dispatcher rejects any other name there, and two of a type's modules defining one hook, so a
misspelled or doubled hook fails instead of being skipped.

| Hook | Called by | Default without it |
|---|---|---|
| `prepare_for_loading(checkpoint_path, trust_remote_code, *, model_class)` | `prepare_model_for_loading` | no context |
| `has_mtp_weights(checkpoint_path)` | `checkpoint_has_mtp` | `False` |
| `prefers_bundled_code(hf_config, checkpoint_path, config_kwargs)` | `prefers_bundled_code` | `False` |
| `prepare_for_calibration(model)` | `prepare_model_for_calibration` | nothing |
| `forward_calibration_batch(model, batch)` | `forward_calibration_batch` | the caller's plain forward (returns `False`) |
| `load_calibration_processor(checkpoint_path, trust_remote_code)` | `load_calibration_processor` | `None`: the model brings no calibration data |
| `make_calibration_dataloader(model, processor, dataset_names, calib_size, batch_size, device)` | `make_calibration_dataloader` | `None`: the caller's text or image-text data |
| `calibration_sample_input(model)` | `calibration_sample_input` | `None`: the caller builds a token-id sample |

Do not import runtime helpers from a model package's `__init__.py`: registering
specs must remain independent of optional Transformers and calibration imports.
