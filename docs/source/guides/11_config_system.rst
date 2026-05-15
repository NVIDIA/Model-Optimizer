.. _modelopt-config-system:

ModelOpt Config System
######################

ModelOpt configs use Python types as the contract and YAML as the portable data
representation. A YAML file is loaded into ordinary Python ``dict``/``list``
data, optional YAML composition is resolved, and the result is validated by the
owning Pydantic-compatible schema.

The config system is intentionally general. Quantization configs, reusable YAML
snippets, and recipes are all consumers of the same lower-level semantics.
Recipes are one of the main applications; for the recipe-specific authoring
workflow, see :ref:`recipes`.

.. contents:: On this page
   :local:
   :depth: 2


Requirements
============

The core configuration system has four required properties and one optional
authoring feature:

* **Typed / schematized**: each config surface has an explicit Python type
  contract. Concrete model configs inherit from
  :class:`~modelopt.torch.opt.config.ModeloptBaseConfig`; reusable container
  shapes can use Pydantic-compatible type aliases such as
  ``list[QuantizerCfgEntry]``.
* **Validated**: invalid values fail at load or schema-construction time.
  Type errors, range violations, and unknown fields surface as Pydantic
  validation errors instead of being silently ignored.
* **Persistent**: a resolved config can be serialized as plain YAML/JSON data,
  and the same plain data can be embedded in a PyTorch checkpoint and restored
  against the schema.
* **Composable YAML**: shared fragments such as numeric formats and list units
  can be defined once and referenced from multiple YAML files. This is optional
  authoring convenience, not a correctness requirement.

These requirements split the system into three layers:

* Python/Pydantic-compatible schemas define what is valid.
* YAML stores the user-facing config data.
* The loader resolves YAML conveniences, returns plain data, and invokes schema
  validation where the file itself declares a schema.


Schema layer
============

``ModeloptBaseConfig`` is the common base class for structured ModelOpt config
objects:

.. code-block:: python

   class ModeloptBaseConfig(BaseModel):
       model_config = PyDanticConfigDict(extra="forbid", validate_assignment=True)

The base class adds ModelOpt-specific behavior on top of Pydantic:

* ``extra="forbid"`` rejects unknown keys by default.
* ``validate_assignment=True`` revalidates field updates after construction.
* ``ModeloptField(...)`` is a thin wrapper over Pydantic ``Field`` that asserts a
  default value is supplied, so every config field is constructible without
  explicit arguments.
* ``model_dump()`` and ``model_dump_json()`` default to ``by_alias=True`` and
  ``warnings=False``, so serialized output uses the documented field aliases
  and Pydantic serializer warnings are suppressed.
* ``ModeloptBaseConfig`` inherits from ``collections.abc.MutableMapping``, so
  config objects can be used wherever dict-style access is expected:
  ``cfg["field"]`` / ``cfg["field"] = value``, ``cfg.get("field")``,
  ``key in cfg``, ``len(cfg)``, ``iter(cfg)``, ``cfg.keys()``, ``cfg.values()``,
  ``cfg.items()``, ``cfg.update({...})``, plus the ``MutableMapping`` mixins
  ``pop``, ``setdefault``, and ``popitem``. Keys use aliases when defined.
  ``cfg["unknown"] = ...`` raises ``KeyError`` rather than silently adding a
  new key; ``del cfg["field"]`` raises ``TypeError`` because schema fields are
  not removable.
* ``__init_subclass__`` registers each config subclass with PyTorch safe
  globals so config objects can be deserialized by ``torch.load`` with
  ``weights_only=True``.

A typical config schema is a regular Pydantic model with field validators:

.. code-block:: python

   class QuantizeConfig(ModeloptBaseConfig):
       quant_cfg: QuantizeQuantCfgType = ModeloptField(
           default=[{"quantizer_name": "*", "cfg": {"num_bits": 8, "axis": None}}],
           title="Quantization configuration",
           validate_default=True,
       )
       algorithm: QuantizeAlgoCfgType = ModeloptField(
           default="max",
           title="Calibration algorithm",
           validate_default=True,
       )

       @field_validator("quant_cfg", mode="before")
       @classmethod
       def normalize_quant_cfg(cls, v):
           return normalize_quant_cfg_list(v) if isinstance(v, (list, dict)) else v

Not every reusable config shape needs its own top-level config class. Any
type that Pydantic's ``TypeAdapter`` can validate is acceptable as a snippet
schema:

* Pydantic model classes (``ModeloptBaseConfig`` subclasses or other
  ``BaseModel`` subclasses) for object snippets such as a single quantizer
  rule (``QuantizerCfgEntry``) or a numeric format
  (``QuantizerAttributeConfig``).
* ``list[T]`` aliases for list snippets. For example,
  ``QuantizerCfgListConfig`` is defined as ``list[QuantizerCfgEntry]``.
* ``TypedDict`` and ``list[TypedDict]`` shapes when a plain dict layout is the
  natural representation. These return validated dict/list data rather than
  model instances.
* Unions and other ``TypeAdapter``-compatible annotations when the reusable
  data shape is a typed container rather than a standalone model class.

The important invariant is that the schema lives in Python, while YAML remains
data. Snippet schemas are validation contracts; they are not arbitrary Python
execution hooks.


Validation model
================

Validation happens at two boundaries.

Imported snippets
-----------------

Every file referenced by a YAML ``imports`` block is a reusable snippet. It must
include a ``# modelopt-schema: ...`` comment in the initial comment preamble:

.. code-block:: yaml

   # modelopt-schema: modelopt.torch.quantization.config.QuantizerAttributeConfig
   num_bits: e4m3
   axis:

The loader resolves the schema path, validates the resolved snippet payload with
Pydantic ``TypeAdapter``, and only then exposes that snippet to the importing
file. This makes snippets independently reviewable and prevents a malformed
shared fragment from being copied into many configs silently.

Schema paths are intentionally restricted:

* they must resolve under the ``modelopt.`` package;
* they must point at a Pydantic-compatible type;
* they are validation contracts, not arbitrary Python execution hooks.

Top-level configs
-----------------

Top-level user configs do not always need a ``modelopt-schema`` comment. The
owning API often supplies schema context directly through ``schema_type=``:

.. code-block:: python

   from modelopt.recipe import load_config
   from modelopt.torch.quantization.config import QuantizeConfig

   cfg = load_config("configs/ptq/presets/model/fp8", schema_type=QuantizeConfig)
   # cfg is a validated QuantizeConfig instance.

An *effective schema* is selected from the explicit ``schema_type`` argument
and the file's ``# modelopt-schema: ...`` comment, with ``schema_type``
winning when both are present. When an effective schema exists, it serves
two purposes:

* It guides import resolution, especially deciding whether a list import
  should append one element or splice several elements.
* It validates the resolved payload and returns it as an instance of that
  schema — a Pydantic model instance for ``BaseModel`` schemas, or a
  validated ``dict`` / ``list`` for ``TypedDict`` and ``list[TypedDict]``
  schemas.

When neither a ``schema_type`` argument nor a schema comment is supplied,
``load_config()`` returns the resolved payload as plain ``dict`` or ``list``
data without validation.


YAML loading
============

The general loader lives in ``modelopt.torch.opt.config_loader`` and is exported
through ``modelopt.recipe.load_config``. It is intentionally below the recipe
layer so quantization and other core config modules can use it without depending
on recipes.

``load_config(path, schema_type=...)`` performs this flow:

1. Locate the YAML file. Filesystem paths are checked first; if the path is
   relative and not found locally, the built-in ``modelopt_recipes`` package is
   checked. ``.yml`` and ``.yaml`` suffixes may be omitted.
2. Read the optional ``# modelopt-schema: ...`` comment preamble.
3. Parse one YAML document, or two documents when a list-valued snippet also
   needs an ``imports`` declaration.
4. Convert ``eXmY`` strings in ``num_bits`` and ``scale_bits`` fields into
   ``(X, Y)`` tuples.
5. Resolve a file-local ``imports`` mapping.
6. Recursively resolve nested imports, detect circular imports, and validate
   imported snippets against their declared schemas.
7. Walk the YAML tree and replace ``$import`` references.
8. Select the effective top-level schema (``schema_type=`` argument wins over
   ``# modelopt-schema:`` comment when both are present).
9. If an effective schema exists, validate the resolved payload and return a
   schema instance (a Pydantic model, or a validated ``dict`` / ``list`` for
   ``TypedDict``-shaped schemas); otherwise return the plain resolved data.

The loader is not a general templating engine. It only understands YAML data,
``imports``, ``$import``, schema comments, and the ``eXmY`` numeric shorthand.
Application-specific CLI or environment overrides should be applied by the
caller before final schema validation.


Self-contained YAML
===================

The simplest YAML config is self-contained and has no cross-file composition:

.. code-block:: yaml

   algorithm: max
   quant_cfg:
     - quantizer_name: '*'
       enable: false
     - quantizer_name: '*weight_quantizer'
       cfg:
         num_bits: e2m1
         block_sizes:
           -1: 16
           type: dynamic
           scale_bits: e4m3

This is the baseline format. YAML stores values; Python schemas define and
validate the allowed shape.

Self-contained YAML is the right choice when a config is small, used once, or
clearer without indirection. Composable YAML is for repeated fragments and large
families of related configs.


YAML persistence
================

A loaded config should round-trip through plain data. After loading and
validation, serialize the resolved config rather than the authoring-time YAML:

.. code-block:: python

   import yaml

   from modelopt.recipe import load_config
   from modelopt.torch.quantization.config import QuantizeConfig

   cfg = load_config("configs/ptq/presets/model/fp8", schema_type=QuantizeConfig)

   with open("resolved_quantize.yaml", "w", encoding="utf-8") as f:
       yaml.safe_dump(cfg.model_dump(), f)

The output is fully materialized plain data. YAML comments, ``imports`` blocks,
``$import`` markers, and schema comments are authoring metadata; they do not
survive in the resolved dump. This is intentional. Resolved dumps are suitable
for bug reports, reproducibility artifacts, and diffs across runs.

Reloading a resolved dump is the same operation as any other load: parse plain
YAML data and validate it against the schema.


Checkpoint persistence
======================

Configs embedded in checkpoints should use the same plain-data contract. Store
``cfg.model_dump()`` in the checkpoint and restore it with the owning schema:

.. code-block:: python

   import torch

   state = {
       "model": model.state_dict(),
       "modelopt_state": {
           "quantize_config": cfg.model_dump(),
       },
   }
   torch.save(state, "checkpoint.pt")

   loaded = torch.load("checkpoint.pt", weights_only=True)
   restored_cfg = QuantizeConfig.model_validate(
       loaded["modelopt_state"]["quantize_config"]
   )

Persisting plain data keeps checkpoints independent of the original YAML files
and of the authoring-time import graph. Future readers need the schema, not the
source snippets.

``ModeloptBaseConfig`` also registers subclasses as PyTorch safe globals, which
allows config objects to participate in safe deserialization. Plain-data
persistence remains the most portable form because it is easy to inspect, diff,
and migrate.


Composable YAML
===============

Python already has composition through variables, functions, imports, and
mutation. YAML does not. ModelOpt's YAML composition layer exists so repeated
YAML fragments can be shared without moving the canonical config into Python.

Typical repeated fragments include:

* one numeric format used by several quantizer entries;
* one complete quantizer-entry snippet reused in many configs;
* a list of quantizer entries reused as a unit;
* a snippet that depends on another snippet;
* related variants such as dynamic and static numeric formats.

The chosen design is a small YAML-native DSL: a file-local ``imports`` mapping
binds names to YAML files, and inline ``$import`` references insert those
resolved snippets into the data tree. Python remains responsible for schema
validation; YAML remains data.


Alternatives considered
-----------------------

Several other approaches can give YAML configs some form of composability.
Each was considered and rejected for ModelOpt's library-of-configs use case:

* **Plain YAML anchors and aliases** reuse data inside one file but do not
  compose across files and do not validate fragments independently.
* **Hard-coded Python registries** map well-known names like ``nvfp4`` to
  Python-side constants. Adding a new fragment requires a Python edit, and
  YAML can only reference what Python has pre-declared.
* **YAML files with Python-side name-to-file mappings** keep fragment data in
  YAML, but the registration of each fragment still lives in Python. Adding a
  new fragment requires both a YAML file and a Python edit.
* **General config frameworks such as OmegaConf and Hydra** provide deep merge
  and ``${...}`` interpolation, but there is no native cross-file include
  keyword, no native list-concatenation primitive, and the list
  append-vs-splice rule must still come from somewhere ModelOpt-specific.
  OmegaConf can be useful at the edges (for example for CLI dotted overrides
  or environment-variable substitution applied after import resolution) but
  is not sufficient as the composition primitive.
* **Python factory systems such as Fiddle or nemo_run** ``_factory_`` make
  Python callables the canonical config representation. They are a good fit
  when the audience is exclusively Python engineers and configs primarily
  build runnable objects. They are a poor fit for ModelOpt because reusable
  fragments are typically small typed values (numeric formats, quantizer-list
  entries), persisting a factory-based config loses provenance unless the
  on-disk format ties to Python qualified names, and Fiddle-style
  ``@auto_config`` cannot return bare ``dict`` or ``list`` values without a
  wrapper class that duplicates the Pydantic schema.

ModelOpt uses a small YAML DSL instead: each file declares its own imports,
references them with ``$import``, and resolves to plain data before validation.
This keeps the import graph self-describing, lets config authors add reusable
fragments as YAML without Python edits, and still validates every resolved
value against Python schemas. The on-disk representation is plain YAML data,
so persisted configs do not depend on Python qualified names.


Import declarations
-------------------

Imports are declared once per YAML file:

.. code-block:: yaml

   imports:
     nvfp4: configs/numerics/nvfp4
     kv_fp8: configs/ptq/units/kv_fp8

The names are scoped to that file. An imported snippet may declare its own
``imports`` block, and those names are scoped to the snippet file. Recursive
imports are resolved depth-first. Circular imports are detected using canonical
resolved paths and fail with ``ValueError``.

A file that declares no ``imports`` may not contain ``$import`` markers. This
keeps authoring mistakes explicit: an unknown reference fails instead of being
left as literal data.


Dict imports
------------

When ``$import`` appears inside a mapping, the imported mapping is copied into
the current mapping. Inline keys override imported keys at that same mapping
level:

.. code-block:: yaml

   cfg:
     $import: nvfp4
     block_sizes:
       -1: 16
       type: static
       scale_bits: e4m3

Multiple imports are applied in order, then inline keys are applied last:

.. code-block:: yaml

   cfg:
     $import: [base_format, override_format]
     axis: 0

The merge is shallow at the mapping where ``$import`` appears. If one nested
leaf changes, provide the complete nested value inline or define a named snippet
for that variant. This avoids hidden deep-merge rules that are hard to review.


List imports
------------

List imports are type-directed. For a containing list with schema ``list[T]``:

* importing a snippet with schema ``list[T]`` splices all imported entries into
  the containing list;
* importing a snippet with schema ``T`` appends the imported object as a single
  list element;
* importing any other schema raises an error;
* importing into an untyped list raises an error.

Example:

.. code-block:: yaml

   quant_cfg:
     - $import: base_disable_all          # QuantizerCfgEntry, appended
     - quantizer_name: '*weight_quantizer'
       cfg:
         $import: nvfp4                   # QuantizerAttributeConfig, dict import
     - $import: kv_fp8                    # QuantizerCfgListConfig, spliced

A list-entry import must be a mapping whose only key is ``$import``. If an entry
needs local changes, either write that entry inline or create a snippet for the
variant.


Multi-document list snippets
----------------------------

A YAML file has one root node per document. A list-valued snippet that also
needs an ``imports`` block therefore uses two YAML documents: the first document
holds import declarations, and the second document holds the list payload.

.. code-block:: yaml

   # modelopt-schema: modelopt.torch.quantization.config.QuantizerCfgListConfig
   imports:
     fp8: configs/numerics/fp8
   ---
   - quantizer_name: '*[kv]_bmm_quantizer'
     cfg:
       $import: fp8

Only ``imports`` from the first document is meaningful for a list snippet. The
loader resolves imports in the second document and returns the resolved list.


Composition error model
-----------------------

The loader raises ``ValueError`` for invalid input. The full set of conditions
covers file-shape, schema declaration, and composition rules:

File-shape errors:

* the YAML file cannot be located on the filesystem or in built-in
  ``modelopt_recipes``;
* a YAML file contains more than two documents;
* the root of a single-document file is not a mapping or a list;
* in a two-document file, the first document is not a mapping or the second
  document is neither a mapping nor a list;
* multiple ``# modelopt-schema:`` comments are present in the preamble.

Schema-declaration errors:

* a schema path does not start with ``modelopt.``;
* a schema path is missing a module or attribute component, or it fails to
  resolve to a real Python object;
* an imported snippet does not declare ``modelopt-schema``;
* an imported snippet does not validate against its declared schema.

Composition errors:

* ``imports`` is present but is not a mapping;
* an import path is empty;
* a ``$import`` reference appears in a file that declares no ``imports``;
* a ``$import`` name is not listed in the file-local ``imports`` mapping;
* a dict-form ``$import`` resolves to something other than a dict;
* a list import is used without a typed containing list;
* a list import schema is neither the containing list schema nor its element
  schema;
* a circular import is detected (reported with the import chain).

These failures are load-time errors by design. A composed config should either
resolve to valid plain data or fail before the owning optimization pass starts.


Consumers of the config system
==============================

The config system is shared infrastructure. Current consumers include:

* lower-level optimization configs such as PTQ ``QuantizeConfig``;
* built-in YAML config snippets under ``modelopt_recipes/configs`` (numeric
  formats, reusable quantizer-entry units, model-level presets);
* higher-level recipes under ``modelopt_recipes/general`` and
  ``modelopt_recipes/models``, which package metadata together with one or
  more type-specific config sections.

Recipes do not define separate config semantics. ``load_recipe()`` is a
consumer-specific wrapper that uses ``load_config()`` to resolve YAML, supplies
schema context for each recipe section, and returns a validated
``ModelOptRecipeBase`` subclass instance. Both ``metadata`` and ``quantize``
sections are required:

* A **file recipe** is a single YAML file that declares
  ``# modelopt-schema: modelopt.recipe.config.ModelOptPTQRecipe`` and is loaded
  with ``schema_type=ModelOptPTQRecipe``; both ``metadata`` and ``quantize``
  sections live in the same file. ``load_config()`` returns a fully validated
  ``ModelOptPTQRecipe`` instance directly.
* A **directory recipe** is a directory containing ``metadata.yaml`` /
  ``metadata.yml`` and ``quantize.yaml`` / ``quantize.yml``. Each file is
  loaded with its own schema (``RecipeMetadataConfig`` and ``QuantizeConfig``,
  both ``ModeloptBaseConfig`` subclasses), and the recipe is assembled as
  ``ModelOptPTQRecipe(metadata=..., quantize=...)`` from the validated
  sections.

The general contract remains the same: YAML authoring data resolves to plain
Python data, Python schemas validate the result, and validated configs are
returned as schema instances. Callers can move between dict and model views
through ``cfg.model_dump()`` and ``Schema.model_validate(data)``.


Authoring guidelines
====================

When adding config schemas or YAML files:

* Put the canonical schema in Python, not in YAML comments or loader logic.
* Use ``ModeloptBaseConfig`` for structured config objects that need methods,
  defaults, and validators.
* Use ``ModeloptBaseConfig`` subclasses or typed aliases for reusable snippets.
* Prefer self-contained YAML unless a fragment is reused or factoring materially
  improves reviewability.
* Add ``# modelopt-schema: ...`` to every file that can be referenced from an
  ``imports`` block.
* Keep top-level user config files free of schema comments unless they are also
  intended to be imported as snippets.
* Use a concrete typed list schema for list snippets so append-vs-splice
  behavior is unambiguous.
* Serialize resolved configs with ``model_dump()`` for long-term artifacts.
* Store plain config data, not authoring-time YAML paths, in checkpoints.
* Do not parse ModelOpt config YAML with raw YAML APIs in application code. Use
  ``load_config()`` or a higher-level API built on it so imports, schema checks,
  and ``eXmY`` conversion are applied consistently.
