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
* **Backward compatible**: schemas evolve over time. Loading older persisted
  configs against newer schemas must remain deliberate and testable. ModelOpt
  does not yet have a formal compatibility window, but config authors should
  treat compatibility as a schema-design requirement.
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
* ``ModeloptField(...)`` requires every field to declare a default value.
* ``model_dump()`` and ``model_dump_json()`` default to aliases and suppress
  Pydantic serialization warnings.
* Mapping-style access, such as ``cfg["field"]``, ``cfg.get("field")``,
  ``cfg.items()``, and ``cfg.update({...})``, keeps config objects compatible
  with existing dict-oriented code.
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

Not every reusable config shape needs its own top-level config class. Some
YAML fragments are validated by narrower schema contracts:

* Pydantic model classes work for object snippets such as one quantizer rule.
* ``list[T]`` aliases work for list snippets such as a group of quantizer rules.
* unions and other Pydantic ``TypeAdapter``-compatible annotations can be used
  when the reusable data shape is a typed container rather than a standalone
  model class.

The important invariant is that the schema lives in Python, while YAML remains
data.


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
owning API often supplies schema context directly:

.. code-block:: python

   from modelopt.recipe import load_config
   from modelopt.torch.quantization.config import QuantizeConfig

   data = load_config("configs/ptq/presets/model/fp8", schema_type=QuantizeConfig)
   cfg = QuantizeConfig.model_validate(data)

``schema_type`` has one narrow loader responsibility: it provides typing context
for import resolution, especially for deciding whether a list import should
append one element or splice several elements. It is not a blanket request to
validate a top-level file. Top-level validation is performed by the owning
config object, or by ``load_config()`` when the top-level YAML file itself
contains ``# modelopt-schema: ...``.


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
8. Validate the top-level payload if the file declares ``modelopt-schema``.
9. Return resolved plain Python ``dict`` or ``list`` data.

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

   data = load_config("configs/ptq/presets/model/fp8", schema_type=QuantizeConfig)
   cfg = QuantizeConfig.model_validate(data)

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


Schema evolution
================

Backward compatibility is a schema concern. When a persisted config outlives the
code version that produced it, a newer schema must either accept it or reject it
with a clear migration path.

Use these rules when evolving config schemas:

* Prefer additive fields with defaults over required fields with no default.
* Keep validators tolerant of older spellings when a rename is in flight.
* Normalize legacy forms in ``mode="before"`` validators, then store the
  canonical form in ``model_dump()`` output.
* Avoid changing the meaning of an existing key. Add a new key when semantics
  change materially.
* Add tests that load representative old plain-data configs against the new
  schema.

ModelOpt does not yet define a formal compatibility window for every config
surface, so schema authors should document compatibility-sensitive changes in
the owning feature area.


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

The main alternative is to move more composition knowledge into Python, either
through hard-coded fragment registries, Python-owned name-to-file mappings, or
factory-style configs. Those approaches are useful for object construction, but
they make ordinary YAML reuse depend on Python edits or make Python callables
part of the canonical config representation.

ModelOpt uses a small YAML DSL instead: each file declares its own imports,
references them with ``$import``, and resolves to plain data before validation.
This keeps the import graph self-describing, lets config authors add reusable
fragments as YAML, and still validates every resolved value against Python
schemas.


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

The loader raises ``ValueError`` for invalid composition, including:

* ``imports`` is not a mapping;
* an import path is empty or cannot be resolved;
* a ``$import`` reference is not listed in the file-local ``imports`` mapping;
* an imported snippet does not declare ``modelopt-schema``;
* a schema path does not resolve under ``modelopt.``;
* an imported snippet does not validate against its declared schema;
* a list import has no typed containing list;
* a list import schema is neither the containing list schema nor its element
  schema;
* a circular import is detected.

These failures are load-time errors by design. A composed config should either
resolve to valid plain data or fail before the owning optimization pass starts.


Consumers of the config system
==============================

The config system is shared infrastructure. Current consumers include:

* lower-level optimization configs such as PTQ ``QuantizeConfig``;
* built-in YAML config snippets under ``modelopt_recipes/configs``;
* higher-level recipes, which package metadata together with one or more
  type-specific config sections.

Recipes do not define separate config semantics. ``load_recipe()`` is a
consumer-specific wrapper: it uses ``load_config()`` to resolve YAML, supplies
schema context for each recipe section, and then constructs a typed recipe
object. The general contract remains the same: YAML authoring data resolves to
plain Python data, and Python schemas validate the result.


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
