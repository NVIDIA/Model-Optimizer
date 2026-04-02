.. _recipes:

Recipes
#######

A **recipe** is a declarative YAML specification that fully describes how to optimize a model.
Recipes decouple optimization settings from Python code, enabling reuse, sharing, version
control, and reproducibility.  Instead of editing Python scripts to change quantization
parameters, you author (or select) a recipe file and pass it to the ModelOpt tooling.

.. contents:: On this page
   :local:
   :depth: 2


Motivation
==========

Without recipes, optimization settings are scattered across command-line arguments, Python
constants, and ad-hoc code edits.  This makes it difficult to:

* **Reproduce** a published result -- the exact configuration is buried in script arguments.
* **Share** a configuration -- there is no single artifact to hand off.
* **Version-control** changes -- diffs are mixed in with unrelated code changes.
* **Onboard new models** -- inference engineers must read source code to discover which
  settings to tweak.

Recipes solve these problems by capturing **all** the configuration needed to optimize a
model in a single YAML file (or a small directory of files).


Design overview
===============

The recipe system is part of the :mod:`modelopt.recipe` package and consists of three
layers:

1. **Recipe files** -- YAML documents stored in the ``modelopt_recipes/`` directory (shipped
   with the package) or on the user's filesystem.
2. **Config loader** -- :func:`~modelopt.recipe.load_config` reads YAML files, resolves
   paths, and performs automatic ``ExMy`` floating-point notation conversion.
3. **Recipe loader** -- :func:`~modelopt.recipe.load_recipe` validates the YAML against
   Pydantic models and returns a typed recipe object ready for use.


Recipe file format
==================

A recipe is a YAML file with two top-level sections: ``metadata`` and a
type-specific configuration section (currently ``ptq_cfg`` for PTQ recipes).

Single-file format
------------------

The simplest form is a single ``.yml`` or ``.yaml`` file:

.. code-block:: yaml

   # modelopt_recipes/general/ptq/fp8_default-fp8_kv.yml

   metadata:
     recipe_type: ptq
     description: FP8 per-tensor weight and activation (W8A8), FP8 KV cache, max calibration.

   ptq_cfg:
     algorithm: max
     quant_cfg:
       - quantizer_path: '*'
         enable: false
       - quantizer_path: '*input_quantizer'
         cfg:
           num_bits: e4m3
           axis:
       - quantizer_path: '*weight_quantizer'
         cfg:
           num_bits: e4m3
           axis:
       - quantizer_path: '*[kv]_bmm_quantizer'
         enable: true
         cfg:
           num_bits: e4m3
       # ... standard exclusions omitted for brevity

Directory format
----------------

For larger recipes or when you want to keep metadata separate from the
quantization configuration, use a directory with two files:

.. code-block:: text

   my_recipe/
     recipe.yml      # metadata section
     ptq_cfg.yml     # ptq_cfg section (quant_cfg + algorithm)

``recipe.yml``:

.. code-block:: yaml

   metadata:
     recipe_type: ptq
     description: My custom NVFP4 recipe.

``ptq_cfg.yml``:

.. code-block:: yaml

   algorithm: max
   quant_cfg:
     - quantizer_path: '*'
       enable: false
     - quantizer_path: '*weight_quantizer'
       cfg:
         num_bits: e2m1
         block_sizes: {-1: 16, type: dynamic, scale_bits: e4m3}
     - quantizer_path: '*input_quantizer'
       cfg:
         num_bits: e4m3
         axis:


Metadata section
================

Every recipe file must contain a ``metadata`` mapping with at least a ``recipe_type`` field:

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Field
     - Required
     - Description
   * - ``recipe_type``
     - Yes
     - The optimization category.  Currently only ``"ptq"`` is supported.
   * - ``description``
     - No
     - A human-readable summary of what the recipe does.


PTQ configuration section
=========================

For PTQ recipes (``recipe_type: ptq``), the ``ptq_cfg`` mapping contains:

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Field
     - Required
     - Description
   * - ``quant_cfg``
     - Yes
     - An ordered list of :class:`~modelopt.torch.quantization.config.QuantizerCfgEntry`
       dicts.  See :ref:`quant-cfg` for the full specification of entries, ordering
       semantics, and atomicity rules.
   * - ``algorithm``
     - No
     - The calibration algorithm: ``"max"`` (default), ``"mse"``, ``"smoothquant"``,
       ``"awq_lite"``, ``"awq_full"``, ``"awq_clip"``, ``"gptq"``, or ``null`` for
       formats that need no calibration (e.g. MX formats).


ExMy floating-point notation
=============================

Recipe files support a convenient shorthand for floating-point bit formats in
``num_bits`` and ``scale_bits`` fields.  Instead of writing a Python tuple, you
write the format name directly:

.. code-block:: yaml

   num_bits: e4m3       # automatically converted to (4, 3)
   scale_bits: e8m0     # automatically converted to (8, 0)

The notation is case-insensitive (``E4M3``, ``e4m3``, ``E4m3`` all work).  The
conversion is performed by :func:`~modelopt.recipe.load_config` when loading any
YAML file, so it works in both recipe files and standalone config files.

Common formats:

.. list-table::
   :header-rows: 1
   :widths: 15 15 70

   * - Notation
     - Tuple
     - Description
   * - ``e4m3``
     - ``(4, 3)``
     - FP8 E4M3 -- standard FP8 weight/activation format
   * - ``e5m2``
     - ``(5, 2)``
     - FP8 E5M2 -- wider dynamic range, used for gradients
   * - ``e2m1``
     - ``(2, 1)``
     - FP4 E2M1 -- NVFP4 weight format
   * - ``e8m0``
     - ``(8, 0)``
     - E8M0 -- MX block scaling format


Built-in recipes
================

ModelOpt ships a library of built-in recipes under the ``modelopt_recipes/`` package.
These are bundled with the Python distribution and can be referenced by their relative
path (without the ``modelopt_recipes/`` prefix).

General PTQ recipes
-------------------

General recipes are model-agnostic and apply to any supported architecture:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Recipe path
     - Description
   * - ``general/ptq/fp8_default-fp8_kv``
     - FP8 per-tensor W8A8, FP8 KV cache, max calibration
   * - ``general/ptq/nvfp4_default-fp8_kv``
     - NVFP4 W4A4 with FP8 KV cache, max calibration
   * - ``general/ptq/nvfp4_mlp_only-fp8_kv``
     - NVFP4 for MLP layers only, FP8 KV cache
   * - ``general/ptq/nvfp4_experts_only-fp8_kv``
     - NVFP4 for MoE expert layers only, FP8 KV cache
   * - ``general/ptq/nvfp4_omlp_only-fp8_kv``
     - NVFP4 for output projection + MLP layers, FP8 KV cache

Model-specific recipes
----------------------

Model-specific recipes are tuned for a particular architecture and live under
``models/<model_name>/``:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Recipe path
     - Description
   * - ``models/Step3.5-Flash/nvfp4-mlp-only``
     - NVFP4 MLP-only for Step 3.5 Flash MoE model


Loading recipes
===============

Python API
----------

Use :func:`~modelopt.recipe.load_recipe` to load a recipe.  The path is resolved
against the built-in library first, then the filesystem:

.. code-block:: python

   from modelopt.recipe import load_recipe, ModelOptPTQRecipe

   # Load a built-in recipe by relative path (suffix optional)
   recipe = load_recipe("general/ptq/fp8_default-fp8_kv")
   assert isinstance(recipe, ModelOptPTQRecipe)

   # The ptq_cfg dict can be passed directly to mtq.quantize()
   import modelopt.torch.quantization as mtq

   model = mtq.quantize(model, recipe.ptq_cfg, forward_loop)

.. code-block:: python

   # Load a custom recipe from the filesystem
   recipe = load_recipe("/path/to/my_custom_recipe.yml")
   model = mtq.quantize(model, recipe.ptq_cfg, forward_loop)

Command-line usage
------------------

The ``hf_ptq.py`` example accepts a ``--recipe`` flag:

.. code-block:: bash

   python examples/llm_ptq/hf_ptq.py \
       --model Qwen/Qwen3-8B \
       --recipe general/ptq/fp8_default-fp8_kv \
       --export_path build/fp8 \
       --calib_size 512 \
       --export_fmt hf

When ``--recipe`` is provided, the script loads the recipe and uses its ``ptq_cfg``
directly, bypassing the ``--qformat`` / ``--kv_cache_qformat`` flags.


Loading standalone configs
--------------------------

:func:`~modelopt.recipe.load_config` loads arbitrary YAML config files with
automatic ``ExMy`` conversion and built-in path resolution.  This is useful
for loading shared configuration fragments:

.. code-block:: python

   from modelopt.recipe import load_config

   cfg = load_config("configs/some_shared_config")


Path resolution
===============

Both :func:`~modelopt.recipe.load_recipe` and :func:`~modelopt.recipe.load_config`
resolve paths using the same strategy:

1. If the path is absolute, use it directly.
2. If relative, check the **built-in recipes library** first
   (``modelopt_recipes/``), probing ``.yml`` and ``.yaml`` suffixes.
3. Then check the **filesystem**, probing the same suffixes.

This means built-in recipes can be referenced without any prefix:

.. code-block:: python

   # These are all equivalent:
   load_recipe("general/ptq/fp8_default-fp8_kv")
   load_recipe("general/ptq/fp8_default-fp8_kv.yml")


Writing a custom recipe
=======================

To create a custom recipe:

1. Start from an existing recipe that is close to your target configuration.
2. Copy it and modify the ``quant_cfg`` entries as needed (see :ref:`quant-cfg`
   for entry format details).
3. Update the ``metadata.description`` to describe your changes.
4. Save the file and pass its path to ``load_recipe()`` or ``--recipe``.

Example -- creating an INT8 per-channel recipe:

.. code-block:: yaml

   # my_int8_recipe.yml
   metadata:
     recipe_type: ptq
     description: INT8 per-channel weight, per-tensor activation.

   ptq_cfg:
     algorithm: max
     quant_cfg:
       - quantizer_path: '*'
         enable: false
       - quantizer_path: '*weight_quantizer'
         cfg:
           num_bits: 8
           axis: 0
       - quantizer_path: '*input_quantizer'
         cfg:
           num_bits: 8
           axis:
       - quantizer_path: '*lm_head*'
         enable: false
       - quantizer_path: '*output_layer*'
         enable: false


Recipe repository layout
========================

The ``modelopt_recipes/`` package is organized as follows:

.. code-block:: text

   modelopt_recipes/
   +-- __init__.py
   +-- general/                    # Model-agnostic recipes
   |   +-- ptq/
   |       +-- fp8_default-fp8_kv.yml
   |       +-- nvfp4_default-fp8_kv.yml
   |       +-- nvfp4_mlp_only-fp8_kv.yml
   |       +-- nvfp4_experts_only-fp8_kv.yml
   |       +-- nvfp4_omlp_only-fp8_kv.yml
   +-- models/                     # Model-specific recipes
   |   +-- Step3.5-Flash/
   |       +-- nvfp4-mlp-only.yaml
   +-- configs/                    # Shared configuration fragments


Recipe data model
=================

Recipes are validated at load time using Pydantic models:

:class:`~modelopt.recipe.config.ModelOptRecipeBase`
   Base class for all recipe types.  Contains ``recipe_type`` and ``description``.

:class:`~modelopt.recipe.config.ModelOptPTQRecipe`
   PTQ-specific recipe.  Adds the ``ptq_cfg`` field (a dict with ``quant_cfg`` and
   ``algorithm``).

:class:`~modelopt.recipe.config.RecipeType`
   Enum of supported recipe types.  Currently only ``PTQ``.


Future directions
=================

The recipe system is designed to grow:

* **QAT recipes** -- ``recipe_type: qat`` with training hyperparameters, distillation
  settings, and dataset configuration.
* **Sparsity recipes** -- structured and unstructured pruning configurations.
* **Speculative decoding recipes** -- draft model and vocabulary calibration settings.
* **Composite recipes** -- chaining multiple optimization stages
  (e.g., quantize then prune) in a single recipe.
* **Dataset configuration** -- standardized ``dataset`` section for calibration data
  specification.
* **Recipe merging and override utilities** -- programmatic tools to compose and
  customize recipes.
* **Unified entry point** -- a ``nv-modelopt`` CLI that accepts ``--recipe`` as the
  primary configuration mechanism, replacing per-example scripts.
