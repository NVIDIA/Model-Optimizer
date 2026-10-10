.. _recipes:

Recipes
#######

A **recipe** is a declarative specification that fully describes how to optimize a model.
A recipe can be a single YAML file or a directory containing YAML configs and other files
that together define a model optimization workflow.
Recipes decouple optimization settings from Python code, enabling reuse, sharing, version
control, and reproducibility.  Instead of editing Python scripts to change optimization
parameters, you author (or select) a recipe and pass it to the ModelOpt tooling.
While the examples below focus on PTQ (the first supported recipe type), the recipe
system is designed to support any optimization technique.

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
* **Onboard new models** -- engineers must read source code to discover which
  settings to tweak.

Recipes solve these problems by capturing **all** the configuration needed to optimize a
model in a single, portable artifact -- either a YAML file or a directory of files.


Design overview
===============

The recipe system is part of the :mod:`modelopt.recipe` package and consists of three
layers:

1. **Recipe sources** -- YAML files or directories stored in the ``modelopt_recipes/``
   directory (shipped with the package) or on the user's filesystem.
2. **Config loader** -- :func:`~modelopt.recipe.load_config` reads YAML files, resolves
   paths, and performs automatic ``ExMy`` floating-point notation conversion.
3. **Recipe loader** -- :func:`~modelopt.recipe.load_recipe` validates the loaded
   configuration against Pydantic models and returns a typed recipe object ready for use.


Recipe format
=============

A recipe contains two top-level sections: ``metadata`` and a type-specific
configuration section (for example, ``quantize`` for PTQ recipes).  These can live
in a single YAML file or be split across files in a directory.

Single-file format
------------------

The simplest form is a single ``.yml`` or ``.yaml`` file.  Here is a PTQ example:

.. code-block:: yaml

   # modelopt_recipes/general/ptq/fp8_default-fp8_kv.yml

   metadata:
     recipe_type: ptq
     description: FP8 per-tensor weight and activation (W8A8), FP8 KV cache, max calibration.

   quantize:
     algorithm: max
     quant_cfg:
       - quantizer_name: '*'
         enable: false
       - quantizer_name: '*input_quantizer'
         cfg:
           num_bits: e4m3
           axis:
       - quantizer_name: '*weight_quantizer'
         cfg:
           num_bits: e4m3
           axis:
       - quantizer_name: '*[kv]_bmm_quantizer'
         enable: true
         cfg:
           num_bits: e4m3
       # ... standard exclusions omitted for brevity

Directory format
----------------

For larger recipes or when you want to keep metadata separate from the
optimization configuration, use a directory with multiple files.  Here is a PTQ
example:

.. code-block:: text

   my_recipe/
     recipe.yml      # metadata section
     quantize.yml    # quantize section (quant_cfg + algorithm)

``recipe.yml``:

.. code-block:: yaml

   metadata:
     recipe_type: ptq
     description: My custom NVFP4 recipe.

``quantize.yml``:

.. code-block:: yaml

   algorithm: max
   quant_cfg:
     - quantizer_name: '*'
       enable: false
     - quantizer_name: '*weight_quantizer'
       cfg:
         num_bits: e2m1
         block_sizes: {-1: 16, type: dynamic, scale_bits: e4m3}
     - quantizer_name: '*input_quantizer'
       cfg:
         num_bits: e4m3
         axis:


Metadata section
================

Every recipe must contain a ``metadata`` mapping with at least a ``recipe_type`` field:

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Field
     - Required
     - Description
   * - ``recipe_type``
     - Yes
     - The optimization category.  Determines which configuration sections are
       expected (e.g., ``"ptq"`` expects a ``quantize`` section).  See
       :class:`~modelopt.recipe.config.RecipeType` for supported values.
   * - ``description``
     - No
     - A human-readable summary of what the recipe does.


Type-specific configuration sections
=====================================

Each recipe type defines its own configuration section.  The section name and
schema depend on the ``recipe_type`` value in the metadata.

PTQ (``recipe_type: ptq``)
--------------------------

PTQ recipes contain a ``quantize`` mapping with:

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

The config loader supports a convenient shorthand for floating-point bit formats.
This is primarily used in PTQ recipes for ``num_bits`` and ``scale_bits`` fields,
but applies to any YAML value loaded through :func:`~modelopt.recipe.load_config`.
Instead of writing a Python tuple, you write the format name directly:

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

PTQ recipes
-----------

General PTQ recipes are model-agnostic and apply to any supported architecture:

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
against the built-in library first, then the filesystem.  The returned object's
type depends on the ``recipe_type`` in the metadata:

.. code-block:: python

   from modelopt.recipe import load_recipe

   # Load a built-in recipe by relative path (suffix optional)
   recipe = load_recipe("general/ptq/fp8_default-fp8_kv")

   # For PTQ recipes, the quantize dict can be passed directly to mtq.quantize()
   import modelopt.torch.quantization as mtq

   model = mtq.quantize(model, recipe.quantize, forward_loop)

.. code-block:: python

   # Load a custom recipe from the filesystem (file or directory)
   recipe = load_recipe("/path/to/my_custom_recipe.yml")
   # or: recipe = load_recipe("/path/to/my_recipe_dir/")

Command-line usage
------------------

Some example scripts accept a ``--recipe`` flag.  For instance, the PTQ example:

.. code-block:: bash

   python examples/llm_ptq/hf_ptq.py \
       --model Qwen/Qwen3-8B \
       --recipe general/ptq/fp8_default-fp8_kv \
       --export_path build/fp8 \
       --calib_size 512 \
       --export_fmt hf

When ``--recipe`` is provided, the script loads the recipe and uses its configuration
directly, bypassing format-specific flags (e.g., ``--qformat`` / ``--kv_cache_qformat``
for PTQ).


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
   (``modelopt_recipes/``), probing ``.yml`` and ``.yaml`` suffixes as well as
   directories.
3. Then check the **filesystem**, probing the same suffixes and directories.

This means built-in recipes can be referenced without any prefix:

.. code-block:: python

   # These are all equivalent:
   load_recipe("general/ptq/fp8_default-fp8_kv")
   load_recipe("general/ptq/fp8_default-fp8_kv.yml")


Writing a custom recipe
=======================

To create a custom recipe:

1. Start from an existing recipe that is close to your target configuration.
2. Copy it and modify the type-specific configuration as needed (for PTQ recipes,
   see :ref:`quant-cfg` for ``quant_cfg`` entry format details).
3. Update the ``metadata.description`` to describe your changes.
4. Save the file (or directory) and pass its path to ``load_recipe()`` or ``--recipe``.

Example -- creating a custom PTQ recipe (INT8 per-channel):

.. code-block:: yaml

   # my_int8_recipe.yml
   metadata:
     recipe_type: ptq
     description: INT8 per-channel weight, per-tensor activation.

   quantize:
     algorithm: max
     quant_cfg:
       - quantizer_name: '*'
         enable: false
       - quantizer_name: '*weight_quantizer'
         cfg:
           num_bits: 8
           axis: 0
       - quantizer_name: '*input_quantizer'
         cfg:
           num_bits: 8
           axis:
       - quantizer_name: '*lm_head*'
         enable: false
       - quantizer_name: '*output_layer*'
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
   PTQ-specific recipe.  Adds the ``quantize`` field (a dict with ``quant_cfg`` and
   ``algorithm``).

:class:`~modelopt.recipe.config.RecipeType`
   Enum of supported recipe types.


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
