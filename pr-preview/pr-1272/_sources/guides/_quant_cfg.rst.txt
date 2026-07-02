.. _quant-cfg:

======================================
Quantization Configuration (quant_cfg)
======================================

The ``quant_cfg`` field is the primary mechanism for controlling which quantizers are active in a
model and how they are configured. This guide explains the format, ordering semantics, and common
patterns for composing quantization configurations.

.. tip::

    For the list of built-in configs and supported formats, see :any:`quantization-formats`.
    For how to apply a config to a model, see :any:`_pytorch_quantization`.

----------

Overview
========

A quantization config is a Python dictionary with two top-level keys:

.. code-block:: python

    config = {
        "quant_cfg": [...],   # ordered list of QuantizerCfgEntry dicts
        "algorithm": "max",   # calibration algorithm
    }

The ``quant_cfg`` value is an **ordered list** of :class:`QuantizerCfgEntry
<modelopt.torch.quantization.config.QuantizerCfgEntry>` dicts. Each entry targets a set of
quantizer modules in the model and specifies their configuration.

----------

Entry Format
============

Each entry in the list is a dictionary with the following fields:

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Field
     - Required
     - Description
   * - ``quantizer_name``
     - Yes
     - Wildcard string matched against quantizer module names (e.g. ``"*weight_quantizer"``).
       Uses :func:`fnmatch` rules.
   * - ``parent_class``
     - No
     - Restricts matching to quantizers whose immediate parent module is of this PyTorch class
       (e.g. ``"nn.Linear"``). If omitted, all modules are targeted regardless of class.
   * - ``cfg``
     - No
     - A dict of quantizer attributes as defined by :class:`QuantizerAttributeConfig
       <modelopt.torch.quantization.config.QuantizerAttributeConfig>`, or a list of such dicts
       for sequential quantization (see :ref:`sequential-quantizers`).
   * - ``enable``
     - No
     - ``True`` or ``False``. Toggles matched quantizers on or off, independently of ``cfg``.
       When ``cfg`` is absent, **only** the enabled/disabled state is changed — all other
       attributes remain untouched. When ``cfg`` is present, ``enable`` sets the enabled state
       of the newly-configured quantizer. When ``cfg`` is present and ``enable`` is omitted,
       the quantizer is implicitly enabled (``True``).

.. note::

    Every entry must specify at least one of ``cfg`` or ``enable`` in addition to
    ``quantizer_name``.  An entry with only ``quantizer_name`` and no other keys is **invalid**
    and will raise a ``ValueError`` at config-processing time.  This prevents subtle bugs where
    a bare ``{"quantizer_name": "*"}`` would silently behave as ``enable=True`` for all
    quantizers.

----------

Default Quantizer Configuration
================================

When a quantizer is enabled but has never been touched by a ``cfg`` entry — either because no
entry in the list matched it, or because it was only reached by enable-only entries — it operates
with the default attributes of
:class:`QuantizerAttributeConfig <modelopt.torch.quantization.config.QuantizerAttributeConfig>`:

.. code-block:: python

    {
        "num_bits":                 8,       # 8-bit integer quantization
        "axis":                     None,    # per-tensor scale (no per-channel axis)
        "fake_quant":               True,    # simulate quantization in forward pass (PTQ / QAT)
        "unsigned":                 False,   # signed integer range, e.g. [-128, 127] for INT8
        "narrow_range":             False,   # full range; True would restrict to [-127, 127] for INT8
        "type":                     "static",  # static calibration (not dynamic per-inference)
        "block_sizes":              None,    # no block quantization; set for NF4 / MXFP formats
        "bias":                     None,    # no affine bias correction
        "calibrator":               "max",   # use max-abs calibration to determine amax
        "rotate":                   False,   # no Hadamard rotation (QuaRot / SpinQuant)
        "pass_through_bwd":         True,    # straight-through estimator for QAT gradients
        "trt_high_precision_dtype": "Float", # cast QDQ nodes to fp32 for TRT StronglyType export
        "backend":                  None,    # use the built-in quantization backend
        "backend_extra_args":       None,    # no extra args for custom backends
        "use_constant_amax":        False,   # calibrate amax; True hard-codes FP8 E4M3 max (448.0)
    }

In practice this means an un-configured but enabled quantizer performs **INT8 per-tensor static
fake-quantization** with a max-calibrated scale. This is rarely the intended behavior — every
quantizer you want active should be explicitly configured with a ``cfg`` entry.

----------

Ordering and Precedence
=======================

Entries are applied **in list order**. Later entries override earlier ones for any quantizer they
match. This gives a clear, composable precedence model:

- Put broad rules (e.g. deny-all) **first**.
- Put format-specific enable rules **after**.
- Put fine-grained exclusions (specific layers, classes) **last**.

The recommended pattern used by all built-in configs is:

.. code-block:: python

    "quant_cfg": [
        # 1. Deny all quantizers by default
        {"quantizer_name": "*", "enable": False},

        # 2. Enable and configure the target quantizers
        {"quantizer_name": "*weight_quantizer", "cfg": {"num_bits": 8, "axis": 0}},
        {"quantizer_name": "*input_quantizer", "cfg": {"num_bits": 8, "axis": None}},

        # 3. Apply standard exclusions last (BatchNorm, LM head, MoE routers, etc.)
        *mtq.config._default_disabled_quantizer_cfg,
    ]

.. note::

    The deny-all entry ``{"quantizer_name": "*", "enable": False}`` is available as
    :data:`modelopt.torch.quantization.config._base_disable_all` and is prepended to every
    built-in config. This ensures quantizers not explicitly targeted remain disabled.

----------

Entry Atomicity
===============

Each ``cfg``-bearing entry in ``quant_cfg`` is a **complete, self-contained configuration unit**.
When an entry with ``cfg`` matches a quantizer, it **completely replaces** that quantizer's
configuration — it does not merge with or incrementally update settings left by earlier entries.

Concretely, if an entry specifies only a subset of quantizer attributes (e.g. only ``num_bits``),
all unspecified attributes are filled in with their default values from
:class:`QuantizerAttributeConfig <modelopt.torch.quantization.config.QuantizerAttributeConfig>`.
The resulting *complete* config is then written to the quantizer, discarding whatever any prior
matching entry had set.

This means:

- **Last cfg-entry wins, fully.** If two entries both match ``*weight_quantizer`` and both carry
  a ``cfg``, the second entry does not inherit the first entry's settings — it replaces them entirely.
- **No hidden state accumulation.** The final configuration of a quantizer depends only on the
  *last* ``cfg``-bearing entry in the list that matched it, making behavior easy to reason about.
- **Changing one field requires a full spec.** Because each ``cfg`` entry is a complete replacement,
  to change only one attribute of a quantizer that was already configured, you must reproduce the
  full desired config in the new entry. Any attribute omitted from the entry will revert to its
  default, not to the value set by an earlier entry.

**Enable-only entries are the exception.** An entry with no ``cfg`` (only ``enable``) is *not* a
full replacement — it solely flips the on/off state of matched quantizers, leaving all other
attributes unchanged:

- ``{"quantizer_name": "*", "enable": False}`` disables all quantizers without touching their
  configured attributes. Use this as the first step in a deny-all-then-configure pattern.
- ``{"quantizer_name": "*weight_quantizer", "enable": True}`` (no ``cfg``) re-enables weight
  quantizers using whatever attributes they currently carry (or their defaults if they were never
  configured by a ``cfg`` entry).

For example, given the following two entries both matching ``*weight_quantizer``:

.. code-block:: python

    # Entry 1 — sets FP8 per-channel
    {"quantizer_name": "*weight_quantizer", "cfg": {"num_bits": (4, 3), "axis": 0}},

    # Entry 2 — sets INT4 blockwise (axis is NOT inherited from Entry 1)
    {"quantizer_name": "*weight_quantizer", "cfg": {"num_bits": 4, "block_sizes": {-1: 128}}},

After Entry 2 is applied, the quantizer has ``num_bits=4``, ``block_sizes={-1: 128}``, and
``axis=None`` (the default). The ``axis=0`` set by Entry 1 is gone.

.. note::

    The deny-all-then-configure pattern is safe and predictable precisely because
    ``{"quantizer_name": "*", "enable": False}`` **only** disables quantizers without resetting
    their attributes. Subsequent ``cfg`` entries then configure targets from a known default state.

----------

Common Patterns
===============

Skipping Specific Layers
------------------------

Append a disable entry after the existing config to exclude layers matched by a path pattern.
Because it is appended last, it takes precedence over all earlier entries:

.. code-block:: python

    import copy
    import modelopt.torch.quantization as mtq

    config = copy.deepcopy(mtq.FP8_DEFAULT_CFG)

    # Skip the final projection layer
    config["quant_cfg"].append({"quantizer_name": "*lm_head*", "enable": False})

    model = mtq.quantize(model, config, forward_loop)

Skipping Layers by Module Class
--------------------------------

Use ``parent_class`` to target quantizers only within a specific type of layer, leaving the
same quantizer path in other layer types unaffected:

.. code-block:: python

    config["quant_cfg"].append({
        "quantizer_name": "*input_quantizer",
        "parent_class": "nn.LayerNorm",
        "enable": False,
    })

Overriding Quantizer Precision for Specific Layers
---------------------------------------------------

A later entry with a matching ``quantizer_name`` replaces the configuration set by an earlier
entry. This allows per-layer precision overrides without restructuring the entire config:

.. code-block:: python

    config = copy.deepcopy(mtq.FP8_DEFAULT_CFG)

    # Quantize attention output projections in higher-precision INT8 instead of FP8
    config["quant_cfg"].append({
        "quantizer_name": "*o_proj*weight_quantizer",
        "cfg": {"num_bits": 8, "axis": 0},
    })

Building a Config from Scratch
-------------------------------

For entirely custom recipes, compose the list directly:

.. code-block:: python

    from modelopt.torch.quantization.config import _base_disable_all, _default_disabled_quantizer_cfg

    MY_CUSTOM_CFG = {
        "quant_cfg": [
            *_base_disable_all,
            {"quantizer_name": "*weight_quantizer", "cfg": {"num_bits": 4, "block_sizes": {-1: 128}}},
            {"quantizer_name": "*input_quantizer", "cfg": {"num_bits": 8, "axis": None}},
            *_default_disabled_quantizer_cfg,
        ],
        "algorithm": "max",
    }

    model = mtq.quantize(model, MY_CUSTOM_CFG, forward_loop)

----------

.. _sequential-quantizers:

Sequential Quantization
=======================

When ``cfg`` is a **list** of attribute dicts, the matched
:class:`TensorQuantizer <modelopt.torch.quantization.nn.modules.tensor_quantizer.TensorQuantizer>`
is replaced with a
:class:`SequentialQuantizer <modelopt.torch.quantization.nn.modules.tensor_quantizer.SequentialQuantizer>`
that applies each format in sequence. This is used, for example, in W4A8 quantization where weights
are quantized first in INT4 and then in FP8:

.. code-block:: python

    {
        "quantizer_name": "*weight_quantizer",
        "cfg": [
            {"num_bits": 4, "block_sizes": {-1: 128, "type": "static"}},
            {"num_bits": (4, 3)},  # FP8
        ],
    }

----------

.. _migrating-from-dict-format:

Migrating from Dict Format
===========================

Earlier versions of ModelOpt used a flat dictionary for ``quant_cfg``. The new list format is
preferred because it provides explicit ordering and unambiguous precedence. Existing dict-based
configs continue to work — the normalization layer converts them automatically — but new code
should use the list format.

The table below shows common patterns and their list equivalents:

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Legacy dict format
     - New list format
   * - .. code-block:: python

          "quant_cfg": {
              "*weight_quantizer": {
                  "num_bits": 8,
                  "axis": 0,
              },
              "*input_quantizer": {
                  "num_bits": 8,
                  "axis": None,
              },
              "default": {"enable": False},
          }

     - .. code-block:: python

          "quant_cfg": [
              {"quantizer_name": "*",
               "enable": False},
              {"quantizer_name": "*weight_quantizer",
               "cfg": {"num_bits": 8, "axis": 0}},
              {"quantizer_name": "*input_quantizer",
               "cfg": {"num_bits": 8, "axis": None}},
          ]

   * - .. code-block:: python

          # Disable by key assignment
          config["quant_cfg"]["*lm_head*"] = {
              "enable": False,
          }

     - .. code-block:: python

          # Append to the end (last entry wins)
          config["quant_cfg"].append(
              {"quantizer_name": "*lm_head*",
               "enable": False}
          )

   * - .. code-block:: python

          # Class-scoped entry
          "quant_cfg": {
              "nn.Linear": {
                  "*input_quantizer": {
                      "enable": False,
                  },
              },
          }

     - .. code-block:: python

          "quant_cfg": [
              {"quantizer_name": "*input_quantizer",
               "parent_class": "nn.Linear",
               "enable": False},
          ]

Key differences to keep in mind:

- The ``"default"`` key becomes ``{"quantizer_name": "*", "enable": False}`` placed at the
  **start** of the list (deny-all-then-configure pattern).
- Dict key assignment (``config["quant_cfg"]["*lm_head*"] = ...``) becomes ``list.append()``.
  Because later entries override earlier ones, appending achieves the same override effect.
- ``nn.*``-scoped dict keys become entries with a ``parent_class`` field.

----------

Reference
=========

- :class:`QuantizerCfgEntry <modelopt.torch.quantization.config.QuantizerCfgEntry>`
- :class:`QuantizerAttributeConfig <modelopt.torch.quantization.config.QuantizerAttributeConfig>`
- :class:`QuantizeConfig <modelopt.torch.quantization.config.QuantizeConfig>`
- :func:`set_quantizer_by_cfg <modelopt.torch.quantization.conversion.set_quantizer_by_cfg>`
