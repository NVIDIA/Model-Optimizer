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
   * - ``quantizer_path``
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
     - ``True`` or ``False``. Shorthand for enabling or disabling matched quantizers. When ``enable`` is omitted, the quantizer
       is implicitly enabled.

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
        {"quantizer_path": "*", "enable": False},

        # 2. Enable and configure the target quantizers
        {"quantizer_path": "*weight_quantizer", "cfg": {"num_bits": 8, "axis": 0}},
        {"quantizer_path": "*input_quantizer", "cfg": {"num_bits": 8, "axis": None}},

        # 3. Apply standard exclusions last (BatchNorm, LM head, MoE routers, etc.)
        *mtq.config._default_disabled_quantizer_cfg,
    ]

.. note::

    The deny-all entry ``{"quantizer_path": "*", "enable": False}`` is available as
    :data:`modelopt.torch.quantization.config._base_disable_all` and is prepended to every
    built-in config. This ensures quantizers not explicitly targeted remain disabled.

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
    config["quant_cfg"].append({"quantizer_path": "*lm_head*", "enable": False})

    model = mtq.quantize(model, config, forward_loop)

Skipping Layers by Module Class
--------------------------------

Use ``parent_class`` to target quantizers only within a specific type of layer, leaving the
same quantizer path in other layer types unaffected:

.. code-block:: python

    config["quant_cfg"].append({
        "quantizer_path": "*input_quantizer",
        "parent_class": "nn.LayerNorm",
        "enable": False,
    })

Overriding Quantizer Precision for Specific Layers
---------------------------------------------------

A later entry with a matching ``quantizer_path`` replaces the configuration set by an earlier
entry. This allows per-layer precision overrides without restructuring the entire config:

.. code-block:: python

    config = copy.deepcopy(mtq.FP8_DEFAULT_CFG)

    # Quantize attention output projections in higher-precision INT8 instead of FP8
    config["quant_cfg"].append({
        "quantizer_path": "*o_proj*weight_quantizer",
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
            {"quantizer_path": "*weight_quantizer", "cfg": {"num_bits": 4, "block_sizes": {-1: 128}}},
            {"quantizer_path": "*input_quantizer", "cfg": {"num_bits": 8, "axis": None}},
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
        "quantizer_path": "*weight_quantizer",
        "cfg": [
            {"num_bits": 4, "block_sizes": {-1: 128, "type": "static"}, "enable": True},
            {"num_bits": (4, 3), "enable": True},  # FP8
        ],
    }

----------

Reference
=========

- :class:`QuantizerCfgEntry <modelopt.torch.quantization.config.QuantizerCfgEntry>`
- :class:`QuantizerAttributeConfig <modelopt.torch.quantization.config.QuantizerAttributeConfig>`
- :class:`QuantizeConfig <modelopt.torch.quantization.config.QuantizeConfig>`
- :func:`set_quantizer_by_cfg <modelopt.torch.quantization.conversion.set_quantizer_by_cfg>`
