Module Guide
============

This page describes the public API of :mod:`modelopt.torch.speculative` (imported as ``mtsp``)
and explains how the conversion pipeline works under the hood.


Public API
----------

``import modelopt.torch.speculative as mtsp`` exposes the following:

``mtsp.convert(model, mode)``
    Main entry point. Converts a base model into a speculative decoding model.
    See `mtsp.convert`_ below.

``mtsp.MedusaConfig``, ``mtsp.EagleConfig``
    Configuration dataclasses for each algorithm.
    See the respective algorithm's configuration reference for field details.

``mtsp.EAGLE1_DEFAULT_CFG``, ``mtsp.EAGLE3_DEFAULT_CFG``, ``mtsp.EAGLE_MTP_DEFAULT_CFG``
    Built-in preset dicts for common EAGLE variants, ready to pass directly to ``mtsp.convert()``.


mtsp.convert
------------

.. code-block:: python

    mtsp.convert(model: nn.Module, mode: str | list | dict) -> nn.Module

Converts ``model`` in-place into a speculative decoding model and returns it.

**Parameters**

``model``
    A ``torch.nn.Module`` (typically loaded from HuggingFace) to be converted.

``mode``
    Specifies the algorithm and its configuration. Accepted forms:

    .. code-block:: python

        # 1. (algorithm, config_dict) tuple inside a list — most common
        mtsp.convert(model, [("eagle", {"eagle_decoder_type": "llama", ...})])

        # 2. Preset dict directly
        from modelopt.torch.speculative.config import EAGLE3_DEFAULT_CFG
        mtsp.convert(model, [("eagle", EAGLE3_DEFAULT_CFG["config"])])

        # 3. Algorithm name only — uses all defaults
        mtsp.convert(model, "medusa")

**What convert() does**

Internally, ``convert()`` delegates to the ``SpeculativeDecodingModeRegistry``, which routes
each algorithm name to its registered conversion function:

.. code-block:: text

    convert(model, [("eagle", config)])
        │
        ├─ looks up "eagle" in SpeculativeDecodingModeRegistry
        │  → EagleModeDescriptor.convert → convert_to_eagle_model()
        │
        └─ convert_to_eagle_model():
              1. Resolves model class in EagleDMRegistry
              2. Merges user config with built-in architecture defaults
              3. Wraps model as EagleModel (a DynamicModule subclass)
              4. Calls eagle_model.modify() to store all config as attributes
              5. Attaches the draft module; freezes base model if configured

The result is the same Python object as the input—``convert()`` modifies the model in-place
and also returns it. After conversion the model's ``forward()`` is replaced with a
training-compatible forward that computes speculative decoding losses.

.. note::

    ``convert()`` is designed to be called once, immediately after loading the base model and
    before moving it to GPU. The trainer then moves the converted model to the target device.


Supported algorithms
--------------------

Two algorithms are currently registered:

``"eagle"``
    EAGLE-family speculative decoding. The draft module is a lightweight autoregressive
    decoder operating at the *feature level* (predicts hidden states, not tokens directly).
    Configured via :class:`EagleConfig <modelopt.torch.speculative.config.EagleConfig>`.
    Variants—EAGLE1, EAGLE3, EAGLE-MTP—are selected through ``eagle_architecture_config``
    fields; see the :ref:`EAGLE config reference <eagle-config-reference>`.

``"medusa"``
    Medusa speculative decoding. Adds *K* independent prediction heads on top of the base
    model, each predicting a future token position in parallel.
    Configured via :class:`MedusaConfig <modelopt.torch.speculative.config.MedusaConfig>`
    with two fields: ``medusa_num_heads`` (default: ``2``) and ``medusa_num_layers``
    (default: ``1``).


Model state after conversion
-----------------------------

After ``convert()`` returns, the model object gains the following attributes and behaviours:

- The original model weights are preserved and accessible as before.
- A draft module (``eagle_module`` or medusa heads) is attached to the model.
- ``model.forward()`` is replaced with a training forward that returns speculative
  decoding losses in addition to the normal LM loss.
- If ``eagle_freeze_base_model=True`` (the default), base model parameters have
  ``requires_grad=False``. Only draft module parameters are updated during training.
- The model remains compatible with ``transformers.Trainer`` and FSDP2.


Save and restore
-----------------

ModelOpt tracks the conversion so that checkpoints can be restored to the same
speculative decoding state:

.. code-block:: python

    import modelopt.torch.opt as mto

    # Enable HuggingFace-compatible checkpointing before training
    mto.enable_huggingface_checkpointing()

    # After training: save
    trainer.save_model("<output_dir>")   # saves base + draft module together

    # Restore in a new session
    model = AutoModelForCausalLM.from_pretrained("<output_dir>")
    # model is already an EagleModel / MedusaModel — no explicit mto.restore() needed

Alternatively, if you use ``mto.save()`` / ``mto.restore()`` directly:

.. code-block:: python

    mto.save(model, "<output_dir>")
    model = AutoModelForCausalLM.from_pretrained("<base_model_id>")
    mto.restore(model, "<output_dir>")

.. note::

    ``mto.restore()`` re-runs the same ``convert_to_eagle_model()`` pipeline (with the saved
    config) and then loads the saved weights. No manual ``mtsp.convert()`` call is needed
    after restoration.
