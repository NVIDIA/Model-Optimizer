.. _eagle-config-reference:

Configuration Reference
===============================

EAGLE3 is configured through a dict passed to :meth:`mtsp.convert()
<modelopt.torch.speculative.speculative_decoding.convert>`. The top-level keys correspond to
fields of :class:`EagleConfig <modelopt.torch.speculative.config.EagleConfig>`, with
``eagle_architecture_config`` containing a nested dict of draft module architecture settings.

.. code-block:: python

    config = {
        # --- EagleConfig top-level fields ---
        "eagle_decoder_type": "llama",
        "eagle_freeze_base_model": True,
        "eagle_self_logit_distillation": True,
        "eagle_offline": False,
        "eagle_loss_decay_factor": 0.9,

        # --- Draft module architecture ---
        "eagle_architecture_config": {
            "num_hidden_layers": 1,
            "intermediate_size": 8192,
            ...
        },
    }
    mtsp.convert(model, [("eagle", config)])


EagleConfig fields
------------------

``eagle_decoder_type`` (*str*, default: ``"llama"``)
    Draft decoder architecture. Use ``"llama"`` for most models; ``"kimik2"`` for Kimi-K2 models.

``eagle_freeze_base_model`` (*bool*, default: ``True``)
    Keep the base model weights frozen during training. Disabling this allows joint fine-tuning
    but significantly increases memory usage.

``eagle_self_logit_distillation`` (*bool*, default: ``True``)
    Apply logit-level distillation loss in addition to hidden-state regression. Improves token
    acceptance rates without extra inference cost.

``eagle_offline`` (*bool*, default: ``False``)
    Use pre-computed hidden states from disk instead of running the base model forward pass at
    each training step. Required for large models (70B+) that cannot be co-located with the
    draft module in GPU memory. See :ref:`Offline Training <speculative_decoding_workflow:Offline Training>`.

``eagle_loss_decay_factor`` (*float*, default: ``0.9``)
    Exponential decay applied to losses at successive draft steps, weighting earlier steps more
    heavily during training.

``eagle_architecture_config`` (*dict*, default: ``{}``)
    Overrides for the draft module architecture. See `eagle_architecture_config fields`_ below.
    ``hidden_size``, ``vocab_size``, and ``max_position_embeddings`` are inferred from the base
    model and should not be set here.


eagle_architecture_config fields
---------------------------------

These keys override the default draft module architecture. Only set the fields you need to
change; unspecified fields fall back to the defaults listed below (for ``eagle_decoder_type="llama"``).

``num_hidden_layers`` (*int*, default: ``1``)
    Number of transformer layers in the draft decoder. Increasing this improves acceptance rates
    at the cost of higher draft latency.

``intermediate_size`` (*int*, default: inferred from base model)
    Feed-forward intermediate dimension of the draft decoder MLP.

``num_attention_heads`` (*int*, default: ``32``)
    Number of attention heads in the draft decoder.

``num_key_value_heads`` (*int*, default: ``8``)
    Number of key/value heads (grouped-query attention). Set equal to ``num_attention_heads``
    to disable GQA.

``hidden_act`` (*str*, default: ``"silu"``)
    Activation function used in the MLP layers.

``use_aux_hidden_state`` (*bool*, default: ``False``)
    Feed auxiliary hidden states from intermediate base model layers into the draft decoder—the
    key EAGLE3 feature. Set to ``True`` for EAGLE3; ``False`` gives EAGLE1 behaviour.

``eagle_aux_hidden_state_layer_ids`` (*list[int]*, default: ``[]``)
    Indices of base model layers whose hidden states are used as auxiliary inputs. Populated
    automatically when ``use_aux_hidden_state=True``; override only for custom layer selection.

``use_last_layernorm`` (*bool*, default: ``False``)
    Apply a layer-norm after the last draft decoder layer. Required when
    ``use_aux_hidden_state=True`` (i.e., EAGLE3 mode).

``parallel_draft_step`` (*int*, default: ``1``)
    Number of tokens drafted in parallel per step. Values greater than 1 enable parallel
    speculative decoding and can further reduce latency on suitable hardware.

.. note::

    The complete set of architecture fields and their defaults can be found in
    :mod:`modelopt.torch.speculative.eagle.default_config`.
