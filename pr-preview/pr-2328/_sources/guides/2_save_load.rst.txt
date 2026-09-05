================================
Saving & Restoring
================================

.. _save-restore:

ModelOpt optimization methods such as quantization, pruning<sup>1</sup>, distillation<sup>1</sup>, sparsity etc. modifies the original model
architecture and weights.
Hence it is important to save and restore the model architecture followed by the new weights to use the modified
model for downstream tasks.

.. note::

    <sup>1</sup> Some pruning and distillation methods don't need special handling for saving and restoring model optimizer states and can be saved and restored using their standard APIs.

This guide describes various options for saving and restoring the ModelOpt-modified models correctly.

Saving ModelOpt Models
=======================

The modifications applied to the model are captured in the model's ModelOpt state as given by
:meth:`modelopt_state <modelopt.torch.opt.conversion.modelopt_state>`. ModelOpt supports saving the architecture modifications
together with model weights or separately.

.. _save-full:

Saving ModelOpt state & model weights together
----------------------------------------------

:meth:`mto.save <modelopt.torch.opt.conversion.save>` saves the ModelOpt state together with the
new model weights (i.e, the Pytorch `state_dict <https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.state_dict>`_)
which can be used later to restore the model correctly.

Here is an example of how to save an ModelOpt-modified model:

.. code-block:: python

    import modelopt.torch.opt as mto
    import modelopt.torch.quantization as mtq

    # Initialize the original model and set the original weights
    model = ...

    # Apply ModelOpt modifications to the model. For example, quantization
    model = mtq.quantize(model, config, forward_loop)

    # Save the model weights and the `modelopt_state` to 'modelopt_model.pth'
    mto.save(model, "modelopt_model.pth")

.. _save-just-states:

Saving ModelOpt state and model weights separately
---------------------------------------------------

If you want to save the model weights with your own custom method instead of saving it with ModelOpt :meth:`mto.save`,
you should save the ``modelopt_state`` separately. This way you can correctly restore the model architecture later from
the saved ``modelopt_state`` as shown in :ref:`restoring ModelOpt state and model weights separately <load-just-states>`.

Here is an example of how to save the ``modelopt_state`` separately:

.. code-block:: python

    import torch
    import modelopt.torch.opt as mto
    import modelopt.torch.quantization as mtq

    # Initialize the original model and set the original weights
    model = ...

    # Apply ModelOpt modifications to the model. For example, quantization
    model = mtq.quantize(model, config, forward_loop)

    # Save the `modelopt_state` to 'modelopt_state.pth'
    torch.save(mto.modelopt_state(model), "modelopt_state.pth")

    # Save the model weights separately with your method
    custom_method_to_save_model_weights(model)

.. note::

    Although the above examples shows ModelOpt modifications using quantization,
    the same workflow applies to other ModelOpt modifications such as pruning, sparsity, distillation, etc.
    and combinations thereof.

Restoring ModelOpt Models
==========================


Restoring ModelOpt state & model weights together
-------------------------------------------------

:meth:`mto.restore <modelopt.torch.opt.conversion.restore>` restores the model's :meth:`modelopt_state` and the model weights
that were saved using :meth:`mto.save` as shown in :ref:`saving a ModelOpt-modified model <save-full>`.
The restored model can be used for inference or further training and optimization.

Here is an example of restoring a ModelOpt-modified model:

.. code-block:: python

    import modelopt.torch.opt as mto

    # Initialize the original model
    model = ...

    # Restore the model architecture and weights after applying ModelOpt modifications
    mto.restore(model, "modelopt_model.pth")

    # Use the restored model for inference or further training / optimization

.. _load-just-states:

Restoring ModelOpt state and model weights separately
-----------------------------------------------------

Alternatively, if you saved the ``modelopt_state`` separately as shown in
:ref:`saving modelopt_state separately <save-just-states>`,
you can restore the ModelOpt-modified model architecture using the saved ``modelopt_state``. The model weights after
the ModelOpt modifications should be loaded separately after this step.

Here is the example workflow of restoring the ModelOpt-modified model architecture using the saved
``modelopt_state``:

.. code-block:: python

    import torch
    import modelopt.torch.opt as mto

    # Initialize the original model
    model = ...

    # Restore the model architecture using the saved `modelopt_state`
    model = mto.restore_from_modelopt_state(model, modelopt_state_path="modelopt_state.pth")

    # Load the model weights separately after restoring the model architecture
    custom_method_to_load_model_weights(model)

ModelOpt Save/Restore Using Huggingface Checkpointing APIs
==========================================================

ModelOpt supports automatic save and restore of the modified models when using the
`save_pretrained <https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.save_pretrained>`_
and `from_pretrained <https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.from_pretrained>`_
APIs from Huggingface libraries such as  `transformers <https://huggingface.co/docs/transformers/main/en/index>`_ and
`diffusers <https://huggingface.co/docs/diffusers/index>`_.

To enable this feature, you need to call
:meth:`mto.enable_huggingface_checkpointing() <modelopt.torch.opt.plugins.huggingface.enable_huggingface_checkpointing>`
once in the program before loading/saving any HuggingFace models.

Here is an example of how to enable ModelOpt save/restore with the Huggingface APIs:

.. code-block:: python

    import modelopt.torch.opt as mto
    from transformers import AutoModelForCausalLM

    ...

    # Enable automatic ModelOpt save/restore with
    # Huggingface checkpointing APIs `save_pretrained` and `from_pretrained`
    mto.enable_huggingface_checkpointing()

    # Load the original Huggingface model
    model = AutoModelForCausalLM.from_pretrained(model_path)

    # Apply ModelOpt modifications to the model. For example, quantization
    model = mtq.quantize(model, config, forward_loop)

    # Save the ModelOpt-modified model architecture and weights using Huggingface APIs
    model.save_pretrained(f"ModelOpt_{model_path}")

By default, the modelopt state is saved in the same directory as the model weights.
You can disable this by setting the ``save_modelopt_state`` to ``False`` in the ``save_pretrained`` API, as shown below:

.. code-block:: python

    model.save_pretrained(f"ModelOpt_{model_path}", save_modelopt_state=False)

The model saved as above can be restored using the Huggingface ``from_pretrained`` API.
Do not forget to call :meth:`mto.enable_huggingface_checkpointing() <modelopt.torch.opt.plugins.huggingface.enable_huggingface_checkpointing>`
before loading the model. This needs to be done only once in the program.

See the example below:

.. code-block:: python

    import modelopt.torch.opt as mto
    from transformers import AutoModelForCausalLM

    ...

    # Enable automatic ModelOpt save/restore with huggingface checkpointing APIs
    # This needs to be done only once in the program
    mto.enable_huggingface_checkpointing()

    # Load the ModelOpt-modified model architecture and weights using Huggingface APIs
    model = AutoModelForCausalLM.from_pretrained(f"ModelOpt_{model_path}")
