============
Sparsity
============

Introduction
============

ModelOpt's Sparsity module (:mod:`modelopt.torch.sparsity <modelopt.torch.sparsity>`) enables
you to sparsify the weights of your model. This can be useful for reducing the memory footprint of
your model and can also be used to speed up inference.


Follow the steps described below to obtain a model with sparse weights using ModelOpt's Sparsity
module :mod:`modelopt.torch.sparsity`:

#.  **Training**:You can either train your model using the existing training pipeline or load a
    pre-trained checkpoint for your model.
#.  **Sparsification**: Sparsify the model using the provided
    :meth:`mts.sparsify <modelopt.torch.sparsity.sparsification.sparsify>` API.
#.  **Checkpoint and re-load**: Save the model via :meth:`mto.save <modelopt.torch.opt.conversion.save>`
    and restore via :meth:`mto.restore <modelopt.torch.opt.conversion.restore>`. See
    :ref:`saving and loading of ModelOpt-modified model <save-restore>` to learn more.

*To find out more about Sparsity and related concepts, please refer to the section on*
:ref:`Sparsity Concepts <sparsity-concepts>`.

.. _sparsity-pts:

Post-Training Sparsification
============================

Post-training sparsification is the process of converting a dense model to a sparse model without
retraining. The simplest way to sparsify a model is to use
the :meth:`mts.sparsify <modelopt.torch.sparsity.sparsification.sparsify>` API.

The :meth:`mts.sparsify <modelopt.torch.sparsity.sparsification.sparsify>` API takes a sparsity
config and a sparsity format as input and returns a sparse model. The sparsity config is a
dictionary specifying the layers to sparsify and the optional dataloader for
calibration in data-driven sparsity, e.g., SparseGPT.

:meth:`mts.sparsify` supports `NVIDIA ASP <1_>`_ and `SparseGPT <2_>`_ methods for magnitude-based
and data-driven sparsity, respectively.

Example usage:

.. code-block:: python

    import torch
    from transformers import AutoModelForCausalLM
    import modelopt.torch.sparsity as mts

    # User-defined model
    model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6b")

    # Configure and convert for sparsity
    sparsity_config = {
        # data_loader is required for sparsity calibration
        "data_loader": calib_dataloader,
        "collect_func": lambda x: x,
    }
    sparse_model = mts.sparsify(
        model,
        "sparsegpt",  # or "sparse_magnitude"
        config=sparsity_config,
    )

.. note::
    `data_loader` is only required in case of data-driven sparsity, e.g., for calibration in
    ``sparsegpt``. `sparse_magnitude` does not require `data_loader` as it uses magnitude-based
    method for thresholding.


Save and restore the sparse model
---------------------------------

To store the sparse model for future usage, call
:meth:`mto.save() <modelopt.torch.opt.conversion.save>`:

.. code-block:: python

    mto.save(sparse_model, "modelopt_sparse_model.pth")

.. note::
    :meth:`mto.save() <modelopt.torch.opt.conversion.save>` will save the model state_dict,
    along with the sparse masks and metadata to correctly re-create the sparse model later.

To restore the saved sparse model you can use
:meth:`mto.restore() <modelopt.torch.opt.conversion.restore>`:

.. code-block:: python

    import modelopt.torch.opt as mto

    # Re-initialize the original, unmodified model
    model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6b")

    # Restore the sparse model and metadata.
    sparse_model = mto.restore(model, "modelopt_sparse_model.pth")

.. note::
    :meth:`mto.restore() <modelopt.torch.opt.conversion.restore>` will restore the model state_dict,
    along with the sparse masks and metadata of each sparse module. The plain pytorch module will be
    converted to a sparse module. The sparsity mask will be automatically enforced when the model
    weight is accessed.

.. note::
    :meth:`mts.export() <modelopt.torch.sparsity.sparsification.export>` will export the sparse
    model to a plain pytorch model. The sparse masks will be applied to model weights and all the
    sparse metadata will be removed. After exporting, sparsity will no longer be enforced during
    subsequent fine-tuning. If you want to continue fine-tuning, do not export the model.

.. note::

    Please see :ref:`saving and restoring of ModelOpt-modified models <save-restore>` to learn
    about all the available options for saving and restoring.

.. _sparsity-concepts:

Sparsity Concepts
=================

Below, we will provide an overview of ModelOpt's sparsity feature as well as its basic
concepts and terminology.


Structured and Unstructured Sparsity
------------------------------------

Weight sparsity is a model optimization technique where a fraction of the weights in a model are set
to zero. Model sparsity can be broadly categorized as structured and unstructured sparsity.
Unstructured sparsity refers to the case where the zero weights are randomly distributed across the
weight matrix. Unstructured sparsity is more flexible but can lead to poor utilization on
highly-parallelized hardware architectures like GPUs. Structured sparsity, on the other hand, is
more efficient in terms of memory access and can be exploited to achieve higher math throughput.
Structured sparsity can usually be achieved by enforcing a specific sparsity pattern on the weights.


N:M Sparsity
------------
N:M sparsity refers to special type of fine-grained structured pattern, where in each block of M
contiguous elements, at most N are nonzeros. Due to its regularity N:M sparsity can be efficiently
implemented on GPU architecture and provides the following benefits:

  * **Reduced memory bandwidth requirement:** N:M Sparsity pattern have a smaller memory bandwidth
    requirement than both dense weights and weights with unstructured sparsity pattern.

  * **Higher math throughput:** Sparse Tensor Cores deliver higher math throughput for
    matrix-multiply operations when the first argument is a compressed N:M sparse matrix.
    For example, 2:4 sparsity pattern allows for 2x higher math throughput on sparse Tensor Cores.

On current Nvidia architectures (Ampere or later), `2:4 Sparsity <3_>`_, where in each block of four
contiguous elements two are nonzeros, is supported for accelerated inference on sparse Tensor Cores.

Sparsification algorithm
------------------------

There are many ways to achieve weight sparsity. A commonly-used approach is magnitude-based sparsity
where in block of M elements, the N largest elements are retained and the rest are set to
zero. Magnitude-based sparsity is simple and easy to implement, but may not retain the accuracy of
the original model as well. Other methods such as data-driven sparsity, e.g., Optimal Brain Surgeon,
usually delivers better accuracy. ModelOpt supports both  magnitude-based (`NVIDIA ASP <1_>`_) and
data-driven sparsity (`SparseGPT <2_>`_).

.. _1: https://github.com/NVIDIA/apex/tree/master/apex/contrib/sparsity
.. _2: https://arxiv.org/abs/2301.00774
.. _3: https://arxiv.org/abs/2104.08378
