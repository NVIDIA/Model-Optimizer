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

Decay-aware recurrent-state sparsity (experimental)
---------------------------------------------------

The :mod:`modelopt.torch.sparsity.state_sparsity` package calibrates `DASC
<https://arxiv.org/abs/2608.30386>`_ policies for persisted Gated DeltaNet (GDN) prefix state. It
derives one static decay horizon per complete GDN head from
``A_log`` and ``dt_bias``, then selects the largest caller-evaluated ``Wmax`` that passes every
configured quality, lifecycle, and physical-storage gate. ``Wmax`` may be any positive integer.
The measurements are evidence inputs produced by a caller-owned paired evaluation; this API does
not run the dense-versus-recovery suffix evaluation itself. ``perplexity_retention`` is defined as
``dense_perplexity / DASC_perplexity`` (equivalently ``exp(dense_NLL - DASC_NLL)``), so higher is
better and values above one are valid.

The initial API exports policy metadata only. It does not change model execution, quantize state,
pack ragged checkpoints, replay a suffix, or add a linear-attention kernel. A serving integration
must implement storage and recovery while preserving convolution state exactly and materializing
ordinary dense recurrent state before continuation.

.. code-block:: python

    import modelopt.torch.sparsity.state_sparsity as mtss

    config = {
        "variant": "dasc_wr",  # "dasc_nr" uses zero recovery instead
        "epsilon": 1e-3,
        "static_gate_input": -0.3,
        "wmax_candidates": [32],
        # Set this to the dtype used to store A_log and dt_bias in the checkpoint.
        "decay_parameter_storage_dtype": "bfloat16",
        "model_id": "org/model",
        "model_revision": "immutable-model-revision",
        "model_config_id": "sha256:<config-digest>",
        "calibration_data_id": "sha256:<dataset-and-protocol-digest>",
    }
    # Use these storage-canonical horizons to derive the evaluated mask and the
    # retained_heads/total_heads measurement geometry for every Wmax candidate.
    horizons = mtss.analyze_gdn_decay(
        model,
        epsilon=config["epsilon"],
        static_gate_input=config["static_gate_input"],
        decay_parameter_storage_dtype=config["decay_parameter_storage_dtype"],
    )
    measurements = [
        {
            "variant": "dasc_wr",
            "wmax": 32,
            "retained_heads": 40,
            "total_heads": 96,
            "checkpoint_savings": 0.21,
            "quality": [
                {
                    "slice_id": "validation-context-1024",
                    "perplexity_retention": 0.999,
                    "top1_agreement": 0.99,
                    "finite_continuation_logits": True,
                    "retained_state_exact": True,
                    "omitted_state_matches_recovery": True,
                    "convolution_state_exact": True,
                }
            ],
        },
    ]

    model = mtss.calibrate(model, config, measurements)
    policy = mtss.export_policy(model)

``dasc_nr`` and ``dasc_wr`` remain explicit deployment contracts: DASC-NR restores omitted heads
from zero, while DASC-WR reconstructs them from a zero-initialized suffix replay of at most the
selected ``Wmax`` tokens. Both retain whole GDN heads, preserve convolution state, and resume with
dense recurrence. KDA and serving-runtime integration are not supported by this initial API.
The initial GDN adapter accepts the ``GatedDeltaNet`` and ``Qwen3NextGatedDeltaNet`` base classes,
including ModelOpt-generated dynamic subclasses, and fails closed for unrelated implementations
even when they expose similarly named decay tensors.
Re-running :func:`~modelopt.torch.sparsity.state_sparsity.calibrate` replaces the existing DASC
mode-state entry and supersedes its stale policy without growing the checkpoint history. A policy
with recoverable GDN geometry or decay drift remains serializable, but removing or replacing the
supported GDN architecture fails closed on both save and restore. In every stale-policy case,
:func:`~modelopt.torch.sparsity.state_sparsity.export_policy` rejects it until recalibration.
Set ``decay_parameter_storage_dtype`` to the checkpoint dtype for ``A_log`` and ``dt_bias`` before
calibration. Derive the evaluated head masks and reported ``retained_heads``/``total_heads`` from
:func:`~modelopt.torch.sparsity.state_sparsity.analyze_gdn_decay` using the same ``epsilon``,
``static_gate_input``, and storage dtype passed to calibration. Policy validation allows only the
rounding introduced by the declared storage dtype and the live tensor dtype. When they differ,
distinct lossy inverse rounding bounds are composed in sequence; a duplicate dtype or an exact
widening cast contributes no additional slack. Decay tensors that are live in BF16 or FP16 are
still validated against that live dtype's rounding.

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
