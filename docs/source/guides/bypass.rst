.. _iq-bypass:

IQ Bypass
=========

Bypass repairs native IQ scale fields in an existing ModelOpt PTQ student while preserving its
codes and frozen teacher. It streams teacher inputs at disjoint MLP/MoE boundaries and returns
the supplied, finalized student.

.. code-block:: python

    import modelopt.torch.distill as mtd
    import modelopt.torch.opt as mto

    mtd.bypass(student, teacher, train_batches)
    mto.save(student, "student.pt")

The defaults are 512 updates, Lion, global batch 8, microbatch 1, sequence length 32,768,
``max_lr=0.001``, ``warmup_fraction=0.05``, and cosine decay. Valid-token MSE is normalized over
the entire global batch. Warmup is computed from the training fraction for any update budget.
The Transformers integration requires stable 5.23.x; ordinary Torch Bypass remains optional-HF.

Select logical decoder positions with ``layer_indices`` or exact module paths with ``modules``.
The selectors intersect. Mixed dense/MoE models and per-projection IQ formats are supported;
shared experts remain inside their MoE owner. Frozen FP8/NVFP4 projections retain input gradients
without receiving optimizer state. Caller-owned native meshes determine actual placement and
unique data lanes. The integration reports native CP versus replicated full-context fallback and
refuses unsupported pipeline transport.

See the runnable `Bypass tutorial <https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/bypass>`_
for Cascade preparation, ordinary ModelOpt persistence, held-out full-vocabulary diagnostics,
trimmed original-weight prefixes, native placement, exact resume, and model/codec extension.
