============
Distillation
============

Introduction
============

ModelOpt's Distillation API (:mod:`modelopt.torch.distill <modelopt.torch.distill>`) allows you to enable a
knowledge-distillation training pipeline with minimal script modification.

Follow the steps described below to obtain a model trained with direct knowledge transferred from
a more powerful teacher model using :mod:`modelopt.torch.distill <modelopt.torch.distill>`:

#.  **Convert your model via** :meth:`mtd.convert <modelopt.torch.distill.distillation.convert>`:
    Wrap both a teacher and student model into a larger meta-model which abstracts away the
    interaction between the two.
#.  **Distillation training**: Seamlessly use the meta-model in place of the original model and run
    the original script with only one additional line of code for loss calculation.
#.  **Checkpoint and re-load**: Save the model via :meth:`mto.save <modelopt.torch.opt.conversion.save>`
    Note that restoring the model (via :meth:`mto.restore <modelopt.torch.opt.conversion.restore>`)
    will not reinstantiate the distillation meta-model, in order to avoid unpickling issues.

*To find out more about Distillation and related concepts, please refer to the below section*
:ref:`Distillation Concepts <distillation-concepts>`.

.. _distillation-conversion:

Convert and integrate
=====================

You can convert your model into a :class:`DistillationModel <modelopt.torch.distill.distillation_model.DistillationModel>`
using :meth:`mtd.convert() <modelopt.torch.distill.distillation.convert>`.


Example usage:

.. code-block:: python

    import modelopt.torch.distill as mtd
    from torchvision.models import resnet50

    # User-defined model (student)
    model = resnet50()

    # Configure and convert for distillation
    distillation_config = {
        # `teacher_model` is a model, model class, callable, or a tuple.
        # If a tuple, it must be of the form (model_cls_or_callable,) or
        # (model_cls_or_callable, args) or (model_cls_or_callable, args, kwargs).
        "teacher_model": teacher_model,
        "criterion": mtd.LogitsDistillationLoss(),
        "loss_balancer": mtd.StaticLossBalancer(),
    }
    distillation_model = mtd.convert(model, mode=[("kd_loss", distillation_config)])

    # Export model in original class, with only previously-present attributes
    model_exported = mtd.export(distillation_model)

.. tip::
    When training the student on a small corpus of ground truth data, consider using :class:`MFTLoss <modelopt.torch.distill.MFTLoss>` for to perform Minifinetuning in lieu of the standard
    :class:`LogitsDistillationLoss <modelopt.torch.distill.losses.LogitsDistillationLoss>`. This will allow the student to learn from the teacher's distribution while adapting to the new data, improving the specialization of the new data without overwriting teacher's general knowledge.

.. note::
    As the model is not of the same class anymore, calling ``type()`` on the model after conversion
    will not work as expected.
    Though ``isinstance()`` will still work, as the model dynamically becomes a subclass of the original's.

---

.. _distillation-concepts:

Distillation Concepts
=====================

Below, we will provide an overview of ModelOpt's distillation feature as well as its basic
concepts and terminology.

Overview
--------


..  list-table:: Glossary
    :widths: 55 90
    :header-rows: 0

    * - `Knowledge Distillation`_
      - The transfer of learnable feature information from a teacher model to a student.
    * - `Student`_
      - The model to be trained (can either start from scratch or pre-trained).
    * - `Teacher`_
      - The fixed, pre-trained model used as the example the student will "learn" from.
    * - `Distillation loss`_
      - A loss function used between the features of a student and teacher to perform Knowledge
        Distillation, separate from the student's original task loss.
    * - `Loss Balancer`_
      - An implementation for a utility which determines how to combine Distillation loss(es) and
        original student task loss into a single scalar.
    * - `Soft-label Distillation`_
      - The specific process of performing Knowledge Distillation between output logits of a teacher
        and student models.


Concepts
--------

Knowledge Distillation
^^^^^^^^^^^^^^^^^^^^^^

Distillation can be a broader term used to define any sort of information compressed among models,
but in this case we refer to basic teacher-student Knowledge Distillation. The process creates an
auxiliary loss (or can replace the original one) between a model which is already trained (teacher)
and a model which is not (student), in hopes of making the student learn information (i.e. feature
maps or logits) which the teacher has already mastered. This can serve multiple purposes:

  **A.** Model-size reduction: A smaller, efficient student model (potentially a pruned teacher) reaching
  accuracies near or exceeding that of the larger, slower teacher model. (See the
  `Lottery Ticket Hypothesis <1_>`_ for reasoning behind this, which also applies to pruning)

  **B.** An alternative to pure training: Distilling a model from an existing one (and then
  fine-tuning) can often be faster than training it from scratch.

  **C.** Module replacement: One can replace a single module within a model with a more efficient one
  and use distillation on its original outputs to effectively re-integrate it into the whole model.

  **D.** Minimal modification without catastrophic forgetting: A variant of distillation, called Minifinetuning,
  allows for training a model on even small datasets without losing the original model's knowledge.

Student
^^^^^^^

This is the model we wish to train and use in the end. It ideally meets the desired architectural
and computational requirements, but is either untrained or requires a boost in accuracy.

Teacher
^^^^^^^

This is the model from which learned features/information are used to create a loss for the student.
Usually it is larger and/or slower than desired, but possesses a satisfactory accuracy.

Distillation loss
^^^^^^^^^^^^^^^^^

To actually "transfer" knowledge from a teacher to student, we need to add (or replace) an
optimization objective to the student's original loss function(s). This can be as simple as enacting
MSE on two same-sized activation tensors between the teacher and student, with the assumption that
the features learned by the teacher are of high-quality and should be imitated as much as possible.

ModelOpt supports specifying a different loss function per layer-output pair, and includes a few
pre-defined functions for use, though users may often need to define their own.
Module-pairs-to-loss-function mappings are specified via the ``criterion`` key of the configuration
dictionary - student and teacher, respectively in order - and the loss function itself should accept
outputs in the same order as well:

.. code-block:: python

    # Example using pairwise-mapped criterion.
    # Will perform the loss on the output of ``student_model.classifier`` and ``teacher_model.layers.18``
    distillation_config = {
        "teacher_model": teacher_model,
        "criterion": {("classifier", "layers.18"): mtd.LogitsDistillationLoss()},
    }
    distillation_model = atd.convert(student_model, mode=[("kd_loss", distillation_config)])

The intermediate outputs for the losses are captured by the
:class:`DistillationModel <modelopt.torch.distill.distillation_model.DistillationModel>` and then the loss(es) are
invoked using :meth:`DistillationModel.compute_kd_loss() <modelopt.torch.distill.distillation_model.DistillationModel.compute_kd_loss>`.
If present, the original student's non-distillation loss can be passed in as an argument.

Writing a custom loss function is often necessary, especially to handle outputs that need to be processed
to obtain the logits and activations. Additional arguments to the loss function can be passed in to
:meth:`DistillationModel.compute_kd_loss() <modelopt.torch.distill.distillation_model.DistillationModel.compute_kd_loss>`
as ``kwargs``.

Loss Balancer
^^^^^^^^^^^^^

As Distillation losses may be applied to several pairs of layers, the losses are returned in the
form of a dictionary which should be reduced into a scalar value for backpropagation. A Loss
Balancer (whose interface is defined by
:class:`DistillationLossBalancer <modelopt.torch.distill.loss_balancers.DistillationLossBalancer>`) serves to fill
this purpose.

If Distillation loss is only applied to a single pair of layer outputs, and no student loss is available,
a Loss Balancer should not be provided.

ModelOpt provides a simple Balancer implementation, and the aforementioned interface can be used to create custom ones.

Soft-label Distillation
^^^^^^^^^^^^^^^^^^^^^^^

The scenario involving distillation only on the output logits of student/teacher classification
models is known as Soft-label Distillation. In this case, one could even omit the student's original
classification loss altogether if the teacher's outputs are purely preferred over whatever the
ground truth labels may be.


.. _1: https://arxiv.org/abs/1803.03635

Minifinetuning
^^^^^^^^^^^^^^

Minifinetuning is a technique that allows for training a model on even small datasets without losing the original
model's knowledge. This is achieved by algorithmic modification of the teacher's distribution depending on its
performance on the new dataset. The goal is to ensure that the separation between the correct and incorrect argmax
tokens is large enough, which can be controlled by a threshold parameter. ModelOpt provides a pre-defined loss function
for this purpose, called :class:`MFTDistillationLoss <modelopt.torch.distill.losses.MFTDistillationLoss>`, which can
be used in place of the standard :class:`LogitsDistillationLoss <modelopt.torch.distill.losses.LogitsDistillationLoss>`.
More information about the technique can be found in the original paper:
`Minifinetuning: Low-Data Generation Domain Adaptation through Corrective Self-Distillation <https://arxiv.org/abs/2506.15702>`_.
