=======
Pruning
=======

.. tip::

    Checkout `Qwen 3 NeMo Minitron Pruning & Distillation <https://github.com/NVIDIA-NeMo/NeMo/tree/main/tutorials/llm/qwen/pruning-distillation>`_ and
    `ResNet20 on CIFAR-10 Notebook <https://github.com/NVIDIA/Model-Optimizer/blob/main/examples/pruning/cifar_resnet.ipynb>`_
    for an end-to-end example of pruning.

ModelOpt provides three main pruning methods (aka ``mode``) - Minitron, FastNAS and GradNAS - via a unified API
:meth:`mtp.prune <modelopt.torch.prune.pruning.prune>`. Given a model,
these methods finds the subnet which meets the given deployment constraints (e.g. FLOPs, parameters)
from your provided base model with little to no accuracy degradation (depending on how aggressive is the pruning).
These pruning methods support pruning the convolutional and linear layers, and
attention heads of the model. More details on these pruning modes is as follows:

#.  ``mcore_minitron``: A pruning method developed by NVIDIA Research for pruning GPT, Mamba and Hybrid
    Transformer Mamba models in NVIDIA NeMo or Megatron-LM framework. It uses the activation magnitudes to prune
    the embedding hidden size, mlp ffn hidden size, transformer attention heads, GQA query groups,
    mamba heads and head dimension, and number of layers of the model.
    Checkout more details of the algorithm in the `paper <https://arxiv.org/abs/2408.11796>`_.
#.  ``fastnas``: A pruning method recommended for Computer Vision models. Given a pretrained model,
    FastNAS finds the subnet which maximizes the score function while meeting the given constraints.
#.  ``gradnas``: A light-weight pruning method recommended for language models like Hugging Face BERT and GPT-J.
    It uses the gradient information to prune the model's linear layers and attention heads to meet the given constraints.

Follow the steps described below to obtain the optimal model satisfying your
requirements using :mod:`mtp<modelopt.torch.prune>`:

#.  **Pruning**: Prune the model using our provided :meth:`mtp.prune <modelopt.torch.prune.pruning.prune>`
    API and get an optimal subnet describing the pruned network architecture.
#.  **Fine-tuning**: Fine-tune the resulting subnet to recover the accuracy.

*To find out more about the concepts behind NAS and pruning, please refer to*
:ref:`NAS concepts <nas-concepts>`.


.. _pruning_search:

Pruning and subnet search
=========================

The first step in pruning is to perform a search over potential subnet architectures for your pretrained model,
i.e., prune the network, to find the best subnet satisfying your deployment constraints.


Prerequisites
-------------

#. To perform pruning (:meth:`mtp.prune() <modelopt.torch.prune.pruning.prune>`) on a trained
   model, you need to set up data loaders, provide search ``constraints`` and a ``dummy_input`` (to measure
   your deployment constraints).
#. You can provide one search constraint for either ``flops`` or ``params`` by
   specifying an upper bound in terms of absolute number (``3e-6``) or a percentage (``"60%"``).
#. You should also specify the pruning algorithm (``mode``), you would like to use. Depending on the
   mode, you will need to provide additional ``config`` parameters like ``score_func`` (``fastnas`` mode)
   or ``loss_func`` (``gradnas`` mode), ``dataloader``, ``checkpoint``, etc. The most common score function
   is the validation accuracy of the model and is used to rank the sub-nets sampled from the search space.
   Loss function is used to run some forward and backward passes on the train dataloader to get the gradients.
#. Please see the API reference of :meth:`mtp.prune() <modelopt.torch.prune.pruning.prune>` for more details.

Below we show an example using :class:`"fastnas" <modelopt.torch.prune.fastnas.FastNASModeDescriptor>`.
For Minitron pruning, please refer to the `example snippet <https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/pruning#getting-started>`_ in the pruning readme.

Perform pruning
---------------

.. code-block:: python

    import modelopt.torch.prune as mtp
    from torchvision.models import resnet50

    # User-defined model
    model = resnet50()

    # Load pretrained weights here
    model.load_state_dict(pretrained_weights)


    # Wrap your original validation function to only take the model as input.
    # This function acts as the score function to rank models.
    def score_func(model):
        return validate(model, val_loader, ...)


    # Define a dummy input with similar shape as that of your input data
    dummy_input = torch.randn(1, 3, 224, 244)

    # Prune to a model with less than or equal to 60% of original FLOPs
    prune_constraints = {"flops": "60%"}

    # prune_res (dict) contains state_dict / stats of the pruner/searcher.
    pruned_model, prune_res = mtp.prune(
        model=model,
        mode="fastnas",
        constraints=prune_constraints,
        dummy_input=dummy_input,
        config={
            "data_loader": train_loader,  # training data is used for calibrating BN layers
            "score_func": score_func,  # validation score is used to rank the subnets
            # checkpoint to store the search state and resume or re-run the search with different constraint
            "checkpoint": "modelopt_fastnas_search_checkpoint.pth",
        },
    )

Note that during pruning we first convert the model into a search space containing different
possible network configurations and an optimal configuration is then searched for.

.. tip::

    If the runtime of the score function is longer than a few minutes, consider subsampling the
    dataset used in the score function. A PyTorch dataset can be subsampled using
    `torch.utils.data.Subset <https://pytorch.org/docs/stable/data.html#torch.utils.data.Subset>`_
    as following:

    .. code-block:: python

        subset_dataset = torch.utils.data.Subset(dataset, indices)

.. note::

    Pruning will modify the model in-place.

.. note::

    :meth:`mtp.prune() <modelopt.torch.prune.pruning.prune>` supports distributed data parallelism
    via ``DistributedDataParallel`` in PyTorch.

    Currently, the API does not support pruning pytorch Fully Sharded Data Parallel (FSDP) models
    so you would need to run pruning on a CPU and then finetune using FSDP. Note that GradNAS is
    much much faster than FastNAS (hence feasible on CPU as well) and is recommended for
    language models like BERT and GPT-J 6B.


Storing the pruned model
------------------------

To store the pruned model for future use you can use
:meth:`mto.save() <modelopt.torch.opt.conversion.save>`:

.. code-block:: python

    import modelopt.torch.opt as mto

    mto.save(pruned_model, "modelopt_pruned_model.pth")

.. note::

    Please see :ref:`saving and restoring of ModelOpt-modified models <save-restore>` to learn
    about all the available options for saving and restoring.


Customizing pruning config
--------------------------

In the above example, we have used the default mode config for ``mtp.prune()``. You can see it using
:meth:`mtp.fastnas.FastNASConfig() <modelopt.torch.prune.fastnas.FastNASConfig>`.
You can also specify custom mode configs to have a different search space. See
:meth:`mtp.prune() <modelopt.torch.prune.pruning.prune>` documentation for more information. An
example config is shown below:

.. code-block:: python

    import modelopt.torch.prune as mtp

    # config to restrict the search space to have a Conv2d out channels as multiple of 64
    ss_config = mtp.fastnas.FastNASConfig()
    ss_config["nn.Conv2d"]["*"]["channel_divisor"] = 64

    # run pruning as shown above
    mtp.prune(model, mode=[("fastnas", ss_config)], ...)


.. _fastnas_profile:

Profiling the search space and choosing constraints
---------------------------------------------------

The search space describes the candidates of potential pruned subnets. You can obtain information
about the overall statistics of the search space in :meth:`mtp.prune() <modelopt.torch.prune.pruning.prune>` API.
Following info will be printed before the pruning process is started:

.. code-block:: none

        Profiling the following subnets from the given model: ('min', 'centroid', 'max').
    --------------------------------------------------------------------------------

                                Profiling Results
    ┏━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
    ┃ Constraint   ┃ min          ┃ centroid     ┃ max          ┃ max/min ratio ┃
    ┡━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
    │ flops        │ 548.68M      │ 2.56G        │ 9.18G        │ 16.73         │
    │ params       │ 2.70M        │ 9.75M        │ 25.50M       │ 9.43          │
    └──────────────┴──────────────┴──────────────┴──────────────┴───────────────┘

                Constraints Evaluation
    ┏━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┓
    ┃              ┃              ┃ Satisfiable  ┃
    ┃ Constraint   ┃ Upper Bound  ┃ Upper Bound  ┃
    ┡━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━┩
    │ flops        │ 5.50G        │ True         │
    └──────────────┴──────────────┴──────────────┘


    Search Space Summary:
    ----------------------------------------------------------------------------------------------------
    * conv1.out_channels                                                               [32, 64]
      conv1.in_channels                                                                [3]
      bn1.num_features                                                                 [32, 64]
    * layer1.0.conv1.out_channels                                                      [32, 64]
      layer1.0.conv1.in_channels                                                       [32, 64]
      layer1.0.bn1.num_features                                                        [32, 64]
    * layer1.0.conv2.out_channels                                                      [32, 64]
      ...
      ...
      ...
    * layer4.2.conv1.out_channels                                                      [32, 64, 96, 128, ..., 416, 448, 480, 512]
      layer4.2.conv1.in_channels                                                       [2048]
      layer4.2.bn1.num_features                                                        [32, 64, 96, 128, ..., 416, 448, 480, 512]
    * layer4.2.conv2.out_channels                                                      [32, 64, 96, 128, ..., 416, 448, 480, 512]
      layer4.2.conv2.in_channels                                                       [32, 64, 96, 128, ..., 416, 448, 480, 512]
      layer4.2.bn2.num_features                                                        [32, 64, 96, 128, ..., 416, 448, 480, 512]
      layer4.2.conv3.out_channels                                                      [2048]
      layer4.2.conv3.in_channels                                                       [32, 64, 96, 128, ..., 416, 448, 480, 512]
    ----------------------------------------------------------------------------------------------------
    Number of configurable hparams: 36
    Total size of the search space: 2.48e+28
    Note: all constraints can be satisfied within the search space!

The profiling results will help you understand the search space and come up with a potential search
constraint that you can iterate on.

.. tip::

    Generally a search space with max/min ratio above 3 is a good search space with many layers prunable.
    The higher the max/min ratio, the more prunable the model is (potentially making FastNAS slower but better).

    A good starting point for your search constraints is the centroid of the search space. If you are using flops/params
    constraints, we highly recommend you first obtain a pruned model and measure its latency on your target deployment
    before you finetune the pruned model. Depending on the latency, you can adjust the constraints accordingly.
    Once you have a model that is within your latency constraints, you can fine-tune it to recover the accuracy.
    If you are unable to recover the accuracy (perhaps because of too aggressive pruning), you can try increasing
    the constraints and repeat the process.

.. note::

    If the constraint cannot be satisfied within the search space, the pruning will be interrupted
    and an error will be raised.

.. _pruning_fine_tuning:


Fine-tuning
===========

The final step of architecture search is to fine-tune the pruned model on your dataset. This way
you can ensure to obtain the best possible performance for your pruned model.

Prerequisites
-------------

#. To perform fine-tuning you need a pruned subnet as explained in the previous section.

#. You can reuse your existing training pipeline. We recommend running fine-tuning with your
   original training schedule:

   * 1x training epochs (or 1x downstream task fine-tuning),
   * same or smaller (0.5x-1x) learning rate.

Load the pruned model
---------------------

You can simply restore your pruned model (weights and architecture) using
:meth:`mto.restore() <modelopt.torch.opt.conversion.restore>`:

.. code-block:: python

    import modelopt.torch.opt as mto
    from torchvision.models import resnet50

    # Build original model
    model = resnet50()

    # Restore the pruned architecture and weights
    pruned_model = mto.restore(model, "modelopt_pruned_model.pth")

Run fine-tuning
---------------

Now, please go ahead and fine-tune the pruned subnet using your standard training pipeline with
the pre-configured hyperparameters. A usually good fine-tuning schedule is
to repeat the pre-training schedule with 0.5x-1x initial learning rate.

Do not forget to save the model using :meth:`mto.save() <modelopt.torch.opt.conversion.save>`.

.. code-block:: python

    train(pruned_model)

    mto.save(pruned_model, "modelopt_pruned_finetuned_model.pth")


Deploy
------
The pruned and finetuned model is now ready for downstream tasks like deployment. The model you
have in hand now should be the best neural network meeting your deployment-aware search constraint.

.. code-block:: python

    import modelopt.torch.opt as mto
    from torchvision.models import resnet50

    # Build original model
    model = resnet50()

    model = mto.restore(model, "modelopt_pruned_finetuned_model.pth")

    # Continue with downstream tasks like deployment (e.g. TensorRT or TensorRT-LLM)
    ...


.. _pruning-concepts:

Pruning Concepts
================

Pruning is the process of removing redundant components from a neural network for a given task.
Conceptually, pruning is similar to NAS, but has less computational overhead compared to NAS at the
cost of potentially finding a less optimal architecture compared to NAS. Most APIs are based on the
corresponding NAS APIs but are adapted to reflect the simpler workflow.

Specifically, for pruning we do not specifically train the search space and all its subnets.
Instead, a pre-trained checkpoint is used to approximate the search space. Therefore, we can skip
the (potentially expensive) search space training step and directly
:ref:`search <search-space-search-selection>` for a subnet architecture before fine-tuning the
resulting subnet.

.. note::

    If you want to learn more about the concept behind NAS and pruning, take a look at
    :ref:`nas-concepts` including a more detailed comparison between NAS and pruning.
