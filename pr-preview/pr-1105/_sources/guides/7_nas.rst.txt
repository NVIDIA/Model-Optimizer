===
NAS
===


Introduction
============

ModelOpt provides NAS methods (aka ``mode``) like AutoNAS via the
:mod:`modelopt.torch.nas<modelopt.torch.nas>` module. Given a model, these methods finds the
subnet which meets the given deployment constraints (e.g. FLOPs, parameters) from your provided
base model with little to no accuracy degradation (depending on how aggressive is the model size reduced).
More details on this NAS mode is as follows:

#.  ``autonas``: A NAS method suitable for Computer Vision models that searches for the layerwise parameters like number of channels,
    kernel size, network depth etc.

Follow the steps described below to obtain the optimal model meeting your unique requirements
using :mod:`modelopt.torch.nas<modelopt.torch.nas>`:

#.  **Convert your model via** :meth:`mtn.convert <modelopt.torch.nas.conversion.convert>`:
    Natively generate a neural architecture search space from your PyTorch base model using a simple
    set of configurations. Conveniently save and restore the model architecture and weights during
    the process.
#.  **NAS training**: Seamlessly train the resulting search space within your existing training
    pipeline.
#.  **Subnet architecture search via** :meth:`mtn.search <modelopt.torch.nas.algorithms.search>`:
    Search for the best neural architecture (subnet) satisfying your deployment constraints, e.g.,
    FLOPs / parameters.
#.  **Fine-tuning**: Optionally, fine-tune the resulting subnet to achieve even higher accuracy.

*To find out more about NAS and related concepts, please refer to the below section*
:ref:`NAS Concepts <nas-concepts>`.


.. _nas-conversion:

Convert and save
================

You can convert your model and generate a search space from it using
:meth:`mtn.convert() <modelopt.torch.nas.conversion.convert>`.
The resulting search space should be saved using :meth:`mto.save() <modelopt.torch.opt.conversion.save>`.
It can be loaded back using :meth:`mto.restore() <modelopt.torch.opt.conversion.restore>`
to perform the subsequent steps of architecture search.

Example usage:

.. code-block:: python

    import modelopt.torch.nas as mtn
    import modelopt.torch.opt as mto
    from torchvision.models import resnet50

    # User-defined model
    model = resnet50()

    # Generate the search space for AutoNAS
    model = mtn.convert(model, mode="autonas")

    # Save the search space for future use
    mto.save(model, "modelopt_model.pth")

.. note::

    The NAS API's are a super-set of the pruning API's. You can use the pruning modes (e.g. ``"fastnas"``, ``"gradnas"``, etc.)
    here as well.

.. note::

    In the above example, we have used the default AutoNAS ``config`` for ``mtn.convert()``.
    You can see it using
    :meth:`mtn.autonas.AutoNASConfig() <modelopt.torch.nas.autonas.AutoNASConfig>`.
    You can also specify custom configurations to have a different search space. See
    :meth:`mtn.convert() <modelopt.torch.nas.conversion.convert>` documentation for more information.
    An example config is shown below:

    .. code-block:: python

        import modelopt.torch.nas as mtn

        config = mtn.autonas.AutoNASConfig()
        config["nn.Conv2d"]["*"]["out_channels_ratio"] += (0.1,)  # include more channel choices

        model = mtn.convert(model_or_model_factory, mode=[("autonas", config)])

.. note::

    If you want to learn more about the conversion process and the prerequisites for your model,
    you can take a look at :ref:`NAS Model Prerequisites <nas-prereqs>`.

.. note::

    Please see :ref:`saving and restoring of ModelOpt-modified models <save-restore>` to learn
    about all the available options for saving and restoring.


Profiling a search space
------------------------

The search space can be used to perform architecture search according to your desired deployment
constraints.

To better understand the performance and the range of the resulting search space, you can profile
the search space together with your deployment constraints using
:meth:`mtn.profile() <modelopt.torch.nas.algorithms.profile>`:

.. code-block:: python

    import torch

    # Looking for a subnet with at most 4 GFLOPs
    constraints = {"flops": 4.0e9}

    # Measure FLOPs against dummy_input
    # Can be provided as a single tensor or tuple of input args to the model.
    dummy_input = torch.randn(1, 3, 224, 224)

    is_sat, search_space_stats = mtn.profile(model, dummy_input, constraints=constraints)

Following info will be printed:

.. code-block:: none

    Profiling the following subnets from the given model: ('min', 'centroid', 'max').
    --------------------------------------------------------------------------------

                                Profiling Results
    ┏━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
    ┃ Constraint   ┃ min          ┃ centroid     ┃ max          ┃ max/min ratio ┃
    ┡━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
    │ flops        │ 975.84M      │ 3.68G        │ 9.18G        │ 9.40          │
    │ params       │ 4.84M        │ 12.33M       │ 25.50M       │ 5.27          │
    └──────────────┴──────────────┴──────────────┴──────────────┴───────────────┘

                Constraints Evaluation
    ┏━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┓
    ┃              ┃              ┃ Satisfiable  ┃
    ┃ Constraint   ┃ Upper Bound  ┃ Upper Bound  ┃
    ┡━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━┩
    │ flops        │ 4.00G        │ True         │
    └──────────────┴──────────────┴──────────────┘

    Search Space Summary:
    ----------------------------------------------------------------------------------------------------
    * conv1.out_channels                                                               [32, 64]
      conv1.in_channels                                                                [3]
      bn1.num_features                                                                 [32, 64]
    * layer1.depth                                                                     [1, 2, 3]
    * layer1.0.conv1.out_channels                                                      [32, 64]
      layer1.0.conv1.in_channels                                                       [32, 64]
      layer1.0.bn1.num_features                                                        [32, 64]
    * layer1.0.conv2.out_channels                                                      [32, 64]
      ...
      ...
      ...
    * layer4.2.conv1.out_channels                                                      [256, 352, 512]
      layer4.2.conv1.in_channels                                                       [2048]
      layer4.2.bn1.num_features                                                        [256, 352, 512]
    * layer4.2.conv2.out_channels                                                      [256, 352, 512]
      layer4.2.conv2.in_channels                                                       [256, 352, 512]
      layer4.2.bn2.num_features                                                        [256, 352, 512]
      layer4.2.conv3.out_channels                                                      [2048]
      layer4.2.conv3.in_channels                                                       [256, 352, 512]
    ----------------------------------------------------------------------------------------------------
    Number of configurable hparams: 40
    Total size of the search space: 1.90e+18
    Note: all constraints can be satisfied within the search space!

You can also skip the ``constraints`` parameter to just print the range of available constraints
without checking if it is within your constraints. The profiling results will help you understand
the search space and come up with a potential search constraint that you can iterate on.


NAS training
============

Prerequisites
-------------

During NAS training, you can use your existing training infrastructure. However, we recommend
you make the following modifications to your training hyperparameters:

For AutoNAS:

#.  Increase the training time (epochs) by 2-3x.

#.  Make sure that the learning rate schedule is adjusted for the longer training time.

#.  We recommend using a continuously decaying learning
    rate schedule such as the cosine annealing schedule (see
    `PyTorch documentation <https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html#torch.optim.lr_scheduler.CosineAnnealingLR>`_).

Restore the search space
------------------------

Please restore the search space from the saved one to continue with the rest of the steps as shown
below:

.. code-block:: python

    # Provide the model before conversion to mto.restore
    model = mto.restore(model_or_model_factory, "modelopt_model.pth")

Training
--------

You can now proceed with your existing training pipeline with the changes in training time and learning rate.


Subnet architecture search
==========================

The next step in NAS is to perform architecture search on the resulting search space to find the
best subnet satisfying your deployment constraints.


Prerequisites
-------------

#.  To perform the search (:meth:`mtn.search() <modelopt.torch.nas.algorithms.search>`) on a trained
    model, a score function, a dummy input (to measure your deployment constraints), the training
    dataloader (to calibrate the normalization layers) and the constraints are required. Please see
    the :meth:`mtn.search() <modelopt.torch.nas.algorithms.search>` API for more details.

#.  Depending on the algorithm, you may be able to provide multiple search constraint such as
    ``flops`` or ``params`` by specifying an upper bound for each.

Performing search
-----------------

Below is an example of running search on an AutoNAS converted and trained model.

.. code-block:: python

    # Wrap your original validation function to only take the model as input.
    # This function acts as the score function to rank models.
    def score_func(model):
        return validate(model, val_loader, ...)


    # Specify the sample input including target data shape for FLOPs calculation.
    dummy_input = torch.randn(1, 3, 224, 224)

    # Looking for a subnet with at most 4 GFLOPs
    search_constraints = {"flops": 4.0e9}

    # search_res (dict) contains state_dict / stats of the searcher
    searched_model, search_res = mtn.search(
        model=model,
        constraints=search_constraints,
        dummy_input=dummy_input,
        config={
            "data_loader": train_loader,  # training data is used for calibrating BN layers
            "score_func": score_func,  # validation score is used to rank the subnets
            # checkpoint to store the search state and resume or re-run the search with different constraint
            "checkpoint": "modelopt_search_checkpoint.pth",
        },
    )

    # Save the searched model for further fine-tuning
    mto.save(searched_model, "modelopt_searched_model.pth")

.. tip::

    If the runtime of the score function is longer than a few minutes, consider subsampling the
    dataset used in the score function. A PyTorch dataset can be subsampled using
    `torch.utils.data.Subset <https://pytorch.org/docs/stable/data.html#torch.utils.data.Subset>`_
    as following:

    .. code-block:: python

        subset_dataset = torch.utils.data.Subset(dataset, indices)

.. note::

    NAS will modify the model in-place.

.. note::

    :meth:`mtn.search() <modelopt.torch.nas.algorithms.search>` supports distributed data parallelism
    via ``DistributedDataParallel`` in PyTorch.


Fine-tuning
===========

After search, the accuracy drop may be less significant compared with pruning, however, we still
recommend to run fine-tuning to recover the best accuracy. A usually good fine-tuning schedule
for AutoNAS is to repeat the pre-training schedule (1x epochs) with 0.5x-1x initial learning rate as done in FastNAS.
Please refer to the :ref:`Pruning fine-tuning section <pruning_fine_tuning>` for more details.


.. _nas-prereqs:

NAS Model Prerequisites
=======================

In this guide, we will go through the steps to set up your model to work with NAS and pruning. At
the end of this guide, you will be able to :meth:`convert<modelopt.torch.nas.conversion.convert>`
your own model to generate a search space that can be used for NAS and pruning.

Convert your model
------------------

Most PyTorch models, including custom models, are natively compatible with ModelOpt (depending on how the forward is
implemented). To quickly test whether your model is compatible you can simply try to convert it:

.. code-block:: python

    import modelopt.torch.nas as mtn
    from torchvision.models import resnet50

    # User-defined model
    model = resnet50()

    # Convert the model into a search space
    model = mtn.convert(model, mode="fastnas")


If you encounter problems or would like to understand more about the conversion process, please
continue reading. Otherwise, you can skip the rest of this guide.


The conversion process
----------------------

ModelOpt will automatically generate a search space for you from your custom PyTorch model.
This is a one time process process performed during pruning and NAS. Once a model is converted, you
can save and restore it for downstream tasks like training, inference, and fine-tuning.

To help you better understand how the search space is derived from your model, we go through the
process in more detail below.

Layer support
^^^^^^^^^^^^^

You can make the most use out of ModelOpt with model architectures consisting of layers that
ModelOpt can automatically convert into searchable units.

Specifically, the following `native PyTorch layers <https://pytorch.org/docs/stable/nn.html>`_
can be converted into searchable units:

.. code-block:: python

    import torch.nn as nn

    # We convert native PyTorch convolutional layers to automatically search over the number of
    # channels and optionally over the kernel size.
    nn.Conv1d
    nn.Conv2d
    nn.Conv3d
    nn.ConvTranspose1d
    nn.ConvTranspose2d
    nn.ConvTranspose3d

    # We convert native PyTorch linear layers to automatically search over the number of features
    nn.Linear

    # We convert native PyTorch sequential layers that contain residual blocks to automatically
    # search over the number of layers (depth) in the sequential layer.
    nn.Sequential

    # We convert Megatron-core / NeMo GPT or MoE or Mamba Hybrid style models (e.g. Llama3, Nemotron-H, Qwen3-30B-A3B)
    # to automatically search over the
    # MLP hidden size, number of attention heads
    # number of mamba heads, mamba head dimension,
    # number of moe experts, moe ffn hidden size, moe shared expert intermediate size,
    # and depth of the model.
    megatron.core.models.gpt.GPTModel
    megatron.core.models.mamba.MambaModel
    nemo.collections.llm.gpt.model.base.GPTModel

    # We convert Hugging Face Attention layers to automatically search over the number of heads
    # and MLP hidden size.
    # Make sure `config.use_cache` is set to False during pruning.
    transformers.models.bert.modeling_bert.BertAttention
    transformers.models.gptj.modeling_gptj.GPTJAttention

Generating a search space
^^^^^^^^^^^^^^^^^^^^^^^^^

To generate a search space from your desired model, a simple call to
:meth:`mtn.convert()<modelopt.torch.nas.conversion.convert>` suffices:

.. code-block:: python

    import modelopt.torch.nas as mtn
    from torchvision.models import resnet50

    # User-defined model
    model = resnet50()

    # Convert the model for NAS/pruning
    model = mtn.convert(model, mode="fastnas")

Your generated ``model`` represents a search space consisting of a collection of subnets.
Note that you can use the converted model like any other, regular PyTorch model. It will behave
according to the currently activated subnet.


Roughly, the :meth:`convert<modelopt.torch.nas.conversion.convert>` process can be broken down into
the following steps:

1. Trace through the model to resolve layer dependencies and record how layers are connected.

2. Convert supported layers into searchable units, i.e., dynamic layers and connect them
   according to the recorded dependencies.

3. Generate a consistent search space from the converted model.

.. note::

    During pruning, the conversion is performed implicitly when
    :meth:`mtp.prune<modelopt.torch.prune.pruning.prune>` is called.

Prerequisites
-------------

In order to correctly generate a search space, your original model should satisfy the following
prerequisites.


Traceability
^^^^^^^^^^^^

The model needs to be traceable with ModelOpt's `torch.fx <https://pytorch.org/docs/stable/fx.html>`_-like tracer.

If not, you will see errors or warnings when you run :meth:`mtn.convert() <modelopt.torch.nas.conversion.convert>`.
Note that some of these warnings may not affect the search space and hence can be ignored.

Note that in some cases certain layers cannot be traced and, if possible, you should adjust their
definition and forward method to be traceable. Otherwise, such layers and all affected layers will
be ignored in the conversion process.

.. Onnx exportable
.. ^^^^^^^^^^^^^^^

.. The model needs to be exportable via
.. `torch.onnx <https://pytorch.org/docs/stable/onnx.html#torch.onnx.export>`_ for certain tasks.

.. You will see errors or warnings when you run
.. :meth:`mtn.profile() <modelopt.torch.nas.algorithms.profile>` and have to make appropriate changes in
.. your model if required.

DistributedDataParallel
^^^^^^^^^^^^^^^^^^^^^^^

Wrapping the model with ``DistributedDataParallel`` should occur **after** the conversion process
and during wrapping ``find_unused_parameters=True`` needs to be set:

.. code-block:: python

    model = mtn.convert(model, ...)
    model = DistributedDataParallel(model, find_unused_parameters=True)

Auxiliary modules
^^^^^^^^^^^^^^^^^

If your model contains auxiliary modules, e.g., branches that are active only
during the training, ensure that you convert the full model such that **all** modules
are active during the conversion process.


Known limitations
-----------------

Please be aware of other potential limitations as mentioned in the :ref:`NAS FAQs <nas_faqs>`!


.. _nas-concepts:

NAS Concepts
============

Below, we will provide an overview of ModelOpt's neural architecture search (NAS) and pruning
algorithms as well as its basic concepts and terminology.

Overview
--------


..  list-table:: Glossary
    :widths: 55 90
    :header-rows: 0

    * - `Neural Architecture Search (NAS)`_
      - The process of finding the best neural network architecture for a given task.
    * - `Search space`_
      - The set of possible candidate architecture that are searched during pruning or NAS.
    * - `Architecture hyperparameters`_
      - The set of hyperparameters, e.g., number of layers, describing the search space.
    * - `Subnet`_
      - A candidate architecture in the search space.
    * - `NAS-based training`_
      - The process of training the collection of subnets in the search space.
    * - `Architecture search & selection`_
      - The process of finding an optimal subnet within a trained search space.
    * - `Subnet fine-tuning`_
      - The process of training the selected subnet in isolation for improved final accuracy.
    * - :ref:`Pruning <pruning-concepts>`
      - The process of removing redundant components from a neural network for a given task.



Concepts
--------

Below, we provide an introduction to the concepts and terminology of neural architecture
search. During regular neural network training, only the neural network weights are
trained. However during NAS, both the weights and the architecture of the model are trained.


Neural Architecture Search (NAS)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Neural architecture search is the process of finding the best neural network architecture from
a set of candidate architectures. NAS is usually performed before, during, or in-between training.
During NAS different performance metrics, such as accuracy, on-device latency, or size of the model,
are used to evaluate the candidate architectures.

.. _search-space-subnets:

Search space
^^^^^^^^^^^^

The search space is defined as the (discrete) set of all possible neural architectures that are
trained. Search spaces are derived from a (user-specified) base architecture (e.g., ResNet50) and a
set of **configs** that describe how to parameterize the base architecture, see
:ref:`NAS Model Prerequisites <nas-prereqs>` for more info.

Architecture hyperparameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The search space is parameterized via a set of discrete architecture hyperparameters that describe
individual "modifications" to the base architecture, e.g., the number of channels in a convolutional
layer, the number of repeated building blocks, number of attention heads in a transformer layer, etc.
Each possible architecture in the search space can be described as a distinct configuration of the
set of architecture hyperparameters.

Subnet
^^^^^^
The search space consists of a collection of *subnets*, where each subnet represents a neural
architecture. Each **subnet** constitutes a neural architecture with different layers and operators
or different parameterization (e.g. channel number) of each layer.

To better characterize a given search space, we usually consider a few distinct subnets:

* Minimum subnet (``min``): The smallest subnet within the search space.

* Centroid subnet (``centroid``): The subnet for which each architecture hyperparameter is set to
  the value closest to its centroid (mean).

* Maximum subnet (``max``): The largest subnet within the search space.

ModelOpt-converted model
^^^^^^^^^^^^^^^^^^^^^^^^

After the conversion, the user-provided neural network will represent the search space. It can be
obtained via :meth:`mtn.convert()<modelopt.torch.nas.conversion.convert>`, see :ref:`nas-conversion`.

During the conversion process, the search space is automatically derived from a given base
architecture and the relevant architecture hyperparameters are automatically identified.

The next step is to train the converted model (instead of the original architecture) to find the
optimal subnet for your deployment constraints.

NAS-based training
^^^^^^^^^^^^^^^^^^

During training of an search space, we simultaneously train both the model's weights and
architecture:

* Using :mod:`modelopt.torch.nas<modelopt.torch.nas>` you can re-use your existing
  training loop to train the search space.

* During search space training the entire collection of subnets is automatically trained together
  with its weights.

* Given that we train both the architecture (all subnets) and the weights, training data may vary
  compared to regular training as described in the NAS Training section above.

.. _search-space-search-selection:

Architecture search & selection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

At the end of search space training process, the next step is to **search** and select the subnet
from the search space:

* The search procedure is a discrete optimization problem to determine the optimal subnet
  configuration from the search space.

* The search procedure takes your deployment constraints, e.g., FLOPs, parameters or latency and inference device, into
  account to determine the optimal (most accurate) subnet configuration while satisfying the
  constraints.

* The resulting subnet can be used for further downstream tasks, e.g., fine-tuning and deployment.


Subnet fine-tuning
^^^^^^^^^^^^^^^^^^

To further boost the accuracy of the selected subnet, the subnet is usually fine-tuned on the
original task:

* To fine-tune the subnet, you can simply repeat the training pipeline of the original model
  with the adjusted training schedule as described in the Fine-tuning section above.

* The fine-tuned model constitutes the deployable model with the optimal trade-off between
  accuracy and your provided constraints.


NAS vs. Pruning
---------------

The difference between NAS and pruning is summarized below.

.. list-table::
    :widths: 20 40 40
    :header-rows: 1
    :stub-columns: 1

    * -
      - NAS
      - Pruning
    * - Search space
      - More flexible search space with additional searchable dimensions such as network depth,
        kernel size, or selection of activation function.
      - Less flexible search space with searchable dimensions constrained to fewer options such as
        number of channels and features or attention heads.
    * - Training time
      - Usually requires training a model for additional time before a subnet can be searched.
      - No training is required when a pre-trained checkpoint is available. If not, regular
        training can be used to pre-train a checkpoint.
    * - Performance
      - Can provide improved accuracy-latency trade-off due to more flexible search space and the
        increased training time.
      - May provide similar performance to NAS in particular applications, however, usually exhibits
        worse performance due to the limited search space and training time.


[Advanced] Adding a new NAS/Prune Algorithm
===========================================

* Please refer to this `template <https://github.com/NVIDIA/Model-Optimizer/compare/template/new-nas-mode>`_
  for adding a new NAS algorithm.
* Please refer to `mcore_minitron.py <https://github.com/NVIDIA/Model-Optimizer/blob/main/modelopt/torch/prune/plugins/mcore_minitron.py>`_
  for an actual example of adding Minitron Pruning algorithm.
