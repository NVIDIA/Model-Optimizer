====================
PyTorch Quantization
====================

Key advantages offered by ModelOpt's PyTorch quantization:

#. Support advanced quantization formats, e.g., Block-wise Int4 and FP8.
#. Native support for LLM models in Hugging Face and NeMo.
#. Advanced Quantization algorithms, e.g., SmoothQuant, AWQ.
#. Deployment support to ONNX and NVIDIA TensorRT.

.. note::

    ModelOpt quantization is fake quantization, which means it only simulates the low-precision computation in PyTorch.
    Real speedup and memory saving should be achieved by exporting the model to deployment frameworks.

.. tip::

    This guide covers the usage of ModelOpt quantization. For details on the quantization formats and recommended use cases,
    please refer to :any:`quantization-formats`.

Apply Post Training Quantization (PTQ)
======================================

PTQ can be achieved with simple calibration on a small set of training or evaluation data (typically 128-512 samples) after converting a regular PyTorch model to a quantized model.
The simplest way to quantize a model using ModelOpt is to use :meth:`mtq.quantize() <modelopt.torch.quantization.model_quant.quantize>`.

:meth:`mtq.quantize` takes a model, a quantization config and a forward loop callable as input.  The quantization config specifies the layers to quantize, their quantization formats as well as the algorithm to use for calibration. Please
refer to :any:`quantization-configs` for the list of quantization configs supported by default. You may also define your own quantization config as
described in :ref:`customizing quantizer config <customize_quantizer_config>`.

ModelOpt supports algorithms such as AWQ, SmoothQuant, SVDQuant or max for calibration. Please refer to :meth:`mtq.calibrate <modelopt.torch.quantization.model_quant.calibrate>`
for more details.

The forward loop is used to pass data through the model in-order to collect statistics for calibration.
It should wrap around the calibration dataloader and the model.

Here is an example of performing PTQ using ModelOpt:

.. code-block:: python

    import modelopt.torch.quantization as mtq

    # Setup the model
    model = get_model()

    # Select quantization config
    config = mtq.INT8_SMOOTHQUANT_CFG

    # Quantization need calibration data. Setup calibration data loader
    # An example of creating a calibration data loader looks like the following:
    data_loader = get_dataloader(num_samples=calib_size)


    # Define forward_loop. Please wrap the data loader in the forward_loop
    def forward_loop(model):
        for batch in data_loader:
            model(batch)


    # Quantize the model and perform calibration (PTQ)
    model = mtq.quantize(model, config, forward_loop)

To verify that the quantizer nodes are placed correctly in the model, let's print the quantized model summary as show below:

.. code-block:: python

    # Print quantization summary after successfully quantizing the model with mtq.quantize
    # This will show the quantizers inserted in the model and their configurations
    mtq.print_quant_summary(model)


After PTQ, the model can be exported to ONNX with the normal PyTorch ONNX export flow.

.. code-block:: python

    torch.onnx.export(model, sample_input, onnx_file)

ModelOpt also supports direct export of Huggingface or Nemo LLM models to TensorRT-LLM for deployment.
Please see :doc:`TensorRT-LLM Deployment <../deployment/1_tensorrt_llm>` for more details.

.. note::
    To reduce the memory footprint of the quantized model, please refer to :doc:`Compress Quantized Models <./_compress_quantized_models>`.

Quantization-aware Training (QAT)
=================================

QAT is the technique of fine-tuning a quantized model to recover model quality degradation due to quantization.
While QAT requires much more compute resources than PTQ, it is highly effective in recovering model quality.

A model quantized using  :meth:`mtq.quantize() <modelopt.torch.quantization.model_quant.quantize>` could be directly fine-tuned with QAT.
Typically during QAT, the quantizer states are frozen and the model weights are fine-tuned.

Here is an example of performing QAT:

.. code-block:: python

    import modelopt.torch.quantization as mtq

    # Select quantization config
    config = mtq.INT8_DEFAULT_CFG


    # Define forward loop for calibration
    def forward_loop(model):
        for data in calib_set:
            model(data)


    # QAT after replacement of regular modules to quantized modules
    model = mtq.quantize(model, config, forward_loop)

    # Fine-tune with original training pipeline
    # Adjust learning rate and training duration
    train(model, train_loader, optimizer, scheduler, ...)

.. tip::

    We recommend QAT for 10% of the original training epochs. For LLMs, we find that QAT fine-tuning for even
    less than 1% of the original pre-training duration is often sufficient to recover the model quality.

Storing and restoring quantized model
======================================

The model weights and quantizer states need to saved for future use or to resume training.
Please see :ref:`saving and restoring of ModelOpt-modified models <save-restore>` to learn
how to save and restore the quantized model.


Optimal Partial Quantization using ``auto_quantize``
===================================================================

:meth:`auto_quantize <modelopt.torch.quantization.model_quant.auto_quantize>` is a PTQ algorithm from ModelOpt which
quantizes a model by searching for the best quantization format per-layer
while meeting the performance constraint specified by the user. ``auto_quantize`` enables to trade-off model accuracy
for performance. Please see :meth:`auto_quantize <modelopt.torch.quantization.model_quant.auto_quantize>` for more details
on the API usage.

Currently ``auto_quantize`` supports only ``effective_bits`` as the performance constraint (for both weight-only
quantization and weight & activation quantization). ``effective_bits`` constraint specifies the effective number of bits for the quantized model.

You may specify a ``effective_bits`` constraint such as 4.8 for mixed precision quantization using  :attr:`NVFP4_DEFAULT_CFG` & :attr:`FP8_DEFAULT_CFG`.
``AutoQuantize`` will automatically quantize highly sensitive layers in :attr:`FP8_DEFAULT_CFG` while keeping less sensitive layers in :attr:`NVFP4_DEFAULT_CFG`
(and even skip quantization for any extremely sensitive layers) so that
the the final mixed precision quantized model has an effective quantized bits of 4.8.
This model would give a better accuracy than the model quantized with vanilla :attr:`NVFP4_DEFAULT_CFG` since
the more aggressive :attr:`NVFP4_DEFAULT_CFG` quantization was not applied for the highly sensitive layers.

Here is how to perform ``auto_quantize``:

.. code::

    import modelopt.torch.quantization as mtq
    import modelopt.torch.opt as mto

    # Define the model & calibration dataloader
    model = ...
    calib_dataloader = ...

    # Define forward_step function.
    # forward_step should take the model and data as input and return the output
    def forward_step(model, data):
        output =  model(data)
        return output

    # Define loss function which takes the model output and data as input and returns the loss
    def loss_func(output, data):
        loss = ...
        return loss


    # Perform auto_quantize
    model, search_state_dict = mtq.auto_quantize(
        model,
        constraints = {"effective_bits": 4.8},
        # supported quantization formats are listed in `modelopt.torch.quantization.config.choices`
        quantization_formats = [mtq.NVFP4_DEFAULT_CFG, mtq.FP8_DEFAULT_CFG]
        data_loader = calib_dataloader,
        forward_step=forward_step,
        loss_func=loss_func,
        ...
        )

    # Save the searched model for future use
    mto.save(model, "auto_quantize_model.pt")


Advanced Topics
===============

TensorQuantizer
---------------

Under the hood, ModelOpt :meth:`mtq.quantize() <modelopt.torch.quantization.model_quant.quantize>` inserts
:class:`TensorQuantizer <modelopt.torch.quantization.nn.modules.tensor_quantizer.TensorQuantizer>`
(quantizer modules) into the model layers like linear layer, conv layer etc. and patches their forward method to perform quantization.

The quantization parameters are as described in :class:`QuantizerAttributeConfig <modelopt.torch.quantization.config.QuantizerAttributeConfig>`.
They can be set at initialization by passing :class:`QuantizerAttributeConfig <modelopt.torch.quantization.config.QuantizerAttributeConfig>`
or later by calling  :meth:`TensorQuantizer.set_from_attribute_config() <modelopt.torch.quantization.nn.modules.tensor_quantizer.TensorQuantizer.set_from_attribute_config>`.
If the quantization parameters are not set explicitly, the quantizer will use the default values.

Here is an example of creating a quantizer module:

.. code-block:: python

    from modelopt.torch.quantization.config import QuantizerAttributeConfig
    from modelopt.torch.quantization.nn import TensorQuantizer

    # Create quantizer module with default quantization parameters
    quantizer = TensorQuantizer()

    quant_x = quantizer(x)  # Quantize input x

    # Create quantizer module with custom quantization parameters
    # Example setting for INT4 block-wise quantization
    quantizer_custom = TensorQuantizer(QuantizerAttributeConfig(num_bits=4, block_sizes={-1: 128}))

    # Quantize input with custom quantization parameters
    quant_x = quantizer_custom(x)  # Quantize input x


.. _customize_quantizer_config:

Customizing Quantizer Configuration
-----------------------------------

ModelOpt inserts input quantizer, weight quantizer and output quantizer into Pytorch building blocks such as ``nn.Linear``, ``nn.Conv<N>d`` layers.
By default the output quantizer is disabled.

The following examples demonstrate how to customize quantization behavior.

Basic Configuration Modification
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For debugging purposes or simple customizations, you can modify an existing configuration:

.. code-block:: python

    # Create a copy of the default INT8 configuration
    config = mtq.INT8_DEFAULT_CFG.copy()

    # Disable input quantizers for all layers
    config["quant_cfg"]["*input_quantizer"]["enable"] = False

    # Disable all quantizers for layers matching the pattern "layer1.*"
    config["quant_cfg"]["*layer1.*"] = {"enable": False}

Advanced Configuration Creation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For exploring new quantization recipes, you can compose a completely new configuration. The example below creates a custom configuration with INT4 block-wise weight quantization and INT8 token-wise dynamic activation quantization:

.. code-block:: python

    # Custom configuration for INT4 block-wise weights and INT8 dynamic activations
    MY_CUSTOM_CONFIG = {
        "quant_cfg": {
            # Configure weight quantizers with 4-bit precision and 128-element blocks
            "*weight_quantizer": {"num_bits": 4, "block_sizes": {-1: 128}, "enable": True},

            # Configure input quantizers with 8-bit dynamic quantization
            "*input_quantizer": {"num_bits": 8, "type": "dynamic", "block_sizes": {-1: None}},

            # Include default disabled quantizer configurations
            **_default_disabled_quantizer_cfg,
        },
        "algorithm": "max",
    }

.. note::

    For a detailed explanation of each configuration field, please refer to the source code in
    ``modelopt/torch/quantization/config.py``.

.. _custom_quantied_module:

Custom quantized module and quantizer placement
-----------------------------------------------

``modelopt.torch.quantization`` has a default set of quantized modules (see :mod:`modelopt.torch.quantization.nn.modules <modelopt.torch.quantization.nn.modules>` for a detailed list) and quantizer placement rules (input, output and weight quantizers).
However, there might be cases where you want to define a custom quantized module and/or customize the quantizer placement.

ModelOpt provides a way to define custom quantized modules and register them with the quantization framework. This allows you to:

#. Handle unsupported modules, e.g., a subclassed Linear layer that require quantization.
#. Customize the quantizer placement, e.g., placing the quantizer in special places like the KV Cache of an Attention layer.

The custom quantized modules must have a ``_setup`` method which instantiates the quantizers that are called in the forward method.

Here is an example of defining a custom quantized linear module:

.. note::

     ModelOpt assigns a ``parallel_state`` of type :class:`ParallelState <modelopt.torch.utils.distributed.ParallelState>`
     to each module. The ``parallel_state`` of each module specifies its distributed parallelism such as ``data_parallel_group`` and ``tensor_parallel_group``.

     The ``parallel_state`` groups are used to correctly synchronize the quantization parameters across different process groups during calibration.

     The ``parallel_state`` by default configures the default PyTorch distributed process group as the ``data_parallel_group`` - the specialized process groups such as ``tensor_parallel_group`` are
     set to ``-1`` which means this parallelism is not used.

     When working with distributed training or inference with specialized parallelism like tensor parallelism,
     you need to initialize the correct ``parallel_state`` in the ``_setup`` method. This will override the default ``parallel_state``
     and use the correct parallel groups for that module. ModelOpt provides built-in support for common parallel libraries
     like Megatron-LM, APEX, FairScale etc, through plugin modules :mod:`plugins <modelopt.torch.quantization.plugins>` that automatically handle
     the correct parallel state initialization.

.. code-block:: python

    from modelopt.torch.quantization.nn import TensorQuantizer
    from modelopt.torch.utils.distributed import ParallelState

    # Quantized module for `CustomColumnParallelLinear`
    class QuantColumnParallelLinear(CustomColumnParallelLinear):
        def __init__(self, in_features, out_features):
            super().__init__(in_features, out_features)
            self._setup()

        def _setup(self):
            # Method to setup the quantizers
            self.input_quantizer = TensorQuantizer()
            self.weight_quantizer = TensorQuantizer()

            # Optional step for specialized distributed parallel training/inference
            self.parallel_state = ParallelState(
                data_parallel_group=data_parallel_group, # specify the data parallel group (user defined)
                tensor_parallel_group=tensor_parallel_group, # specify the tensor parallel group (user defined)
            )

        def forward(self, input):
            # You can customize the quantizer placement anywhere in the forward method
            input = self.input_quantizer(input)
            weight = self.weight_quantizer(self.weight)
            # Call the unquantized forward method, for example
            return custom_tensor_parallel_linear_func(input, weight, bias=self.bias)

After defining the custom quantized module, you need to register this module so ``mtq.quantize`` API will automatically replace the original module with the quantized version.
Here is the code to register the custom quantized module:

.. code-block:: python

    import modelopt.torch.quantization as mtq

    # Register the custom quantized module
    mtq.register(original_cls=CustomColumnParallelLinear, quantized_cls=QuantColumnParallelLinear)

    # Perform PTQ
    # CustomColumnParallelLinear modules in the model will be replaced with the QuantColumnParallelLinear module
    model = mtq.quantize(model, config, forward_loop)

The quantization config might need to be customized if you define a custom quantized module. Please see
:ref:`customizing quantizer config <customize_quantizer_config>` for more details.

.. _custom_calibration_algorithm:

Custom calibration algorithm
----------------------------

ModelOpt provides a set of quantization calibration algorithms such as awq, smoothquant, and max calibration. However, there might be cases where you want to define a custom calibration algorithm for quantizing your model. You can do this by creating a custom calibration algorithm derived from :mod:`BaseCalibrateModeDescriptor <modelopt.torch.quantization.mode.BaseCalibrateModeDescriptor>` and registering to :mod:`CalibrateModeRegistry <modelopt.torch.quantization.mode.CalibrateModeRegistry>`.
Write a custom calibration class derived from :mod:`BaseCalibrateModeDescriptor <modelopt.torch.quantization.mode.BaseCalibrateModeDescriptor>` and it should define attributes such as ``convert``, ``config_class``, ``restore`` etc. Find more details on these functions and their arguments in :mod:`BaseCalibrateModeDescriptor <modelopt.torch.quantization.mode.BaseCalibrateModeDescriptor>` and :mod:`ModeDescriptor <modelopt.torch.opt.mode.ModeDescriptor>`.

Here is an example of creating custom calibration mode:

.. code-block:: python

    from modelopt.torch.opt.config import ModeloptField
    from modelopt.torch.quantization.config import QuantizeAlgorithmConfig
    from modelopt.torch.quantization.mode import CalibrateModeRegistry, BaseCalibrateModeDescriptor
    # custom configuration comprising of method name and
    # any other parameters required by custom calibration function
    class CustomConfig(QuantizeAlgorithmConfig):
        method: Literal["custom_calib"] = ModeloptField("custom_calib")
        ...

    # custom calibration mode class to register to base calibrator
    @CalibrateModeRegistry.register_mode
    class CustomCalibrateModeDescriptor(BaseCalibrateModeDescriptor):
        @property
        def config_class(self) -> QuantizeAlgorithmConfig:
            """Specifies the config class."""
            return CustomConfig

        # define attributes such as `convert`, `restore`, `update_for_save`
        # see `BaseCalibrateModeDescriptor` for more details
        @property
        def convert(self) -> ConvertEntrypoint:
        ...


You can specify ``custom_calib`` as ``algorithm`` in ``quant_cfg`` to use it. Here is an example of using a custom calibrator to quantize your model:

.. code-block:: python

    # create quantization configuration with "custom_calib" method
    quant_cfg = {
        'quant_cfg': {'*weight_quantizer': ..},
        'algorithm':  {"method": 'custom_calib'},
    }


    model = mtq.quantize(model, quant_cfg, forward_loop)

Fast evaluation
---------------

Weight folding avoids repeated quantization of weights during each inference forward pass and speedup evaluation. This can be done with the following code:

.. code-block:: python

    # Fold quantizer together with weight tensor
    mtq.fold_weight(quantized_model)

    # Run model evaluation
    user_evaluate_func(quantized_model)

.. note::

    After weight folding, the model can no longer be exported to ONNX or fine-tuned with QAT.
