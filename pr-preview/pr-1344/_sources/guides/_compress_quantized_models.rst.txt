=========================
Compress quantized models
=========================

Model compression is a technique to further reduce the memory footprint of already quantized models by storing weights in a more memory-efficient format. While quantization reduces precision to save memory, compression optimizes how these quantized weights are stored in memory.

**Key Benefits:**

* **Reduced Memory Footprint:** Compression can reduce memory usage by up to 4x compared to standard quantized formats. This enables loading very large models on small GPUs.
* **Fast Evaluation:** Compressed models can utilize low-precision kernels if available to speed up inference.

.. note::

    ModelOpt only supports compression for selected quantization formats (currently FP8 and NVFP4).


**How It Works:**

Compression reorganizes quantized weights into a more compact memory layout. For instance, with NVFP4 quantization, the compressed format eliminates storage overhead by packing multiple low-precision values efficiently. This is particularly valuable for large language models where memory is the primary constraint.

ModelOpt provides a API :meth:`mtq.compress() <modelopt.torch.quantization.compress>` to compress the model weights after quantization.
This API can be used to reduce the memory footprint of the quantized model for future evaluation or fine-tuning such as QLoRA.

After PTQ, the model can be compressed with the following code:

.. code-block:: python

    # Compress the model
    mtq.compress(model)


Initialize HF models with compressed weights for lower memory usage
===================================================================

When working with large language models, memory constraints can be a significant challenge. ModelOpt provides a workflow for initializing HF models with compressed weights across multiple GPUs to dramatically reduce memory usage.

For quantized formats like NVFP4, you can reduce memory usage by up to 4x compared to FP16/BF16 models. One limitation is that this workflow only works with max calibration algorithm.

**Example Usage:**

.. code-block:: python

    import modelopt.torch.quantization as mtq
    from modelopt.torch.quantization.plugins import init_quantized_weights
    from transformers import AutoModelForCausalLM, AutoConfig

    # Step 1: Initialize the model with compressed weights
    with init_quantized_weights(mtq.NVFP4_DEFAULT_CFG):
        model = AutoModelForCausalLM.from_pretrained(ckpt_path)

    # Step 2: Calibrate the model
    mtq.calibrate(model, algorithm="max", forward_loop=calibrate_loop)

    # downstream tasks, e.g., export quantized model/accuracy evaluation
    ...


.. note::

    An example implementation of this workflow can be found in:
    ``examples/llm_ptq/hf_ptq.py``, which reduces the memory requirements of model calibration.
