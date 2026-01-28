.. _ONNX_PTQ_Guide_Windows:

===========================
ONNX Quantization - Windows
===========================

Quantization is a powerful method to reduce model size and boost computational efficiency. By converting model weights to low-precision formats, significant storage and memory savings are achieved. Quantization applies lower-precision data types to model parameters; for example:

- FP16 uses 2 bytes per value.
- INT4 requires only 4 bits (0.5 bytes) per value.

This transformation reduces model size and allows deployment on systems with limited memory.

**Quantization Techniques**:

1. **Post-Training Quantization (PTQ)**: Quantization is applied after model training.
2. **Quantization-Aware Training (QAT)**: Quantization is integrated during model training.


ModelOpt-Windows Quantization
-----------------------------

The Model Optimizer - Windows is designed to create optimized ONNX models for DirectML and TensorRT* backends.

**Supported Techniques**:

- PTQ: Supported in ONNX
- QAT: Experimental for PyTorch

Refer :ref:`Support_Matrix` for details about supported features and models.

Preparing and Optimizing Base Models for ONNX PTQ
-------------------------------------------------

ModelOpt-Windows's ONNX PTQ API requires a base ONNX model, which can be obtained from Hugging Face or various ONNX exporter tools such as:

- `torch.onnx.export <https://pytorch.org/docs/stable/onnx.html>`_
- `HuggingFace-Optimum <https://huggingface.co/docs/optimum/en/exporters/onnx/usage_guides/export_a_model/>`_
- `GenAI <https://github.com/microsoft/onnxruntime-genai/>`_
- `Olive <https://github.com/microsoft/Olive/>`_

Each tool offers unique features and options for conversion from PyTorch or other frameworks to the ONNX format.

**Opset requirements of different data-types**: ModelOpt-Windows uses ONNX's `Q <https://onnx.ai/onnx/operators/onnx__QuantizeLinear.html>`_/`DQ <https://onnx.ai/onnx/operators/onnx__DequantizeLinear.html>`_ nodes for applying quantization and dequantization operations in the ONNX model. The INT4 datatype support in Q/DQ node came in opset-21 and, FP8 datatype support in Q/DQ node came in opset-19. So, ensure that model's opset is 19+ for FP8 quantization, and it is 21+ for INT4 quantization. This is needed for deployment of the quantized model on onnxruntime framework (e.g. ORT-DirectML backend). Different ONNX exporter tools usually have option or argument for target 'opset' field. See their documentation for details about its usage and max-supported opset limit.

**Base Model Precision**: ModelOpt-Windows supports base models in both FP16 and FP32 formats. Choosing FP16 over FP32 can help reduce memory usage and improve speed, especially on hardware optimized for lower precision, such as NVIDIA GPUs with Tensor Cores. However, FP16's smaller dynamic range may require careful tuning.

**ONNX FP16 Conversion Tools**: Some popular FP32 to FP16 ONNX conversion tools include:

- `ONNX Converter Tool  <https://onnxruntime.ai/docs/performance/model-optimizations/float16.html>`_
- Hugging Face *Optimum* tool with a *dtype* argument for FP16 generation - Refer to optimum's `CLI <https://huggingface.co/docs/optimum/en/exporters/onnx/usage_guides/export_a_model/>`_, and `API <https://github.com/huggingface/optimum/blob/main/optimum/exporters/onnx/convert.py/>`_ usage.
- Microsoft Olive, which supports FP16 via configuration files. Refer *float16* option in this example `config <https://github.com/microsoft/Olive/blob/main/examples/directml/llm/config_llm.json/>`_.

Once base model is obtained, ModelOpt-Windows's PTQ can be applied to get the quantized mode. The resulting quantized model can be deployed on backends like DirectML, CUDA and TensorRT*.

.. _Apply_ONNX_PTQ:

Apply Post Training Quantization (PTQ)
--------------------------------------

Applying PTQ on a model involves preparing calibration-data (if needed), invoking quantization API, saving the quantized model, and any additional post-processing like opset upgrade as needed.

**Prepare calibration data**

Quantization algorithms like SmoothQuant (SQ), Activation-Aware-Quantization (AWQ), and static quantization of activations often require calibration data. If the *quantize* API's calibration-data argument is not provided (i.e., set to *None*), ModelOpt-Windows will internally use randomly generated model inputs for calibration.

As an example, preparing calibration data for INT4 AWQ quantization of LLMs may involve following major steps:

1. **Generate Token Encodings**: Use a dataset like *cnn-dailymail* or *pile* with the model's tokenizer to generate token encodings and related data from the representative dataset
2. **Format for Model Input**: Convert encodings into model-compatible formats.

Please refer the `example scripts <https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/windows/onnx_ptq/>`_ for details about preparing calibration-data of various supported ONNX models.

**Call Quantization API**

The example below demonstrates how to apply INT4 AWQ quantization on a LLM ONNX model.

.. code-block:: python

    from modelopt.onnx.quantization.int4 import quantize as quantize_int4
    # import other packages as needed
    calib_inputs = get_calib_inputs(dataset, model_name, cache_dir, calib_size, batch_size,...)
    quantized_onnx_model = quantize_int4(
        onnx_path,
        calibration_method="awq_lite",
        calibration_data_reader=None if use_random_calib else calib_inputs,
        calibration_eps=["dml", "cpu"]
    )

Check :meth:`modelopt.onnx.quantization.quantize_int4 <modelopt.onnx.quantization.int4.quantize>` for details about quantization API.

**Upgrade opset of the model**

Opset requirement for different data-types is already explained in the section describing ways to obtain base model. To summarize, the opset requirements stems from the fact that support for different types in ONNX `Q <https://onnx.ai/onnx/operators/onnx__QuantizeLinear.html>`_/`DQ <https://onnx.ai/onnx/operators/onnx__DequantizeLinear.html>`_ nodes is gradually added in different opsets. For instance, the INT4 data-type support in Q/DQ node came in opset-21 and, FP8 data-type support in Q/DQ node came in opset-19. So, ensure that model's opset is 19+ for FP8 quantization, and it is 21+ for INT4 quantization. This is needed for deployment of the quantized model on onnxruntime framework (e.g. ORT-DirectML backend).

Generally, different ONNX exporter tools have option or argument for providing the desired or target opset. It is possible that the desired opset is greater than the max-opset user's ONNX exporter tool supports. In that case, user would need to manually 'patch' the opset of the ONNX model. This would require updating the ONNX metadata that stores opset field of the graph, and it may additionally require updating some nodes in the graph as per new opset (if they have changed in later opsets). Alternatively, one can try using the ONNX exporter tool which already supports the desired opset (if exists any).

A few sample code snippets to inspect the opset of the given ONNX model, and to update the opset field in the ONNX model's meta-data, are provided below.

.. code-block:: python

  # Example steps to check opset

  def get_onnx_opset(model_path):
    # Load the ONNX model
    model = onnx.load(model_path)

    # Get the opset import information
    opset_imports = model.opset_import

    # Print opset information
    for opset in opset_imports:
      print(f"Domain: {opset.domain}")
      print(f"  Version: {opset.version}\n")

Use the above steps to inspect the ONNX model's opset version.

    *ONNX Opset Upgrade Tools*:
      - `ONNX opset conversion utility <https://github.com/onnx/onnx/blob/main/onnx/version_converter.py>`_
      - `Optimum-CLI <https://huggingface.co/docs/optimum/en/exporters/onnx/usage_guides/export_a_model>`_

    .. code-block:: python

      # Example steps for updating opset metadata of default (onnx) domain
      # Update opsets for other domains as needed for your requirement (or exclude them as suitable).

      model = onnx.load(onnx_path)

      op = onnx.OperatorSetIdProto()
      op.version = 21
      new_opset_imports = [
          helper.make_opsetid("", 21),  # Default domain with opset version 15
          #helper.make_opsetid("ai.onnx.ml", 2)  # ai.onnx.ml domain with opset version 2
          helper.make_opsetid("com.microsoft", 1)
      ]

      updated_quantized_onnx_model = onnx.helper.make_model(model.graph, opset_imports=new_opset_imports)

The ONNX models produced using `GenAI <https://github.com/microsoft/onnxruntime-genai/>`_ are generally seen to work fine with above opset upgrade patch. ONNX models produced using other ONNX exporter tool might require further post-processing on case-by-case basis for nodes that have changed in later opsets.

**Save Quantized Model**

To save a quantized ONNX model with external data, use the following code:

.. code-block:: python

    onnx.save_model(
        updated_quantized_onnx_model,
        output_path,
        save_as_external_data=True,
        location=os.path.basename(output_path) + "_data",
        size_threshold=0,
    )

Deploy Quantized ONNX Model
---------------------------

Inference of the quantized models can be done using tools like `GenAI <https://github.com/microsoft/onnxruntime-genai/>`_, `OnnxRunTime (ORT) <https://onnxruntime.ai//>`_. These APIs can do inference on backends like DML, CUDA, TensorRT-RTX. For details about onnxruntime deployment of quantized models, see :ref:`Onnxruntime_Deployment` deployment guide. Also, refer `example scripts <https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/windows/onnx_ptq/>`_ for any possible model-specific inference guidance or script (if any).
