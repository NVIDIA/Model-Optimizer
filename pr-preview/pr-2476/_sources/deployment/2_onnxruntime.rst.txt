.. _Onnxruntime_Deployment:

===========
Onnxruntime
===========

Once an ONNX FP16 model is quantized using Model Optimizer on Windows, the resulting quantized ONNX model can be deployed via the `ONNX Runtime GenAI <https://onnxruntime.ai/docs/genai/>`_ or `ONNX Runtime <https://onnxruntime.ai/>`_. 

ONNX Runtime uses execution providers (EPs) to run models efficiently across a range of backends, including:

- **CUDA EP:** Utilizes NVIDIA GPUs for fast inference with CUDA and cuDNN libraries.
- **DirectML EP:** Enables deployment on a wide range of GPUs.
- **TensorRT-RTX EP:** Targets NVIDIA RTX GPUs, leveraging TensorRT for further optimized inference.
- **CPU EP:** Provides a fallback to run inference on the system's CPU when specialized hardware is unavailable.

Choose the EP that best matches your model, hardware and deployment requirements.

TensorRT-RTX calibration and deployment
=======================================

ModelOpt currently uses the legacy TensorRT-RTX EP by default for calibration. The standalone ABI
EP will become the default in a future release. To use the ABI EP for calibration now, pass
``--calibration_eps NvTensorRtRtx --trt_rtx_backend abi``. CUDA EP calibration remains available
independently.

For deployment, choose one of the following TensorRT-RTX EP paths. The legacy package contains an
ONNX Runtime build with the EP included, whereas the ABI path uses a standard ONNX Runtime package
and a separately registered plugin.

Legacy TensorRT-RTX EP
----------------------

Install the legacy package and ensure that the required TensorRT-RTX libraries are available on
``PATH``:

.. code-block:: bash

    python -m pip install onnxruntime-trt-rtx

Create the inference session by selecting the built-in provider:

.. code-block:: python

    import onnxruntime as ort

    session = ort.InferenceSession(
        "model.onnx",
        providers=["NvTensorRTRTXExecutionProvider"],
    )

TensorRT-RTX ABI EP
-------------------

Install ONNX Runtime and the standalone CUDA 13 plugin package:

.. code-block:: bash

    python -m pip install "onnxruntime>=1.24" onnxruntime-ep-nv-tensorrt-rtx-cu13

Register the plugin, select its devices, and attach them to the session options before creating the
inference session:

.. code-block:: python

    import onnxruntime as ort
    import onnxruntime_ep_nv_tensorrt_rtx as trt_rtx_ep

    ep_name = trt_rtx_ep.get_ep_name()
    ort.register_execution_provider_library(ep_name, trt_rtx_ep.get_library_path())

    trt_rtx_devices = [device for device in ort.get_ep_devices() if device.ep_name == ep_name]
    if not trt_rtx_devices:
        raise RuntimeError("No TensorRT-RTX ABI EP device was found")

    session_options = ort.SessionOptions()
    session_options.add_provider_for_devices(trt_rtx_devices, {})
    session = ort.InferenceSession("model.onnx", sess_options=session_options)

    # Release every session using the plugin before unregistering it.
    del session
    ort.unregister_execution_provider_library(ep_name)

See the `ONNX Runtime TensorRT-RTX EP documentation
<https://onnxruntime.ai/docs/execution-providers/TensorRTRTX-ExecutionProvider.html>`_ for provider
options shared by the legacy and ABI paths. For ABI compatibility, packaging, and more complete
examples, refer to the `NVIDIA TensorRT-RTX EP ABI documentation
<https://github.com/NVIDIA/TensorRT-RTX-EP-ABI>`_ and the `TensorRT-RTX ABI plugin package
<https://pypi.org/project/onnxruntime-ep-nv-tensorrt-rtx-cu13/>`_.

.. note:: Currently, DirectML backend doesn't support 8-bit precision. So, 8-bit quantized models should be deployed on other backends like ORT-CUDA etc. However, DML path does support INT4 quantized models.

ONNX Runtime GenAI
==================

ONNX Runtime GenAI offers a streamlined solution for deploying generative AI models with optimized performance and functionality.

**Key Features**:

- **Enhanced Optimizations**: Supports optimizations specific to generative AI, including efficient KV cache management and logits processing.
- **Flexible Sampling Methods**: Offers various sampling techniques, such as greedy search, beam search, and top-p/top-k sampling, to suit different deployment needs.
- **Control Options**: Use the high-level ``generate()`` method for rapid deployment or execute each iteration of the model in a loop for fine-grained control.
- **Multi-Language API Support**: Provides APIs for Python, C#, and C/C++, allowing seamless integration across a range of applications.

.. note::

   ONNX Runtime GenAI models are typically tied to the execution provider (EP) they were built with; a model exported for one EP (e.g., CUDA or DirectML) is generally not compatible with other EPs. To run inference on a different backend, re-export or convert the model specifically for that target EP.

**Getting Started**:

Refer to the `ONNX Runtime GenAI documentation <https://onnxruntime.ai/docs/genai/>`_ for an in-depth guide on installation, setup, and usage.

**Examples**:

- Explore `inference scripts <https://github.com/microsoft/onnxruntime-genai/tree/main/examples/python//>`_ in the ORT GenAI example repository for generating output sequences using a single function call.
- Follow the `ORT GenAI tutorials <https://onnxruntime.ai/docs/genai/tutorials/>`_ for a step-by-step walkthrough of inference with DirectML using the ORT GenAI package (e.g., refer to the Phi3 tutorial).

ONNX Runtime
============

Alternatively, the quantized model can be deployed using `ONNX Runtime <https://onnxruntime.ai/>`_. This method requires manual management of model inputs, including KV cache inputs and attention masks, for each iteration within the generation loop.

**Examples and Documentation**

For further details and examples, please refer to the `ONNX Runtime documentation <https://onnxruntime.ai/docs/api/python/>`_.

Collection of optimized ONNX models
===================================

The ready-to-deploy optimized ONNX models from ModelOpt-Windows are available at HuggingFace `NVIDIA collections <https://huggingface.co/collections/nvidia/optimized-onnx-models-for-nvidia-rtx-gpus>`_. Follow the instructions provided along with the published models for deployment.
