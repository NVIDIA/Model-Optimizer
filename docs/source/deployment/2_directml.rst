.. _DirectML_Deployment:

===================
DirectML
===================


Once an ONNX FP16 model is quantized using Model Optimizer on Windows, the resulting quantized ONNX model can be deployed on the DirectML (DML) backend via the `ONNX Runtime GenAI <https://onnxruntime.ai/docs/genai/>`_ or `ONNX Runtime <https://onnxruntime.ai/>`_.

.. note:: Currently, DirectML backend doesn't support 8-bit precision. So, 8-bit quantized models should be deployed on other backends like ORT-CUDA etc. However, DML path does support INT4 quantized models.

ONNX Runtime GenAI
==================

ONNX Runtime GenAI offers a streamlined solution for deploying generative AI models with optimized performance and functionality.

**Key Features**:

- **Enhanced Optimizations**: Supports optimizations specific to generative AI, including efficient KV cache management and logits processing.
- **Flexible Sampling Methods**: Offers various sampling techniques, such as greedy search, beam search, and top-p/top-k sampling, to suit different deployment needs.
- **Control Options**: Use the high-level ``generate()`` method for rapid deployment or execute each iteration of the model in a loop for fine-grained control.
- **Multi-Language API Support**: Provides APIs for Python, C#, and C/C++, allowing seamless integration across a range of applications.

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

The ready-to-deploy optimized ONNX models from ModelOpt-Windows are available at HuggingFace `NVIDIA collections <https://huggingface.co/collections/nvidia/optimized-onnx-models-for-nvidia-rtx-gpus>`_. These models can be deployed using DirectML backend. Follow the instructions provided along with the published models for deployment.
