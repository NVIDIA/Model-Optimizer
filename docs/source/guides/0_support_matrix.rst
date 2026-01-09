.. _Support_Matrix:

==============
Support Matrix
==============

Feature Support Matrix
======================

.. tab:: Linux

    .. list-table::
        :widths: 20 40 20 20
        :header-rows: 1
        :stub-columns: 1

        * - Quantization Format
          - Details
          - Supported Model Formats
          - Deployment
        * - FP4
          - * Per-Block FP4 Weight & Activations
            * GPUs: Blackwell and Later
          - PyTorch
          - TensorRT, TensorRT-LLM
        * - FP8
          - * Per-Tensor FP8 Weight & Activations
            * GPUs: Ada and Later
          - PyTorch, ONNX*
          - TensorRT*, TensorRT-LLM
        * - INT8
          - * Per-channel INT8 Weights, Per-Tensor INT8 Activations
            * Uses Smooth Quant Algorithm
            * GPUs: Ampere and Later
          - PyTorch, ONNX*
          - TensorRT*, TensorRT-LLM
        * - W4A16 (INT4 Weights Only)
          - * Block-wise INT4 Weights, F16 Activations
            * Uses AWQ Algorithm
            * GPUs: Ampere and Later
          - PyTorch, ONNX
          - TensorRT, TensorRT-LLM
        * - W4A8 (INT4 Weights, FP8 Activations)
          - * Block-wise INT4 Weights, Per-Tensor FP8 Activations
            * Uses AWQ Algorithm
            * GPUs: Ada and Later
          - PyTorch*, ONNX*
          - TensorRT-LLM

.. tab:: Windows

    .. list-table::
        :widths: 20 40 20 20
        :header-rows: 1
        :stub-columns: 1

        * - Quantization Format
          - Details
          - Supported Model Formats
          - Deployment
        * - W4A16 (INT4 Weights Only)
          - * Block-wise INT4 Weights, F16 Activations
            * Uses AWQ Algorithm
            * GPUs: Ampere and Later
          - PyTorch*, ONNX
          - ORT-DirectML, TensorRT*, TensorRT-LLM*
        * - W4A8 (INT4 Weights, FP8 Activations)
          - * Block-wise INT4 Weights, Per-Tensor FP8 Activations
            * Uses AWQ Algorithm
            * GPUs: Ada and Later
          - PyTorch*
          - TensorRT-LLM*
        * - FP8
          - * Per-Tensor FP8 Weight & Activations (PyTorch)
            * Per-Tensor Activation and Per-Channel Weights quantization (ONNX)
            * Uses Max calibration
            * GPUs: Ada and Later
          - PyTorch*, ONNX
          - TensorRT*, TensorRT-LLM*, ORT-CUDA
        * - INT8
          - * Per-Channel INT8 Weights, Per-Tensor INT8 Activations
            * Uses Smooth Quant (PyTorch)*, Max calibration (ONNX)
            * GPUs: Ada and Later
          - PyTorch*, ONNX
          - TensorRT*, TensorRT-LLM*, ORT-CUDA

.. note:: Features marked with an asterisk (*) are considered experimental.


Model Support Matrix
====================

.. tab:: Linux

    Please checkout the model support matrix `here <https://github.com/NVIDIA/Model-Optimizer?tab=readme-ov-file#model-support-matrix>`_.

.. tab:: Windows

    .. list-table::
        :header-rows: 1

        * - Model
          - ONNX INT4 AWQ (W4A16)
          - ONNX INT8 Max (W8A8)
          - ONNX FP8 Max (W8A8)
        * - Llama3.1-8B-Instruct
          - Yes
          - No
          - No
        * - Phi3.5-mini-Instruct
          - Yes
          - No
          - No
        * - Mistral-7B-Instruct-v0.3
          - Yes
          - No
          - No
        * - Llama3.2-3B-Instruct
          - Yes
          - No
          - No
        * - Gemma-2b-it
          - Yes
          - No
          - No
        * - Gemma-2-2b
          - Yes
          - No
          - No
        * - Gemma-2-9b
          - Yes
          - No
          - No
        * - Nemotron Mini 4B Instruct
          - Yes
          - No
          - No
        * - Qwen2.5-7B-Instruct
          - Yes
          - No
          - No
        * - DeepSeek-R1-Distill-Llama-8B
          - Yes
          - No
          - No
        * - DeepSeek-R1-Distil-Qwen-1.5B
          - Yes
          - No
          - No
        * - DeepSeek-R1-Distil-Qwen-7B
          - Yes
          - No
          - No
        * - DeepSeek-R1-Distill-Qwen-14B
          - Yes
          - No
          - No
        * - Mistral-NeMo-Minitron-2B-128k-Instruct
          - Yes
          - No
          - No
        * - Mistral-NeMo-Minitron-4B-128k-Instruct
          - Yes
          - No
          - No
        * - Mistral-NeMo-Minitron-8B-128k-Instruct
          - Yes
          - No
          - No
        * - whisper-large
          - No
          - Yes
          - Yes
        * - sam2-hiera-large
          - No
          - Yes
          - Yes

  .. note::
    - ``ONNX INT8 Max`` means INT8 (W8A8) quantization of ONNX model using Max calibration. Similar holds true for the term ``ONNX FP8 Max``.
    - The LLMs in above table are `GenAI <https://github.com/microsoft/onnxruntime-genai/>`_ built LLMs unless specified otherwise.
    - Check `examples <https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/windows/onnx_ptq/>`_ for specific instructions and scripts.
