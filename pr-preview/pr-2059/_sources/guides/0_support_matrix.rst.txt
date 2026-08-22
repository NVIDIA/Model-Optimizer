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
          - ORT-DML, ORT-CUDA, ORT-TRT-RTX, TensorRT*, TensorRT-LLM*
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

.. note:: 
  - Features marked with an asterisk (*) are considered experimental.
  - ``ORT-CUDA``, ``ORT-DML``, and ``ORT-TRT-RTX`` are ONNX Runtime Execution Providers (EPs) for CUDA, DirectML, and TensorRT-RTX respectively. Support for different deployment backends can vary across models.


Model Support Matrix
====================

.. tab:: Linux

    Please checkout the model support matrix `here <https://github.com/NVIDIA/Model-Optimizer?tab=readme-ov-file#model-support-matrix>`_.

.. tab:: Windows

    Please checkout the model support matrix `details <https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/windows#support-matrix>`_.
