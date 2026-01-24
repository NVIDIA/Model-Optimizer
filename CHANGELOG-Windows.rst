NVIDIA Model Optimizer Changelog (Windows)
==========================================

0.41 (TBD)
^^^^^^^^^^

**Bug Fixes**

- Fix ONNX 1.19 compatibility issues with CuPy during ONNX INT4 AWQ quantization. ONNX 1.19 uses ml_dtypes.int4 instead of numpy.int8 which caused CuPy failures.

**New Features**

- Add support for ONNX Mixed Precision Weight-only quantization using INT4 and INT8 precisions. Refer quantization `example for GenAI LLMs <https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/windows/onnx_ptq/genai_llm>`_.
- Add support for some diffusion models' quantization on Windows. Refer `example script <https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/windows/torch_onnx/diffusers>`_ for details.
- Add `Perplexity <https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/windows/accuracy_benchmark/perplexity_metrics>`_ and `KL-Divergence <https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/windows/accuracy_benchmark/kl_divergence_metrics>`_ accuracy benchmarks.

0.33 (2025-07-21)
^^^^^^^^^^^^^^^^^

**New Features**

- Model Optimizer for Windows now supports `NvTensorRtRtx <https://onnxruntime.ai/docs/execution-providers/TensorRTRTX-ExecutionProvider.html>`_ execution-provider.


0.27 (2025-04-30)
^^^^^^^^^^^^^^^^^

**New Features**

- New LLM models like DeepSeek etc. are supported with ONNX INT4 AWQ quantization on Windows. Refer `Windows Support Matrix <https://nvidia.github.io/Model-Optimizer/guides/0_support_matrix.html>`_ for details about supported features and models.
- Model Optimizer for Windows now supports ONNX INT8 and FP8 quantization (W8A8) of SAM2 and Whisper models. Check `example scripts <https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/windows/onnx_ptq>`_ for getting started with quantizing these models.


0.19 (2024-11-18)
^^^^^^^^^^^^^^^^^

**New Features**

- This is the first official release of Model Optimizer for Windows
- **ONNX INT4 Quantization:** :meth:`modelopt.onnx.quantization.quantize_int4 <modelopt.onnx.quantization.int4.quantize>` now supports ONNX INT4 quantization for DirectML and TensorRT* deployment. See :ref:`Support_Matrix` for details about supported features and models.
- **LLM Quantization with Olive:** Enabled LLM quantization through Olive, streamlining model optimization workflows. Refer `Olive example <https://github.com/microsoft/Olive/tree/main/examples/phi3#quantize-models-with-nvidia-Model-Optimizer>`_.
- **DirectML Deployment Guide:** Added DML deployment guide. Refer :ref:`Onnxruntime_Deployment` deployment guide for details.
- **MMLU Benchmark for Accuracy Evaluations:** Introduced `MMLU benchmarking <https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/windows/accuracy_benchmark/README.md>`_ for accuracy evaluation of ONNX models on DirectML (DML).
- **Published quantized ONNX models collection:** Published quantized ONNX models at HuggingFace `NVIDIA collections <https://huggingface.co/collections/nvidia/optimized-onnx-models-for-nvidia-rtx-gpus>`_.


\* *This version includes experimental features such as TensorRT deployment of ONNX INT4 models, PyTorch quantization and sparsity. These are currently unverified on Windows.*
