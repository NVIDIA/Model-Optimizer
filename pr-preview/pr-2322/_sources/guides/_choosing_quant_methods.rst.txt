=======================================================
Best practices to choose the right quantization methods
=======================================================

A quantization method comprises three primary components:

1. Weight precision format
2. Activation precision format
3. Calibration algorithms

Typically, in the context of small-batch inference scenarios (batch size ≤ 4), the inference is often 'memory-bound'. In memory-bound inference, the throughput is limited by the weight loading time from GPU memory to GPU cache - i.e, inference is memory bandwidth limited.
In this regime of operation, weight-only quantization methods such as INT4 AWQ or INT4-FP8 AWQ gives superior performance improvement.

Conversely, for large-batch inference scenarios, such as serving scenarios (batch size ≥ 16), both memory bandwidth and computation density become crucial factors.
Consequently, it's recommended to opt for a quantization method that has both weights & activation quantization as well as lower precision computation kernels. For batch size ≥ 16, the choice of quantization method can be model specific.

We suggest prioritizing using FP8 first, as FP8 causes very little accuracy degradation and gives strong performance.
If FP8 performance does not meet your requirements, you could try INT4-FP8 AWQ.
If your deployment is on Ampere GPUs or earlier, we recommend using INT4 AWQ or INT8 SQ.

Based on specific use cases, users might have different tolerances on accuracy degradation and calibration time. The table below summarizes the tradeoffs* to consider when choosing a quantization method.

+-----------------------+-------------+-------------+-------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Quantization Methods  | Performance | Performance | Accuracy    | Details                                                                                                                                                          |
|                       | small-batch | large-batch | degradation |                                                                                                                                                                  |
+=======================+=============+=============+=============+==================================================================================================================================================================+
| FP8                   | Medium      | Medium      | Very Low    | * FP8 per-tensor weight & activation quantization with min-max calibration.                                                                                      |
|                       |             |             |             | * Compresses FP16/BF16 model to 50% of original size.                                                                                                            |
|                       |             |             |             | * Calibration time: minutes**.                                                                                                                                   |
|                       |             |             |             | * Deploy via TensorRT, TensorRT-LLM. Supported GPUs: Ada, Hopper and later.                                                                                      |
+-----------------------+-------------+-------------+-------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| INT8 SmoothQuant      | Medium      | Medium      | Medium      | * 8-bit integer quantization with a variant of `SmoothQuant <https://arxiv.org/pdf/2211.10438.pdf>`_ calibration.                                                |
|                       |             |             |             | * Per-channel weight quantization, per-tensor activation quantization.                                                                                           |
|                       |             |             |             | * Compresses FP16/BF16 model to 50% of original size                                                                                                             |
|                       |             |             |             | * Calibration time: minutes**.                                                                                                                                   |
|                       |             |             |             | * Deploy using TensorRT, TensorRT-LLM. Supported on most GPUs.                                                                                                   |
+-----------------------+-------------+-------------+-------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| INT4 Weights only AWQ | High        | Low         | Low         | * 4-bit integer group-wise/block-wise weight only quantization with `AWQ <https://arxiv.org/pdf/2306.00978.pdf>`_ calibration.                                   |
| (W4A16)               |             |             |             | * Compresses FP16/BF16 model to 25% of original size.                                                                                                            |
|                       |             |             |             | * Calibration time: tens of minutes**.                                                                                                                           |
|                       |             |             |             | * Deploy via TensorRT-LLM. Supported GPUs: Ampere and later.                                                                                                     |
+-----------------------+-------------+-------------+-------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| INT4-FP8 AWQ (W4A8)   | High        | Medium      | Low         | * 4-bit integer group-wise/block-wise weight quantization, FP8 per-tensor activation quantization & `AWQ <https://arxiv.org/pdf/2306.00978.pdf>`_ calibration.   |
|                       |             |             |             | * Compresses FP16/BF16 model to 25% of original size.                                                                                                            |
|                       |             |             |             | * Calibration time: tens of minutes**.                                                                                                                           |
|                       |             |             |             | * Deploy via TensorRT-LLM. Supported GPUs: Ada, Hopper and later.                                                                                                |
+-----------------------+-------------+-------------+-------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------+

| * The performance and impact are measured on 10+ popular LLMs. We'll follow up with more data points.
| ** Calibration time is subject to the actual model size.

Please see how to apply these quantization methods below:
    * :doc:`Quantizing pytorch models <../guides/_pytorch_quantization>`
    * :doc:`Quantizing ONNX models <../guides/_onnx_quantization>`
