Basic Concepts
==============

A quantization format consists of the precision format, the block format, and the calibration
algorithm.
The detailed list of available quantization formats can be found in :any:`quantization-formats`.
Below we provide an overview of the important topics:

Precision format
****************
The precision format defines the bit-width of the quantized values. Generally, there are integer
formats (sign bit + mantissa bits) and floating-point formats (sign bit + exponent bits + mantissa
bits). `Fp8 Formats for Deep Learning <https://arxiv.org/pdf/2209.05433>`_ provides a detailed
explanation of the floating-point formats.

Scaling factor
**************
The scaling factor is a floating-point value that is used to scale and unscale the values before and
after the quantized operation, respectively. The scaling factor is used to map the range of the
original values to the range of the quantized values. The scaling factor is shared across the
quantized values in the same block. The scaling factor is calculated during the calibration process.

Block format
************
The block format defines the way the tensor is divided into blocks for sharing the scaling factors.
The most common block format is per-tensor quantization, where the whole tensor is quantized as a
single block with one global scaling factor. Other block formats include per-channel quantization,
where each channel is quantized separately, and the fine-grained per-block quantization, where the
tensor is divided into fix-size blocks along the channel dimension. For low-bit quantization (e.g.
4-bit), per-block quantization is typically needed to preserve the accuracy.

Weight and activation may share different precision and block formats. For example, in GPTQ and AWQ,
the weight is quantized to 4-bit while activation stays in high precision. Weight-only quantization
is helpful for bandwidth-constrained scenarios, while weight and activation quantization can reduce
both bandwidth and computation cost.

Calibration algorithm
*********************

The calibration algorithm calculate scaling factors and potentially adjust weights to maximize
accuracy post quantization. The simplest calibration algorithm is "max calibration", in which the
scaling factor is calculated from the global maximum of the tensor and the weights are unchanged and
rounded to the nearest quantized value. An example of a more advanced calibration algorithm is
`Entropy Calibration <https://docs.nvidia.com/deeplearning/tensorrt/latest/_static/python-api/infer/Int8/EntropyCalibrator.html>`_,
`SmoothQuant <https://arxiv.org/abs/2211.10438>`_, `AWQ <https://arxiv.org/abs/2306.00978>`_, and
`SVDQuant <https://arxiv.org/pdf/2411.05007>`_.

Quantization-aware training (QAT)
*********************************
QAT can be viewed as regular PTQ followed by fine-tuning during which the original, unquantized
weights are updated to minimize the loss. Compared to regular fine-tuning, we must model the effect
of quantization on the forward and backward passes. Commonly used QAT techniques like
`Straight-Through Estimator (STE) <https://arxiv.org/abs/1308.3432>`_ or STE with clipping have
fixed scaling factors and tune the weights during training to minimize the loss. ModelOpt implements
STE with clipping for QAT.


More Readings
*************

* Math behind quantization: `Integer Quantization <https://arxiv.org/pdf/2004.09602.pdf>`_

* Explicit quantization graph representation with QDQ node:
  `work-with-qat-networks <https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#work-with-qat-networks>`_
