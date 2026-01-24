========
Overview
========

**NVIDIA Model Optimizer**
===================================

Minimizing inference costs presents a significant challenge as generative AI models continue to grow in complexity and size.
The `NVIDIA Model Optimizer <https://github.com/NVIDIA/Model-Optimizer>`_ (referred to as Model Optimizer, or ModelOpt)
is a library comprising state-of-the-art model optimization techniques including quantization and sparsity to compress model.
It accepts a torch or ONNX model as input and provides Python APIs for users to easily stack different model optimization
techniques to produce optimized & quantized checkpoints. Seamlessly integrated within the NVIDIA AI software ecosystem, the quantized checkpoint generated from Model Optimizer is ready for deployment in downstream inference frameworks like `TensorRT-LLM <https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/quantization>`_ or `TensorRT <https://github.com/NVIDIA/TensorRT>`_ (Linux). ModelOpt is integrated with `NVIDIA NeMo <https://github.com/NVIDIA-NeMo/NeMo>`_ and `Megatron-LM <https://github.com/NVIDIA/Megatron-LM>`_ for training-in-the-loop optimization techniques. For enterprise users, the 8-bit quantization with Stable Diffusion is also available on `NVIDIA NIM <https://developer.nvidia.com/blog/nvidia-nim-offers-optimized-inference-microservices-for-deploying-ai-models-at-scale/>`_.

For Windows users, the `Model Optimizer for Windows <https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/windows>`_ (ModelOpt-Windows) delivers model compression techniques, including quantization, on Windows RTX PC systems. ModelOpt-Windows is optimized for efficient quantization, featuring local GPU calibration, reduced system and video memory consumption, and swift processing times. It integrates seamlessly with the Windows ecosystem, with optimized ONNX models as output for `Microsoft DirectML <https://github.com/microsoft/DirectML>`_ and `TensorRT-RTX <https://github.com/NVIDIA/TensorRT-RTX>`_ backends. Furthermore, ModelOpt-Windows supports SDKs such as `Microsoft Olive <https://github.com/microsoft/Olive>`_ and `ONNX Runtime <https://github.com/microsoft/onnxruntime>`_, enabling the deployment of quantized models across various independent hardware vendors through the DirectML path.

Model Optimizer for both Linux and Windows are available for free for all developers on `NVIDIA PyPI <https://pypi.org/project/nvidia-modelopt/>`_. Visit the `Model Optimizer GitHub repository <https://github.com/NVIDIA/Model-Optimizer>`_ for end-to-end
example scripts and recipes optimized for NVIDIA GPUs.

Techniques
----------

Quantization
^^^^^^^^^^^^
Quantization is an effective model optimization technique for large models. Quantization with Model Optimizer can compress
model size by 2x-4x, speeding up inference while preserving model quality. Model Optimizer enables highly performant
quantization formats including NVFP4, FP8, INT8, INT4, etc and supports advanced algorithms such as SmoothQuant, AWQ, SVDQuant, and
Double Quantization with easy-to-use Python APIs. Both Post-training quantization (PTQ) and Quantization-aware training (QAT)
are supported. Visit :meth:`Quantization Format page <modelopt.torch.quantization.config>`
for list of formats supported.

Distillation
^^^^^^^^^^^^
Knowledge Distillation is the use of an existing pretrained "teacher" model to train a smaller, more efficient "student" model.
It allows for increasing the accuracy and/or convergence speed over traditional training.
The feature maps and logits of the teacher and student become the targets and predictions for the (user-specified) loss, respectively.
Model Optimizer allows for minimally-invasive integration of teacher-student Knowledge Distillation into an existing training pipeline
using the :meth:`mtd.convert() <modelopt.torch.distill.distillation.convert>` API.

Pruning
^^^^^^^

Pruning is a technique to reduce the model size and accelerate the inference by removing unnecessary weights.
Model Optimizer provides the Python API :meth:`mtp.prune() <modelopt.torch.prune.pruning.prune>` to prune Linear and
Conv layers, and Transformer attention heads, MLP, and depth through various different state of the art algorithms.

Speculative Decoding
^^^^^^^^^^^^^^^^^^^^

Speculative Decoding enables your model to generate multiple tokens in each generate step by using a draft model to
predict tokens that are then validated by the original model in a single forward pass.
This can be useful for reducing the latency of your model and speeding up inference.
Model Optimizer provides the Python API :meth:`mtsp.convert() <modelopt.torch.speculative.speculative_decoding.convert>` to
add speculative decoding module to your model.

Sparsity
^^^^^^^^
Sparsity is a technique to further reduce the memory footprint of deep learning models and accelerate the inference.
Model Optimizer provides the Python API :meth:`mts.sparsify() <modelopt.torch.sparsity.sparsification.sparsify>` to
automatically apply weight sparsity to a given model. The
:meth:`mts.sparsify() <modelopt.torch.sparsity.sparsification.sparsify>` API supports
`NVIDIA 2:4 <https://arxiv.org/pdf/2104.08378>`_ sparsity pattern and various sparsification methods,
such as `NVIDIA ASP <https://github.com/NVIDIA/apex/tree/master/apex/contrib/sparsity>`_ and
`SparseGPT <https://arxiv.org/abs/2301.00774>`_. It supports both post-training sparsity (PTS) and
sparsity-aware training (SAT). The latter workflow is recommended to minimize accuracy
degradation.
