
====
FAQs
====

.. _FAQ_ModelOpt_Windows:

ModelOpt-Windows
================

**ONNX PTQ**

1. Why is awq-scale search taking too long or stuck at 0% during ONNX INT4 quantization?
----------------------------------------------------------------------------------------

Awq-scale search should complete in minutes with NVIDIA GPU acceleration. If stalled:

- **GPU acceleration may be disabled.** If CUDA 12.x is not available, quantization will fall back to slower ``numpy`` implementation instead of ``cupy-cuda12x``. Make sure that ``cupy`` package is compatible with installed CUDA toolkit.
- **Low GPU memory.** Quantization needs 20-24GB VRAM; low memory forces slower shared memory usage.
- **Using CPU for quantization.** Install ORT-GPU (supports CUDA EP) or ORT-DML (supports DML EP) for better speed.

2. Why is "CUDA EP not found" error showing during ONNX quantization?
---------------------------------------------------------------------

ORT used in GenAI may conflict with ModelOpt-Windows ORT:

- Uninstall ORT, run ``pip cache purge``, and reinstall *nvidia-modelopt[onnx]*.
- Use separate virtual environments for GenAI and quantization (e.g., with *venv* or *conda*).

3. Why does ORT-session creation fail for CUDA-EP despite having CUDA toolkit and cuDNN?
----------------------------------------------------------------------------------------

This usually results from mismatched CUDA and cuDNN versions or missing paths. Ensure:

- Compatible CUDA toolkit and cuDNN versions (check `CUDA EP requirements <https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements/>`_).
- Add cuDNN's *bin* and *lib* paths to *PATH* and restart the command prompt.

4. Why quantized model's size increases on re-runs?
---------------------------------------------------

Make sure that the output directory is clean before each quantization run otherwise, existing quantized model file may get appended in each run leading to increase in model's size and possibly corrupting it.

5. Running INT4 quantized ONNX model on DirectML backend gives following error. What can be the issue?
------------------------------------------------------------------------------------------------------

    `Error Unrecognized attribute: block_size for operator DequantizeLinear`

ModelOpt-Windows uses ONNX's `DequantizeLinear <https://onnx.ai/onnx/operators/onnx__DequantizeLinear.html>`_ (DQ) nodes. The int4 data-type support in DeQuantizeLinear node came in opset-21. And, *block_size* attribute was added in DequantizeLinear node in Opset-21. Make sure that quantized model's opset version is 21 or higher. Refer :ref:`Apply_ONNX_PTQ` for details.

6. Running INT4 quantized ONNX model on DirectML backend gives following kind of error. What can be the issue?
--------------------------------------------------------------------------------------------------------------

    `Error: Type 'tensor(int4)' of input parameter (onnx::MatMul_6508_i4) of operator (DequantizeLinear) in node (onnx::MatMul_6508_DequantizeLinear) is invalid.`

One possible reason for above error is that INT4 quantized ONNX model's opset version (default or onnx domain) is less than 21. Ensure the INT4 quantized model's opset version is 21 or higher since INT4 data-type support in DequantizeLinear ONNX node came in opset-21.

7. Running 8-bit quantized ONNX model with ORT-DML gives onnxruntime error about using 8-bit data-type (e.g. INT8/FP8). What can be the issue?
-----------------------------------------------------------------------------------------------------------------------------------------------

Currently, DirectML backend (ORT-DML) doesn't support 8-bit precision. So, it expectedly complains about 8-bit data-type. Try using ORT-CUDA or other 8-bit compatible backend.

8. How to resolve onnxruntime error about invalid use of FP8 type in QuantizeLinear / DequantizeLinear node?
-------------------------------------------------------------------------------------------------------------

The FP8 type support in QuantizeLinear / DeQuantizeLinear node came with Opset-19. So, ensure that opset of ONNX model is 19+.

.. _nas_faqs:

NAS/Pruning
===========


Parallel search
---------------

If your ``score_func`` in :meth:`mtn.search() <modelopt.torch.nas.algorithms.search>`
supports parallel evaluation, you can make use of it by passing in ``DistributedDataParallel``
module to search.


Monkey-patched functions
------------------------

During the conversion process (:meth:`mtn.convert() <modelopt.torch.nas.conversion.convert>`), we
use a `monkey patch <https://en.wikipedia.org/wiki/Monkey_patch>`_ to augment the ``forward()``,
``eval()``, and ``train()`` methods of ``nn.Module``. This renders the ModelOpt conversion process
incompatible with other monkey patches to those methods.

.. code-block:: python

    # Internally in mtn.convert, we do:
    model.forward = types.MethodType(nas_forward_func, model)
    model.train = types.MethodType(nas_train_func, model)

Known Issues
============

1. Potential memory leak for ``FSDP`` with ``use_orig_params=True``
-------------------------------------------------------------------

When using ``FSDP`` with ``use_orig_params=True``, there is a potential memory leak during training
when using ``FSDP`` in conjunction with modelopt-converted models. Please use
``use_orig_params=False`` to avoid this issue.
