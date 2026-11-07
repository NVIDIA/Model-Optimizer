==========================
TensorRT-LLM
==========================

For current TensorRT-LLM deployments, export quantized models with
:meth:`export_hf_checkpoint <modelopt.torch.export.unified_export_hf.export_hf_checkpoint>`
and load the exported Hugging Face checkpoint with TensorRT-LLM's PyTorch backend.
See the :doc:`unified HF export guide <3_unified_hf>` for export and deployment
examples, supported models, and quantization formats. This workflow does not require
building a TensorRT engine.

.. warning::

    The ``export_tensorrt_llm_checkpoint`` API exports checkpoints for the legacy
    TensorRT backend, which current TensorRT-LLM releases no longer support.
    The API is deprecated as of 0.48.0 and will be removed in 0.49.0.
    Use ``export_hf_checkpoint`` instead.
