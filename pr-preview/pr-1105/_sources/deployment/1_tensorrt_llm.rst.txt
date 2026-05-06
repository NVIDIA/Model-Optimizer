==========================
TensorRT-LLM
==========================

**Deprecation Notice**: The export_tensorrt_llm_checkpoint API will be deprecated in future releases. Users are encouraged to transition to the :doc:`unified HF export API <3_unified_hf>`, which provides enhanced functionality and flexibility for exporting models to multiple inference frameworks including TensorRT-LLM, vLLM, and SGLang.

.. note::

    Please read the `TensorRT-LLM checkpoint workflow <https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/legacy/architecture/checkpoint.md>`_
    first before going through this section.



ModelOpt toolkit supports automatic conversion of ModelOpt exported LLM to the TensorRT-LLM checkpoint and the engines for accelerated inferencing.

This conversion is achieved by:

#. Converting Huggingface, NeMo and ModelOpt exported checkpoints to the TensorRT-LLM checkpoint.
#. Building TensorRT-LLM engine from the TensorRT-LLM checkpoint.


Export Quantized Model
======================

After the model is quantized, the quantized model can be exported to the TensorRT-LLM checkpoint format stored as

#. A single JSON file recording the model structure and metadata (config.json)
#. A group of safetensors files, each recording the local calibrated model on a single GPU rank (model weights, scaling factors per GPU).

The export API (:meth:`export_tensorrt_llm_checkpoint <modelopt.torch.export.model_config_export.export_tensorrt_llm_checkpoint>`) can be used as follows:

.. code-block:: python

    from modelopt.torch.export import export_tensorrt_llm_checkpoint

    with torch.inference_mode():
        export_tensorrt_llm_checkpoint(
            model,  # The quantized model.
            decoder_type,  # The type of the model as str, e.g gpt, gptj, llama.
            dtype,  # the weights data type to export the unquantized layers.
            export_dir,  # The directory where the exported files will be stored.
            inference_tensor_parallel,  # The number of GPUs used in the inference time for tensor parallelism.
            inference_pipeline_parallel,  # The number of GPUs used in the inference time for pipeline parallelism.
        )

If the :meth:`export_tensorrt_llm_checkpoint <modelopt.torch.export.model_config_export.export_tensorrt_llm_checkpoint>` call is successful, the TensorRT-LLM checkpoint will be saved. Otherwise, e.g. the ``decoder_type`` is not supported, a torch state_dict checkpoint will be saved instead.

.. list-table:: Model support matrix for the TensorRT-LLM checkpoint export
   :header-rows: 1

   * - Model / Quantization
     - FP16 / BF16
     - FP8
     - INT8_SQ
     - INT4_AWQ
   * - GPT2
     - Yes
     - Yes
     - Yes
     - No
   * - GPTJ
     - Yes
     - Yes
     - Yes
     - Yes
   * - LLAMA 2
     - Yes
     - Yes
     - Yes
     - Yes
   * - LLAMA 3
     - Yes
     - Yes
     - No
     - Yes
   * - Mistral
     - Yes
     - Yes
     - Yes
     - Yes
   * - Mixtral 8x7B
     - Yes
     - Yes
     - No
     - Yes
   * - Falcon 40B, 180B
     - Yes
     - Yes
     - Yes
     - Yes
   * - Falcon 7B
     - Yes
     - Yes
     - Yes
     - No
   * - MPT 7B, 30B
     - Yes
     - Yes
     - Yes
     - Yes
   * - Baichuan 1, 2
     - Yes
     - Yes
     - Yes
     - Yes
   * - ChatGLM2, 3 6B
     - Yes
     - No
     - No
     - Yes
   * - Bloom
     - Yes
     - Yes
     - Yes
     - Yes
   * - Phi-1, 2, 3
     - Yes
     - Yes
     - Yes
     - Yes
   * - Nemotron 8
     - Yes
     - Yes
     - No
     - Yes
   * - Gemma 2B, 7B
     - Yes
     - Yes
     - No
     - Yes
   * - Recurrent Gemma
     - Yes
     - Yes
     - Yes
     - Yes
   * - StarCoder 2
     - Yes
     - Yes
     - Yes
     - Yes
   * - Qwen-1, 1.5
     - Yes
     - Yes
     - Yes
     - Yes

Convert to TensorRT-LLM
=======================

Once the TensorRT-LLM checkpoint is available, please follow the `TensorRT-LLM build API <https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/legacy/architecture/workflow.md#build-apis>`_ to build and deploy the quantized LLM.
