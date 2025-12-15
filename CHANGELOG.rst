NVIDIA Model Optimizer Changelog (Linux)
========================================

0.41 (2026-01-19)
^^^^^^^^^^^^^^^^^

**New Features**

- Add support for Transformer Engine quantization for Megatron Core models.
- Add automated QDQ placement tool to search optimial QDQ insertion points.

0.40 (2025-12-12)
^^^^^^^^^^^^^^^^^

**Bug Fixes**

- Fix a bug in FastNAS pruning (computer vision models) where the model parameters were sorted twice messing up the ordering.
- Fix Q/DQ/Cast node placements in 'FP32 required' tensors in custom ops in the ONNX quantization workflow.

**New Features**

- Add MoE (e.g. Qwen3-30B-A3B, gpt-oss-20b) pruning support for ``num_moe_experts``, ``moe_ffn_hidden_size`` and ``moe_shared_expert_intermediate_size`` parameters in Minitron pruning (``mcore_minitron``).
- Add ``specdec_bench`` example to benchmark speculative decoding performance. See `examples/specdec_bench/README.md <https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/specdec_bench#speculative-decoding-benchmark>`_ for more details.
- Add FP8/NVFP4 KV cache quantization support for Megatron Core models.
- Add KL Divergence loss based auto_quantize method. See `auto_quantize API docs <https://nvidia.github.io/Model-Optimizer/reference/generated/modelopt.torch.quantization.model_quant.html#modelopt.torch.quantization.model_quant.auto_quantize>`_ for more details.
- Add support for saving and resuming auto_quantize search state. This speeds up the auto_quantize process by skipping the score estimation step if the search state is provided.
- Add flag ``trt_plugins_precision`` in ONNX autocast to indicate custom ops precision. This is similar to the flag already existing in the quantization workflow.
- Add support for PyTorch Geometric quantization.
- Add per tensor and per channel MSE calibrator support.
- Added support for PTQ/QAT checkpoint export and loading for running fakequant evaluation in vLLM. See `examples/vllm_serve/README.md <https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/vllm_serve#load-qatptq-model-and-serve-in-vllm-wip>`_ for more details.

**Documentation**

- Deprecate ``examples/megatron-lm`` in favor of more detailed documentation in `Megatron-LM/examples/post_training/modelopt <https://github.com/NVIDIA/Megatron-LM/tree/main/examples/post_training/modelopt>`_.

**Misc**

- NVIDIA TensorRT Model Optimizer is now officially rebranded as NVIDIA Model Optimizer. GitHub will automatically redirect the old repository path (``NVIDIA/TensorRT-Model-Optimizer``) to the new one (``NVIDIA/Model-Optimizer``). Documentation URL is also changed to `nvidia.github.io/Model-Optimizer <https://nvidia.github.io/Model-Optimizer>`_.
- Bump TensorRT-LLM docker to 1.2.0rc4.
- Bump minimum recommended transformers version to 4.53.
- Replace ONNX simplification package from ``onnxsim`` to ``onnxslim``.

0.39 (2025-11-11)
^^^^^^^^^^^^^^^^^

**Deprecations**

- Deprecated ``modelopt.torch._deploy.utils.get_onnx_bytes`` API. Please use ``modelopt.torch._deploy.utils.get_onnx_bytes_and_metadata`` instead to access the ONNX model bytes with external data. see `examples/onnx_ptq/download_example_onnx.py <https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/onnx_ptq/download_example_onnx.py>`_ for example usage.

**New Features**

- Add flag ``op_types_to_exclude_fp16`` in ONNX quantization to exclude ops from being converted to FP16/BF16. Alternatively, for custom TensorRT ops, this can also be done by indicating ``'fp32'`` precision in ``trt_plugins_precision``.
- Add LoRA mode support for MCore in a new peft submodule: ``modelopt.torch.peft.update_model(model, LORA_CFG)``.
- Support PTQ and fakequant in vLLM for fast evaluation of arbitrary quantization formats. See ``examples/vllm_serve`` for more details.
- Add support for ``nemotron-post-training-dataset-v2`` and ``nemotron-post-training-dataset-v1`` in ``examples/llm_ptq``. Default to a mix of ``cnn_dailymail`` and ``nemotron-post-training-dataset-v2`` (gated dataset accessed using ``HF_TOKEN`` environment variable) if no dataset is specified.
- Allow specifying ``calib_seq`` in ``examples/llm_ptq`` to set the maximum sequence length for calibration.
- Add support for MCore MoE PTQ/QAT/QAD.
- Add support for multi-node PTQ and export with FSDP2 in ``examples/llm_ptq/multinode_ptq.py``. See `examples/llm_ptq/README.md <https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/llm_ptq#multi-node-post-training-quantization-with-fsdp2>`_ for more details.
- Add support for Nemotron Nano VL v1 & v2 models in FP8/NVFP4 PTQ workflow.
- Add flags ``nodes_to_include`` and ``op_types_to_include`` in AutoCast to force-include nodes in low precision, even if they would otherwise be excluded by other rules.
- Add support for ``torch.compile`` and benchmarking in ``examples/diffusers/quantization/diffusion_trt.py``.
- Enabled native Modelopt quantization support for FP8 and NVFP4 formats in SGLang. See `SGLang quantization documentation <https://github.com/sgl-project/sglang/blob/main/docs/advanced_features/quantization.md#using-nvidia-modelopt>`_ for more details.
- Added modelopt quantized checkpoints in vLLM/SGLang CI/CD pipelines (PRs are under review).
- Add support for exporting QLoRA checkpoint fintuned using ModelOpt.
- Update NVFP4 AWQ checkpoint export. It now fuses scaling factors of o_proj and down_proj layers into the model when possible to facilitate deployment.

**Documentation**

- Add general guidelines for Minitron pruning and distillation. See `examples/pruning/README.md <https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/pruning#pruning-guidelines>`_ for more details.
- Added example for exporting QLoRA checkpoint for vLLM deployment. Refer to `examples/llm_qat/README.md <https://github.com/NVIDIA/Model-Optimizer/blob/79ef31bc7269ba4da0cfab446da5b64509cbfcef/examples/llm_qat/README.md#qlora-deployment>`_ for more details

0.37 (2025-10-08)
^^^^^^^^^^^^^^^^^

**Deprecations**

- Deprecated ModelOpt's custom docker images. Please use the PyTorch, TensorRT-LLM or TensorRT docker image directly or refer to the `installation guide <https://nvidia.github.io/Model-Optimizer/getting_started/2_installation.html>`_ for more details.
- Deprecated ``quantize_mode`` argument in ``examples/onnx_ptq/evaluate.py`` to support strongly typing. Use ``engine_precision`` instead.
- Deprecated TRT-LLM's TRT backend in ``examples/llm_ptq`` and ``examples/vlm_ptq``. Tasks ``build`` and ``benchmark`` support are removed and replaced with ``quant``. ``engine_dir`` is replaced with ``checkpoint_dir`` in ``examples/llm_ptq`` and ``examples/vlm_ptq``. For performance evaluation, please use ``trtllm-bench`` directly.
- ``--export_fmt`` flag in ``examples/llm_ptq`` is removed. By default we export to the unified Hugging Face checkpoint format.
- Deprecated ``examples/vlm_eval`` as it depends on the deprecated TRT-LLM's TRT backend.

**New Features**

- ``high_precision_dtype`` default to fp16 in ONNX quantization, i.e. quantized output model weights are now FP16 by default.
- Upgrade TensorRT-LLM dependency to 1.1.0rc2.
- Support Phi-4-multimodal and Qwen2.5-VL quantized HF checkpoint export in ``examples/vlm_ptq``.
- Support storing and restoring Minitron pruning activations and scores for re-pruning without running the forward loop again.
- Add Minitron pruning example for Megatron-LM framework. See `Megatron-LM/examples/post_training/modelopt <https://github.com/NVIDIA/Megatron-LM/tree/main/examples/post_training/modelopt>`_ for more details.

0.35 (2025-09-04)
^^^^^^^^^^^^^^^^^

**Deprecations**

- Deprecate ``torch<2.6`` support.
- Deprecate NeMo 1.0 model support.

**Bug Fixes**

- Fix attention head ranking logic for pruning Megatron Core GPT models.

**New Features**

- ModelOpt now supports PTQ and QAT for GPT-OSS models. See ``examples/gpt_oss`` for end-to-end PTQ/QAT example.
- Add support for QAT with HuggingFace + DeepSpeed. See ``examples/gpt_oss`` for an example.
- Add support for QAT with LoRA. The LoRA adapters can be folded into the base model after QAT and deployed just like a regular PTQ model. See ``examples/gpt_oss`` for an example.
- ModelOpt provides convenient trainers such as :class:`QATTrainer`, :class:`QADTrainer`, :class:`KDTrainer`, :class:`QATSFTTrainer` which inherits from Huggingface trainers.
  ModelOpt trainers can be used as drop in replacement of the corresponding Huggingface trainer. See usage examples in ``examples/gpt_oss``, ``examples/llm_qat`` or ``examples/llm_distill``.
- (Experimental) Add quantization support for custom TensorRT op in ONNX models.
- Add support for Minifinetuning (MFT; https://arxiv.org/abs/2506.15702) self-corrective distillation, which enables training on small datasets with severely mitigated catastrophic forgetting.
- Add tree decoding support for Megatron Eagle models.
- For most VLMs, we now explicitly disable quant on the vision part so we add them to the excluded_modules during HF export.
- Add support for ``mamba_num_heads``, ``mamba_head_dim``, ``hidden_size`` and ``num_layers`` pruning for Megatron Core Mamba or Hybrid Transformer Mamba models in ``mcore_minitron`` (previously ``mcore_gpt_minitron``) mode.
- Add example for QAT/QAD training with `LLaMA Factory <https://github.com/hiyouga/LLaMA-Factory/tree/main>`_. See ``examples/llm_qat/llama_factory`` for more details.
- Upgrade TensorRT-LLM dependency to 1.0.0rc6.
- Add unified HuggingFace model export support for quantized NVFP4 GPT-OSS models.

0.33 (2025-07-14)
^^^^^^^^^^^^^^^^^

**Backward Breaking Changes**

- PyTorch dependencies for ``modelopt.torch`` features are no longer optional and ``pip install nvidia-modelopt`` is now same as ``pip install nvidia-modelopt[torch]``.

**New Features**

- Upgrade TensorRT-LLM dependency to 0.20.
- Add new CNN QAT example to demonstrate how to use ModelOpt for QAT.
- Add support for ONNX models with custom TensorRT ops in Autocast.
- Add quantization aware distillation (QAD) support in ``llm_qat`` example.
- Add support for BF16 in ONNX quantization.
- Add per node calibration support in ONNX quantization.
- ModelOpt now supports quantization of tensor-parallel sharded Huggingface transformer models. This requires ``transformers>=4.52.0``.
- Support quantization of FSDP2 wrapped models and add FSDP2 support in the ``llm_qat`` example.
- Add NeMo 2 Simplified Flow examples for quantization aware training/distillation (QAT/QAD), speculative decoding, pruning & distillation.
- Fix a Qwen3 MOE model export issue.

0.31 (2025-06-04)
^^^^^^^^^^^^^^^^^

**Backward Breaking Changes**

- NeMo and Megatron-LM distributed checkpoint (``torch-dist``) stored with legacy version can no longer be loaded. The remedy is to load the legacy distributed checkpoint with 0.29 and store a ``torch`` checkpoint and resume with 0.31 to convert to a new format. The following changes only apply to storing and resuming distributed checkpoint.
    - ``quantizer_state`` of :class:`TensorQuantizer <modelopt.torch.quantization.nn.modules.TensorQuantizer>` is now stored in ``extra_state`` of :class:`QuantModule <modelopt.torch.quantization.nn.module.QuantModule>` where it used to be stored in the sharded ``modelopt_state``.
    - The dtype and shape of ``amax`` and ``pre_quant_scale`` stored in the distributed checkpoint are now restored. Some dtype and shape are previously changed to make all decoder layers to have homogeneous structure in the checkpoint.
    - Together with megatron.core-0.13, quantized model will store and resume distributed checkpoint in a heterogenous format.
- auto_quantize API now accepts a list of quantization config dicts as the list of quantization choices.
    - This API previously accepts a list of strings of quantization format names. It was therefore limited to only pre-defined quantization formats unless through some hacks.
    - With this change, now user can easily use their own custom quantization formats for auto_quantize.
    - In addition, the ``quantization_formats`` now exclude ``None`` (indicating "do not quantize") as a valid format because the auto_quantize internally always add "do not quantize" as an option anyway.
- Model export config is refactored. The quant config in ``hf_quant_config.json`` is converted and saved to ``config.json``. ``hf_quant_config.json`` will be deprecated soon.


**Deprecations**

- Deprecate ``Python 3.9`` support.

**New Features**

- Upgrade LLM examples to use TensorRT-LLM 0.19.
- Add new model support in the ``llm_ptq`` example: Qwen3 MoE.
- ModelOpt now supports advanced quantization algorithms such as AWQ, SVDQuant and SmoothQuant for cpu-offloaded Huggingface models.
- Add AutoCast tool to convert ONNX models to FP16 or BF16.
- Add ``--low_memory_mode`` flag in the llm_ptq example support to initialize HF models with compressed weights and reduce peak memory of PTQ and quantized checkpoint export.
- Support ``NemotronHForCausalLM``, ``Qwen3ForCausalLM``, ``Qwen3MoeForCausalLM`` Megatron Core model import/export (from/to HuggingFace).

0.29 (2025-05-08)
^^^^^^^^^^^^^^^^^

**Backward Breaking Changes**

- Refactor ``SequentialQuantizer`` to improve its implementation and maintainability while preserving its functionality.

**Deprecations**

- Deprecate ``torch<2.4`` support.

**New Features**

- Upgrade LLM examples to use TensorRT-LLM 0.18.
- Add new model support in the ``llm_ptq`` example: Gemma-3, Llama-Nemotron.
- Add INT8 real quantization support.
- Add an FP8 GEMM per-tensor quantization kernel for real quantization. After PTQ, you can leverage the :meth:`mtq.compress <modelopt.torch.quantization.compress>` API to accelerate evaluation of quantized models.
- Use the shape of Pytorch parameters and buffers of :class:`TensorQuantizer <modelopt.torch.quantization.nn.modules.TensorQuantizer>` to initialize them during restore. This makes quantized model restoring more robust.
- Support adding new custom quantization calibration algorithms. Please refer to :func:`mtq.calibrate <modelopt.torch.quantization.model_quant.calibrate>` or :ref:`custom calibration algorithm <custom_calibration_algorithm>` for more details.
- Add EAGLE3 (``LlamaForCausalLMEagle3``) training and unified ModelOpt checkpoint export support for Megatron-LM.
- Add support for ``--override_shapes`` flag to ONNX quantization.
   - ``--calibration_shapes`` is reserved for the input shapes used for calibration process.
   - ``--override_shapes`` is used to override the input shapes of the model with static shapes.
- Add support for UNet ONNX quantization.
- Enable ``concat_elimination`` pass by default to improve the performance of quantized ONNX models.
- Enable Redundant Cast elimination pass by default in :meth:`moq.quantize <modelopt.onnx.quantization.quantize>`.
- Add new attribute ``parallel_state`` to :class:`DynamicModule <modelopt.torch.opt.dynamic.DynamicModule>` to support distributed parallelism such as data parallel and tensor parallel.
- Add MXFP8, NVFP4 quantized ONNX export support.
- Add new example for torch quantization to ONNX for MXFP8, NVFP4 precision.

0.27 (2025-04-03)
^^^^^^^^^^^^^^^^^

**Deprecations**

- Deprecate real quantization configs, please use :meth:`mtq.compress <modelopt.torch.quantization.compress>` API for model compression after quantization.

**New Features**

- Add new model support in the ``llm_ptq`` example: OpenAI Whisper. Experimental support: Llama4, QwQ, Qwen MOE.
- Add blockwise FP8 quantization support in unified model export.
- Add quantization support to the Transformer Engine Linear module.
- Add support for SVDQuant. Currently, only simulation is available; real deployment (for example, TensorRT deployment) support is coming soon.
- Store ``modelopt_state`` in Megatron Core distributed checkpoint (used in NeMo and Megatron-LM) differently to support distributed checkpoint resume expert-parallel (EP). The legacy ``modelopt_state`` in the distributed checkpoint generated by previous modelopt version can still be loaded in 0.27 and 0.29 but will need to be stored in the new format.
- Add triton-based NVFP4 quantization kernel that delivers approximately 40% performance improvement over the previous implementation.
- Add a new API :meth:`mtq.compress <modelopt.torch.quantization.compress>` for model compression for weights after quantization.
- Add option to simplify ONNX model before quantization is performed.
- Add FP4 KV cache support for unified HF and TensorRT-LLM export.
- Add speculative decoding support to Multi-Token Prediction (MTP) in Megatron Core models.
- (Experimental) Improve support for ONNX models with custom TensorRT op:
   - Add support for ``--calibration_shapes`` flag.
   - Add automatic type and shape tensor propagation for full ORT support with TensorRT EP.

**Known Issues**

- Quantization of T5 models is broken. Please use ``nvidia-modelopt==0.25.0`` with ``transformers<4.50`` meanwhile.

0.25 (2025-03-03)
^^^^^^^^^^^^^^^^^

**Deprecations**

- Deprecate Torch 2.1 support.
- Deprecate ``humaneval`` benchmark in ``llm_eval`` examples. Please use the newly added ``simple_eval`` instead.
- Deprecate ``fp8_naive`` quantization format in ``llm_ptq`` examples. Please use ``fp8`` instead.

**New Features**

- Support fast hadamard transform in :class:`TensorQuantizer <modelopt.torch.quantization.nn.modules.TensorQuantizer>`.
  It can be used for rotation based quantization methods, e.g. QuaRot. Users need to install the package `fast_hadamard_transform <https://github.com/Dao-AILab/fast-hadamard-transform>`_ to use this feature.
- Add affine quantization support for the KV cache, resolving the low accuracy issue in models such as Qwen2.5 and Phi-3/3.5.
- Add FSDP2 support. FSDP2 can now be used for QAT.
- Add `LiveCodeBench <https://livecodebench.github.io/>`_  and `Simple Evals <https://github.com/openai/simple-evals>`_ to the ``llm_eval`` examples.
- Disabled saving modelopt state in unified hf export APIs by default, i.e., added ``save_modelopt_state`` flag in ``export_hf_checkpoint`` API and by default set to False.
- Add FP8 and NVFP4 real quantization support with LLM QLoRA example.
- The :class:`modelopt.deploy.llm.LLM` now support use the :class:`tensorrt_llm._torch.LLM` backend for the quantized HuggingFace checkpoints.
- Add `NVFP4 PTQ example for DeepSeek-R1 <https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/deepseek>`_.
- Add end-to-end `AutoDeploy example for AutoQuant LLM models <https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/llm_autodeploy>`_.

0.23 (2025-01-29)
^^^^^^^^^^^^^^^^^

**Backward Breaking Changes**

- Support TensorRT-LLM to 0.17. Examples (e.g. benchmark task in llm_ptq) may not be fully compatible with TensorRT-LLM 0.15.
- Nvidia Model Optimizer has changed its LICENSE from NVIDIA Proprietary (library wheel) and MIT (examples) to Apache 2.0 in this first full OSS release.
- Deprecate Python 3.8, Torch 2.0, and Cuda 11.x support.
- ONNX Runtime dependency upgraded to 1.20 which no longer supports Python 3.9.
- In the Huggingface examples, the ``trust_remote_code`` is by default set to false and require users to explicitly turning it on with ``--trust_remote_code`` flag.

**New Features**

- Added OCP Microscaling Formats (MX) for fake quantization support, including FP8 (E5M2, E4M3), FP6 (E3M2, E2M3), FP4, INT8.
- Added NVFP4 quantization support for NVIDIA Blackwell GPUs along with updated examples.
- Allows export lm_head quantized TensorRT-LLM checkpoint. Quantize lm_head could benefit smaller sized models at a potential cost of additional accuracy loss.
- TensorRT-LLM now supports Moe FP8 and w4a8_awq inference on SM89 (Ada) GPUs.
- New models support in the ``llm_ptq`` example: Llama 3.3, Phi 4.
- Added Minitron pruning support for NeMo 2.0 GPT models.
- Exclude modules in TensorRT-LLM export configs are now wildcards
- The unified llama3.1 FP8 huggingface checkpoints can be deployed on `SGLang <https://github.com/sgl-project/sglang/pull/2535>`_.

0.21 (2024-12-03)
^^^^^^^^^^^^^^^^^

**Backward Breaking Changes**

- Support TensorRT-LLM to 0.15. Examples (e.g. benchmark task in llm_ptq) may not be fully compatible with TensorRT-LLM 0.14.
- Remove the deprecated arg ``export_npz`` from the :meth:`mt.export.export_tensorrt_llm_checkpoint <modelopt.torch.export.export_tensorrt_llm_checkpoint>` API
- Deprecate :meth:`mt.export.export_to_vllm <modelopt.torch.export.export_to_vllm>` API for :meth:`mt.export.export_hf_checkpoint <modelopt.torch.export.export_hf_checkpoint>`
- Rename decoder type ``gptnext`` to ``gpt`` in ``llm_ptq`` to align with TensorRT-LLM model definition.

**New Features**

- Added new tutorial notebooks for Minitron pruning and distillation in NVIDIA NeMo framework.
- New models support in the ``llm_ptq`` example: Minitron, Phi3.5 MOE.
- New models support in the ``vlm_ptq`` example: Llama3.2(Mllama)
- :meth:`mt.export.export_tensorrt_llm_checkpoint <modelopt.torch.export.export_tensorrt_llm_checkpoint>` and :meth:`mt.export.export_hf_checkpoint <modelopt.torch.export.export_hf_checkpoint>` no longer requires the ``dtype`` arg.
- Added an example to deploy and run quantized fp8 llama3.1 8B instruct model from HuggingFace modelopt model hub on both TensorRT and vLLM.

**Bug Fixes**

- Improve Minitron pruning quality by avoiding possible bf16 overflow in importance calculation and minor change in ``hidden_size`` importance ranking.

**Misc**

- Added deprecation warnings for Python 3.8, torch 2.0, and CUDA 11.x. Support will be dropped in the next release.

0.19 (2024-10-23)
^^^^^^^^^^^^^^^^^

**Backward Breaking Changes**

- Deprecated the summarize task in the ``llm_ptq`` example.
- Deprecated the ``type`` flag in the `huggingface_example.sh <https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/llm_ptq/scripts/huggingface_example.sh>`_
- Deprecated Python plugin support in ONNX.
- Support TensorRT-LLM 0.13. Examples not compatible with TensorRT-LLM 0.12.
- :meth:`mtq.auto_quantize <modelopt.torch.quantization.model_quant.auto_quantize>` API has been updated. The API now
  accepts ``forward_step`` and ``forward_backward_step`` as arguments instead of ``loss_func`` and ``collect_func``.
  Please see the API documentation for more details.

**New Features**

- ModelOpt is compatible for SBSA aarch64 (e.g. GH200) now!
  Except ONNX PTQ with plugins is not supported.
- Add ``effective_bits`` as a constraint for :meth:`mtq.auto_qauntize <modelopt.torch.quantization.model_quant.auto_quantize>`.
- ``lm_evaluation_harness`` is fully integrated to modelopt backed by TensorRT-LLM.
  ``lm_evaluation_harness`` benchmarks are now available in the examples for LLM accuracy evaluation.
- A new ``--perf`` flag is introduced in the ``modelopt_to_tensorrt_llm.py`` example to build engines with max perf.
- Users can choose the execution provider to run the calibration in ONNX quantization.
- Added automatic detection of custom ops in ONNX models using TensorRT plugins.
  This requires the ``tensorrt`` python package to be installed.
- Replaced ``jax`` with ``cupy`` for faster INT4 ONNX quantization.
- :meth:`mtq.auto_quantize <modelopt.torch.quantization.model_quant.auto_quantize>` now supports search based automatic
  quantization for NeMo & MCore models (in addition to HuggingFace models).
- Add ``num_layers`` and ``hidden_size`` pruning support for NeMo / Megatron-core models.


0.17 (2024-09-11)
^^^^^^^^^^^^^^^^^

**Backward Breaking Changes**

- Deprecated ``torch<2.0`` support.
- :meth:`modelopt.torch.utils.dataset_utils.get_dataset_dataloader` now returns a key value pair instead of the tensor.

**New Features**

- New APIs and examples: :mod:`modelopt.torch.prune` for pruning Conv, Linear, and Attention heads for
  NVIDIA Megatron-core GPT-style models (e.g. Llama 3), PyTorch Computer Vision models, and HuggingFace Bert/GPT-J models.
- New API: :mod:`modelopt.torch.distill` for knowledge distillation, along with guides and example.
- New Example: `HF BERT Prune, Distill & Quantize <https://github.com/NVIDIA/Model-Optimizer/blob/main/examples/chained_optimizations>`_
  showcasing how to chain pruning, distillation, and quantization to achieve the best performance on a given model.
- Added INT8/FP8 DQ-only support for ONNX model.
- New API: :mod:`modelopt.torch.speculative` for end-to-end support of Medusa models.
- Added Medusa QAT and End-to-end examples.
- Modelopt now supports automatic save/restore of ``modelopt_state`` with the ``.save_pretrained`` and ``.from_pretrained`` APIs
  from Huggingface libraries, such as ``transformers`` and ``diffusers``. This feature can be enabled by calling
  :meth:`mto.enable_huggingface_checkpointing() <modelopt.torch.opt.plugins.huggingface.enable_huggingface_checkpointing>`.
- ONNX FP8 quantization support with amax calibration.
- TensorRT-LLM dependency upgraded to 0.12.0. Huggingface tokenizer files are now also stored in the engine dir.
- The unified model export API :meth:`modelopt.torch.export.export_hf_checkpoint <modelopt.torch.export.unified_export_hf.export_hf_checkpoint>`
  supports exporting ``fp8`` and ``int4_awq`` quantized checkpoints with packed weights for
  Hugging Face models with namings aligned with its original checkpoints. The exported ``fp8`` checkpoints can be deployed with both TensorRT-LLM and VLLM.
- Add int8 and fp8 quantization support for the FLUX.1-dev model.
- Add a Python-friendly TensorRT inference pipeline for diffusion models.

**Misc**

- Added deprecation warning for :meth:`set_data_parallel_group <modelopt.torch.utils.distributed.set_data_parallel_group>`
  and :meth:`set_tensor_parallel_group <modelopt.torch.utils.distributed.set_tensor_parallel_group>`. These APIs are
  no longer needed for supporting distributed data and tensor parallelism in quantization. They will be removed in
  a future release.


0.15 (2024-07-25)
^^^^^^^^^^^^^^^^^

**Backward Breaking Changes**

- Deprecated :class:`QuantDescriptor <modelopt.torch.quantization.tensor_quant.QuantDescriptor>`.
  Use :class:`QuantizerAttributeConfig <modelopt.torch.quantization.config.QuantizerAttributeConfig>` to
  configure :class:`TensorQuantizer <modelopt.torch.quantization.nn.modules.TensorQuantizer>`.
  :meth:`set_from_attribute_config <modelopt.torch.quantization.nn.modules.TensorQuantizer.set_from_attribute_config>`
  can be used to set the quantizer attributes from the config class or attribute dictionary. This change applies only
  to backend APIs. The change is backward compatible if you are using
  only the :meth:`mtq.quantize <modelopt.torch.quantization.model_quant.quantize>` API.

**New Features**

- Added quantization support for torch ``RNN, LSTM, GRU`` modules. Only available for ``torch>=2.0``.
- ``modelopt.torch.quantization`` now supports module class based quantizer attribute setting for
  :meth:`mtq.quantize <modelopt.torch.quantization.model_quant.quantize>` API.
- Added new LLM PTQ example for DBRX model.
- Added new LLM (Gemma 2) PTQ and TensorRT-LLM checkpoint export support.
- Added new LLM QAT example for NVIDIA NeMo framework.
- TensorRT-LLM dependency upgraded to 0.11.0.
- (Experimental): :meth:`mtq.auto_quantize <modelopt.torch.quantization.model_quant.auto_quantize>` API which quantizes a model
  by searching for the best per-layer quantization formats.
- (Experimental): Added new LLM QLoRA example with NF4 and INT4_AWQ quantization.
- (Experimental): ``modelopt.torch.export`` now supports exporting quantized checkpoints with packed weights for
  Hugging Face models with namings aligned with its original checkpoints.
- (Experimental) Added support for quantization of ONNX models with TensorRT plugin.

**Misc**

- Added deprecation warning for ``torch<2.0``. Support will be dropped in next release.


0.13 (2024-06-14)
^^^^^^^^^^^^^^^^^

**Backward Breaking Changes**

- `PTQ examples <https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/llm_ptq>`_ have been
  upgraded to use TensorRT-LLM 0.10.

**New Features**

- Adding TensorRT-LLM checkpoint export support for Medusa decoding (official ``MedusaModel`` and Megatron Core ``GPTModel``).
- Enable support for mixtral, recurrentgemma, starcoder, qwen in `PTQ examples <https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/llm_ptq>`_.
- Adding TensorRT-LLM checkpoint export and engine building support for sparse models.
- Import scales from TensorRT calibration cache and use them for quantization.
- (Experimental) Enable low GPU memory FP8 calibration for the Hugging Face models when the original model size does not fit into the GPU memory.
- (Experimental) Support exporting FP8 calibrated model to VLLM deployment.
- (Experimental) Python 3.12 support added.


0.11 (2024-05-07)
^^^^^^^^^^^^^^^^^

**Backward Breaking Changes**

- [!!!] The package was renamed from ``ammo`` to ``modelopt``. The new full product
  name is *Nvidia Model Optimizer*. PLEASE CHANGE ALL YOUR REFERENCES FROM ``ammo`` to
  ``modelopt`` including any paths and links!
- Default installation ``pip install nvidia-modelopt`` will now only install minimal core
  dependencies. Following optional dependencies are available depending on the features that are
  being used: ``[deploy], [onnx], [torch], [hf]``. To install all dependencies, use
  ``pip install "nvidia-modelopt[all]"``.
- Deprecated ``inference_gpus`` arg in ``modelopt.torch.export.model_config_export.torch_to_tensorrt_llm_checkpoint``. User should use ``inference_tensor_parallel`` instead.
- Experimental ``modelopt.torch.deploy`` module is now available as ``modelopt.torch._deploy``.

**New Features**

- ``modelopt.torch.sparsity`` now supports sparsity-aware training (SAT). Both SAT and post-training
  sparsification supports chaining with other modes, e.g. SAT + QAT.
- ``modelopt.torch.quantization`` natively support distributed data and tensor parallelism while estimating quantization parameters.
  The data and tensor parallel groups needs to be registered with ``modelopt.torch.utils.distributed.set_data_parallel_group`` and ``modelopt.torch.utils.distributed.set_tensor_parallel_group`` APIs.
  By default, the data parallel group is set as the default distributed group and the tensor parallel group is disabled.
- ``modelopt.torch.opt`` now supports chaining multiple optimization techniques that each require
  modifications to the same model, e.g., you can now sparsify and quantize a model at the same time.
- ``modelopt.onnx.quantization`` supports FLOAT8 quantization format with Distribution calibration algorithm.
- Native support of ``modelopt.torch.opt`` with FSDP (Fully Sharded Data Parallel) for ``torch>=2.1``. This includes
  sparsity, quantization, and any other model modification & optimization.
- Added FP8 ONNX quantization support in ``modelopt.onnx.quantization``.
- Added Windows (``win_amd64``) support for ModelOpt released wheels. Currently supported for ``modelopt.onnx`` submodule only.

**Bug Fixes**

- Fixed the compatibility issue of ``modelopt.torch.sparsity`` with FSDP.
- Fixed an issue in dynamic dim handling in ``modelopt.onnx.quantization`` with random calibration data.
- Fixed graph node naming issue after opset conversion operation.
- Fixed an issue in negative dim handling like dynamic dim in ``modelopt.onnx.quantization`` with random calibration data.
- Fixed allowing to accept ``.pb`` file for input file.
- Fixed copy extra data to tmp folder issue for ONNX PTQ.
