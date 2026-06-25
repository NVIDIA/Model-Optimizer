Changelog
=========

0.46 (2026-08-xx)
^^^^^^^^^^^^^^^^^

**Backward Breaking Changes**

- Remove the ``examples/diffusers/eval`` image-quality evaluation example (ImageReward / CLIP-IQA / CLIP metrics) and its references in ``examples/diffusers/README.md``. The example was deprecated in 0.45 and is no longer maintained.

**Deprecations**

- Consolidated ``examples/vlm_ptq`` into ``examples/llm_ptq``. Vision-language model PTQ now shares the ``hf_ptq.py`` entry point and ``scripts/huggingface_example.sh``; pass ``--vlm`` to run the TensorRT-LLM multimodal quickstart smoke test. The ``examples/vlm_ptq/scripts/huggingface_example.sh`` entry point is deprecated: it now prints a warning and forwards to the ``llm_ptq`` script with ``--vlm``, and will be removed in a future release. See `examples/llm_ptq/README.md <https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/llm_ptq#vlm-quantization>`__.
- Dropped VILA / NVILA vision-language model support in ``examples/llm_ptq``. VILA's modeling code requires ``transformers<=4.50.0``, which conflicts with ModelOpt's minimum supported ``transformers`` version. The VILA-specific bootstrap (repo clone, ``requirements-vila.txt``) and loading paths in ``example_utils.py`` have been removed.

**New Features**

- Add the **D-PACE** loss objective for DFlash speculative-decoding training (`arXiv:2605.18810 <https://arxiv.org/abs/2605.18810>`_) and make it the default (``dflash_loss_objective: dpace``). It replaces the static exponential position decay with dynamic, confidence-derived per-position weights that adapt to whichever block positions currently limit acceptance. Smoothing is controlled by ``dflash_dpace_alpha`` (default 0.5); set ``dflash_loss_objective: decay`` to restore the previous static schedule. Training-only and detached from the gradient (no architecture or inference change).
- Add the ``day0-release`` agent skill (``.agents/skills/day0-release/``), a deterministic end-to-end driver that chains the PTQ → evaluation → comparison skills (the evaluation stage deploys the checkpoint itself) with an enforced gate after each stage and returns a publish decision (ACCEPT / REGRESSION / ANOMALOUS / INFEASIBLE). Ships three GPU-free, unit-tested gate scripts (``gate_ptq.py``, ``gate_run.py``, ``gate_compare.py``) that validate checkpoint coverage, evaluation-run completeness, and baseline-vs-candidate accuracy threshold. v1 reports and stops on regression; the recipe-search loop is deferred.
- Add **streaming** speculative-decoding training (EAGLE3 / DFlash): the draft trains on base-model hidden states produced on the fly by a co-located ``vllm serve`` (no disk dump), moved trainer-side over NIXL RDMA, scaling to multi-node (dedicated serve replicas + DDP trainers). New launcher examples for NVFP4 Kimi-K2.5 / K2.6 on GB200/aarch64 under ``tools/launcher/examples/moonshotai/``.
- Add a fused Triton fast path for ``local_hessian`` NVFP4 weight-scale search (the Hessian-weighted FP8-E4M3 scale sweep). For each NVFP4 block it minimizes ``dwᵀ H dw`` over the 126 candidate scales using the per-cin-block local Hessian on tensor cores, replacing the per-weight Python reference sweep — roughly **34x** faster on a single 8192x4096 weight and bit-exact with the reference for fp32/fp16 weights. Used automatically during ``local_hessian`` calibration for both dense and fused-MoE expert weights; falls back to the reference sweep on CPU, when Triton is unavailable, or via ``MODELOPT_NVFP4_TRITON_SWEEP=0``.
- Add Minitron pruning support for Megatron-Core models with the following new attention and MoE variants. For these, only ``hidden_size`` is pruned (alongside the usual ``ffn_hidden_size`` / ``num_layers`` / MoE dimensions); the variant-internal dimensions noted below are not pruned:

  - **GatedDeltaNet** (linear attention) and **gated attention** (``attention_output_gate``), such as Qwen3.5 (hybrid GatedDeltaNet + gated-attention) language models, including MoE variants — attention / linear-attention heads are not pruned.
  - **Multi-Latent Attention (MLA)**, such as DeepSeek — MLA latent ranks are not pruned.
  - **Latent MoE**, such as Nemotron-3-Super — ``hidden_size`` pruning resizes the latent projections while the MoE latent dim itself is not pruned.
- Add dLLM (tied-weight PTQ and HF-checkpoint export) support for diffusion-based encoder-decoder LLMs (e.g. DiffusionGemma) whose encoder/decoder stacks share parameters via HF ``_tied_weights_keys``.

  - **Deduplicate the modules shared at source** in the quantized export step: ``_export_quantized_weight`` and ``_export_fused_experts`` now alias bit-identical packed ``weight`` / ``weight_scale`` / ``weight_scale_2`` buffers across modules sharing a source weight ``data_ptr()`` so the downstream ``postprocess_state_dict`` dedup catches them (~42% storage reduction on ``nvfp4_experts_only`` for tied 26B MoE checkpoints).
  - New ``sync_tied_input_amax`` helper max-merges per-side ``input_quantizer.amax`` across tied modules before export so single-backbone consumers that load one ``input_scale`` per parameter don't clip either side.
  - The exported state_dict is also **reordered (decoder keys win instead of encoder)** so canonical-side keys per HF's ``_tied_weights_keys`` declaration win the data_ptr dedup; gated to the DiffusionGemma model class in ``_reorder_canonical_first``, no-op for every other model.
  - New DiffusionGemma model-specific recipe under ``modelopt_recipes/huggingface/diffusion_gemma/ptq/`` (``nvfp4_experts_only.yaml`` + its ``disabled_quantizers.yaml`` unit) adds the ``*self_conditioning*`` exclude on top of the standard default, leaving the shared ``default_disabled_quantizers`` unit clean for non-diffusion models — pattern matches the existing ``phi4mm`` / ``nemotron_vl`` model-specific recipes.
  - ``hf_ptq.py`` also unwraps ``ModelOutput`` dataclasses from ``.generate()`` so the preview decode works on diffusion models. Non-tied models see no behavioral change.

**Bug Fixes**

- Fix ``NotImplementedError: "max_all_cuda" not implemented for 'Float8_e4m3fn'`` during quantization calibration of models with natively FP8 (``float8_e4m3fn`` / ``float8_e5m2``) weights, such as DeepSeek-V3. FP8 dtypes implement no reduction (``max``/``amax``), ``abs``, or elementwise ``maximum`` kernels, so ``reduce_amax`` now upcasts FP8 inputs to the default float dtype before reducing; the upcast is lossless and only affects the FP8 path.

0.45 (2026-07-02)
^^^^^^^^^^^^^^^^^

**New Features**

*Quantization*

- Add NVFP4 W4A16 weight-only quantization (``w4a16_nvfp4``): FP4 weights with group_size=16, BF16 activations, no calibration forward pass required. Use ``mtq.W4A16_NVFP4_CFG`` or ``--qformat w4a16_nvfp4`` in ``hf_ptq.py``. vLLM deployment support is in progress.
- Add ``--cast_mxfp4_to_nvfp4`` flag to ``examples/llm_ptq/hf_ptq.py`` for closed-form, bit-exact MXFP4 → NVFP4 weight conversion. Supports the GPT-OSS family (``openai/gpt-oss-20b``, ``openai/gpt-oss-120b``). See `examples/llm_ptq/README.md <https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/llm_ptq#mxfp4--nvfp4-cast-for-gpt-oss>`__ for usage.
- Add ``--cast_mxfp4_to_nvfp4`` flag to ``examples/deepseek/deepseek_v4/quantize_to_nvfp4.py`` for closed-form, bit-exact MXFP4 → NVFP4 conversion of DeepSeek V4 routed-expert weights (mirrors the GPT-OSS cast; w1/w3 share one per-tensor ``scale_2`` for the fused GEMM1). Activation ``input_scale`` still comes from ``--amax_path`` calibration.
- DeepSeek PTQ (``examples/deepseek/ptq.py``) now defaults to native top-k calibration with post-hoc per-layer peer-max sync of expert ``input_quantizer.amax``; the all-experts path is preserved behind ``--calib_all_experts``.
- Add active-MoE cost accounting for ``mtq.auto_quantize`` effective-bits search. Set ``constraints={"effective_bits": ..., "cost_model": "active_moe", "cost": {"active_moe_expert_ratio": ...}}`` to weight routed MoE expert costs by active experts per token while keeping shared experts fully counted. The ``hf_ptq.py`` AutoQuant path exposes this via ``--auto_quantize_cost_model active_moe`` and ``--auto_quantize_active_moe_expert_ratio``.
- Add quantized ``nn.Embedding`` support. ``nn.Embedding`` is now registered in ``QuantModuleRegistry`` and exposes ``weight_quantizer`` (embedding table), ``output_quantizer`` (lookup activations), and a permanently disabled ``input_quantizer`` placeholder — embedding inputs are integer indices and cannot be fake-quantized, so direct ``enable*()`` calls raise. ``export_hf_checkpoint`` packs quantized embedding weights alongside Linear layers. Embedding quantizers are opt-in (``parent_class: nn.Embedding`` disabled by default).
- Add composable ``$import`` system for recipe YAML configs, enabling reusable config snippets referenced via ``{$import: name}`` markers. All built-in PTQ recipes converted to use imports with shared snippets under ``modelopt_recipes/configs/`` (numeric formats, quant_cfg building blocks, presets). See :ref:`composable-imports`.
- The PTQ example scripts ``examples/llm_ptq/hf_ptq.py``, ``examples/llm_ptq/multinode_ptq.py`` and ``examples/megatron_bridge/quantize.py`` now derive their ``--qformat`` / ``--kv_cache_qformat`` (``--quant_cfg`` / ``--kv_cache_quant`` for Megatron-Bridge) CLI vocabularies by discovering the YAML presets under ``modelopt_recipes/configs/ptq/presets/{model,kv}/`` rather than carrying hardcoded ``QUANT_CFG_CHOICES`` / ``KV_QUANT_CFG_CHOICES`` tables. The discovery helper, alias table and ready-built ``QUANT_CFG_CHOICES`` / ``KV_QUANT_CFG_CHOICES`` mappings now live in ``modelopt.recipe.presets`` and are shared by all three scripts. Presets are loaded eagerly into a plain dict at import. Adding a new preset YAML makes it available on the CLI of all three with no script change — note this means each script now accepts every preset under those directories, not just a previously curated subset. All previously-supported short names (``int8_sq``, ``nvfp4_awq``, ``fp8_pb_wo``, ``nvfp4_mse``, ``w4a8_awq``, ``nvfp4_local_hessian``, ``fp8_pc_pt``, ``int8_wo``) keep working via a small deprecation alias table; new formats should be exposed as preset YAMLs (or, longer term, as full ``--recipe`` recipes).
- Add ``configs/ptq/presets/kv/fp8_cast.yaml`` and ``configs/ptq/presets/kv/nvfp4_cast.yaml``, promoting ``fp8_cast`` / ``nvfp4_cast`` to first-class KV presets composed from the existing ``kv_fp8_cast`` / ``kv_nvfp4_cast`` unit fragments. The previous runtime ``use_constant_amax`` post-edit in ``hf_ptq.py`` is removed; ``use_constant_amax: true`` now lives in the YAML and is therefore authoritative. **Custom (out-of-tree) recipes that target a cast KV format must set ``use_constant_amax: true`` themselves on the ``[kv]_bmm_quantizer`` config** — in-tree recipes already do via the ``kv_*_cast`` units.
- Add FP8 KV-cache cast variants for the partial-NVFP4 and weight-only general PTQ recipes: ``general/ptq/nvfp4_mlp_only-kv_fp8_cast``, ``general/ptq/nvfp4_experts_only-kv_fp8_cast``, ``general/ptq/nvfp4_omlp_only-kv_fp8_cast``, and ``general/ptq/nvfp4_weight_only-kv_fp8_cast``. These compose the same model-quant configs as their ``-kv_fp8`` siblings with the ``kv_fp8_cast`` unit (constant-amax FP8 KV cache, no KV calibration forward pass).
- Add Nemotron-3-Super-120B-A12B PTQ recipes ``modelopt_recipes/huggingface/models/nvidia/Nemotron-3-Super-120B-A12B/ptq/nvfp4-mse.yaml`` (MSE-mixed) and ``nvfp4-max-calib.yaml`` (max-calib mixed): NVFP4 W4A4 routed experts + FP8 per-tensor shared experts / Mamba in/out_proj + FP8 KV cache.
- Group layerwise calibration options under a nested ``LayerwiseConfig`` and add two knobs: ``get_qdq_activations_from_prev_layer`` (correct GPTQ-Hessian vs max-calib activation semantics — defaults to True for GPTQ, False for max/mse/local_hessian) and ``save_every`` (gate per-window ``next_inputs.pt`` activation-cache writes). Legacy bool ``layerwise`` and flat ``layerwise_checkpoint_dir`` keys still work; the bool form emits a ``DeprecationWarning``.
- Add ``examples/alpamayo`` showing FP8, NVFP4, and AutoQuantize (mixed-precision) quantization of the Alpamayo (formerly Alpamayo-R1) ~10B vision-language-action model, with a joint VLM + diffusion calibration loop and both fake-quant and ``--real-quant`` packed-checkpoint export. See `examples/alpamayo/README.md <https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/alpamayo>`_ for details.
- Refactor ``llm_qat`` example with unified YAML-based configuration and flexible dataset blending.
  ``ModelOptArgParser`` adds ``--config`` YAML support with CLI overrides and auto-generates ``ARGUMENTS.md`` from dataclass definitions.
  Dataset blending (``configs/dataset/blend.yaml``) supports HuggingFace datasets, local JSON/JSONL/Parquet files, and weighted multi-source blends.
  The legacy FSDP1 accelerate config is removed; ``llm_qat`` now documents FSDP2, DeepSpeed, and DDP backends.

*Megatron Framework (M-LM / M-Bridge)*

- Add quantization examples for the Megatron-Bridge framework (``examples/megatron_bridge/``): post-training quantization (`quantize.py <https://github.com/NVIDIA/Model-Optimizer/blob/main/examples/megatron_bridge/quantize.py>`_ calibrates an HF model via ``--quant_cfg`` alias / full config name or a ``--recipe`` YAML, with optional KV-cache quant, weight-only, compression, and MoE expert-ratio calibration, and saves a Megatron checkpoint with tensor / pipeline / expert parallelism), export to a deployable HuggingFace (unified) checkpoint for TensorRT-LLM / vLLM / SGLang (`export.py <https://github.com/NVIDIA/Model-Optimizer/blob/main/examples/megatron_bridge/export.py>`_), and Quantization Aware Distillation (extend existing `distill.py <https://github.com/NVIDIA/Model-Optimizer/blob/main/examples/megatron_bridge/distill.py>`_). See `examples/megatron_bridge/README.md <https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/megatron_bridge>`_ for details.
- Add Megatron Core export/import mapping for Qwen3-VL (``Qwen3VLForConditionalGeneration``) vision-language models. The mapping handles the ``model.language_model.`` weight prefix used by Qwen3-VL.
- Add shared Megatron-Core calibration forward loop: ``modelopt.torch.utils.plugins.megatron_calibration.get_megatron_calibration_forward_loop`` produces the ``forward_loop`` callable expected by ``mtq.quantize`` / ``mtp.prune``. Replaces the bespoke calibration loops in Megatron-LM and Megatron-Bridge for quantization and pruning with a single canonical implementation.
- Support Megatron-Core checkpoint restore and export for MSE ``NVFP4StaticQuantizer``.
- Add mixed-precision FP8 + NVFP4 export for Megatron-Core: per-layer ``quant_algo`` recorded under ``quantized_layers`` in ``hf_quant_config.json``, PP-aware ``kv_cache_dtype`` gather, fused-QKV exclude split into per-HF-name ``q/k/v_proj`` entries.
- Add support for ``active_params`` (for MoE models) and ``memory_mb`` constraints in Minitron pruning on top of existing ``params`` constraint. You can also provide multiple constraints. See `examples/pruning/README.md <https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/pruning>`_ for more details. The underlying utility functions ``mcore_param_count``, ``mcore_memory_footprint_mb``, and ``print_mcore_model_stats`` in ``modelopt.torch.nas.plugins.megatron_model_stats`` are also available for standalone use to compute parameter counts and memory footprints (weights + KV-cache + Mamba state) for any Megatron-Core model.
- Add Minitron pruning support for Megatron-Bridge Gemma3 models.
- Add end-to-end optimization tutorial for Minitron pruning + two-phase distillation (80B @ 8K + 20B @ 32K long-context = 100B tokens) + FP8 PTQ + vLLM deployment for Nemotron-3-Nano-30B-A3B-BF16 (MoE + Mamba-Transformer hybrid) → Pruned 22B/A3.0B active params, along with data blend preparation steps (with tool-calling data) and detailed pruning / data-blend / long-context ablations. See `examples/megatron_bridge/tutorials/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/README.md <https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/megatron_bridge/tutorials/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/>`_ for details.

*Datasets & Calibration*

- Add ``DATASET_COMBOS`` to ``modelopt.torch.utils.dataset_utils`` — single ``--dataset`` tokens that fan out to multiple registered datasets; per-entry ``num_samples`` is split evenly across the members. Initial combos: ``cnn_nemotron_v2_mix`` (``cnn_dailymail`` + ``nemotron-post-training-dataset-v2``, used by ``hf_ptq.py`` when no ``--dataset`` is provided) and ``nemotron-post-training-v3`` (the seven ``nvidia/Nemotron-*`` SFT datasets added in #1498, mirroring the `nemotron-post-training-v3 collection <https://huggingface.co/collections/nvidia/nemotron-post-training-v3>`_). Combo names are listed by ``get_supported_datasets()`` and surfaced in ``--dataset`` help. ``get_dataset_dataloader`` rejects inputs that mix a combo with one of its member datasets (e.g. ``cnn_dailymail,cnn_nemotron_v2_mix``) to avoid double-sampling, and ``get_dataset_samples`` rejects combo names so callers route through the dataloader. ``hf_ptq.py`` default ``--calib_size`` is bumped from ``512`` to ``1024`` so the total calibration sample count under the new default combo matches the previous two-dataset fallback.
- The ``nemotron-sft-agentic-v2`` registered dataset (added in #1498) now uses only the ``search`` split. The previously configured ``interactive_agent`` and ``tool_calling`` splits contain content-level defects (heterogeneous schema and a malformed JSON row, respectively) that cause pyarrow's streaming JSON reader to fail deterministically.
- Add ``pack=True`` mode to ``get_dataset_dataloader`` (Megatron-LM pretraining-style global-stream document packing): all raw samples concatenated EOS-separated into one token stream, sliced into uniform ``max_sample_length`` rows. Used by the shared megatron calibration loop.

*Misc*

- Add offline DFlash speculative decoding training. Train the draft module from pre-computed base-model hidden states dumped by ``examples/speculative_decoding/collect_hidden_states/compute_hidden_states_hf.py``; base-model transformer layers are deleted after conversion to save memory. Controlled by the auto-derived ``dflash_offline`` flag on ``DFlashConfig`` (derived from ``data_args.offline_data_path``). The dump scripts now share ``collect_hidden_states/common.py`` for aux-layer selection (``--aux-layers eagle|dflash|<list>``) and optional assistant-token ``loss_mask`` for answer-only-loss training.
- Add ``mtsa.config.SKIP_SOFTMAX_TRITON_CALIB`` for skip-softmax attention-sparsity calibration through the fused Triton ``attention_calibrate`` kernel (HF ``modelopt_triton`` backend), measuring multi-threshold tile-skip statistics the way the Triton inference kernel actually skips tiles for both prefill and decode. Exposed as ``--sparse_attn_cfg skip_softmax_triton_calib`` in ``examples/llm_sparsity/attention_sparsity/hf_sa.py`` (with a new ``--calib_data_dir`` flag for RULER calibration data).
- Add DMD2 distillation for few-step diffusion models in ``examples/diffusers/fastgen/``: distill Qwen-Image into a 4/8-step student via Distribution Matching Distillation. See `examples/diffusers/fastgen/README.md <https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/diffusers/fastgen>`_ for details.
- Make ``.agents/skills/`` the canonical location for agent skills; agent-specific directories (``.claude/skills/``, etc.) are now relative symlinks into ``.agents/``, so one skill suite serves multiple coding agents (Claude Code, Codex). See ``.agents/README.md``.
- Extend Claude Code agent skills for PTQ, deployment, evaluation, monitoring, and baseline-vs-quantized result comparison. Adds evaluation task references for additional benchmarks, stronger PTQ checkpoint validation gates, and session-scoped workspace/job tracking.
- Add SLURM Quality of Service (QoS) support to the ModelOpt launcher. Users can set QoS via ``slurm_config.qos`` or ``SLURM_QOS`` and the value is forwarded to ``nemo_run.SlurmExecutor``.

**Backward Breaking Changes**

- ``KDTrainer`` / ``QADTrainer`` evaluation now reports KD as the primary
  ``eval_loss`` and CE as ``eval_ce_loss``; the previous secondary
  ``eval_kd_loss`` metric is removed.
- Reorganize custom CUDA / Triton kernels under ``modelopt.torch.kernels`` into ``common/attention``, ``quantization/{conv,gemm}``, and ``sparsity/attention``. High-level APIs (``mtq.quantize``, ``mtsa.sparsify``, etc.) are unchanged, but **any code importing directly from the kernel subpackages must be updated**: there is no backwards-compatibility shim; the old import paths will raise ``ImportError`` / ``ModuleNotFoundError``. Migration table:

  - ``from modelopt.torch.kernels import IS_AVAILABLE, attention, attention_calibrate, register_triton_attention`` → ``from modelopt.torch.kernels.common.attention import ...``
  - ``from modelopt.torch.kernels.triton_fa import ...`` → ``from modelopt.torch.kernels.common.attention.triton_fa import ...``
  - ``from modelopt.torch.kernels.hf_triton_attention import ...`` → ``from modelopt.torch.kernels.common.attention.hf_triton_attention import ...``
  - ``from modelopt.torch.quantization.triton import ...`` → ``from modelopt.torch.kernels.quantization.gemm import ...``
  - ``from modelopt.torch.quantization.src.conv.implicit_gemm_cuda import ...`` → ``from modelopt.torch.kernels.quantization.conv.implicit_gemm_cuda import ...``
  - ``from modelopt.torch.sparsity.attention_sparsity.kernels import ...`` → ``from modelopt.torch.kernels.sparsity.attention import ...``

- Deprecated GradNAS pruning algorithm as it is not actively maintained and supports very limited and old models. It is recommended to use Minitron or Puzzletron pruning for LLM models. Also deprecates related ``examples/chained_optimizations`` directory.
- Model-specific PTQ ``quant_cfg`` adjustments previously hardcoded in ``examples/llm_ptq/`` (``build_quant_cfg`` / ``mono_quantize``) for gemma, mpt, phi4mm, and Nemotron VL are now opt-in **model-specific recipes** under ``modelopt_recipes/huggingface/<model_type>/ptq/``. Any adjustment specific to a model type or instance must live in that model's recipe; the bare ``--qformat`` path produces only the generic numerics. Pass ``--recipe huggingface/<model_type>/ptq/<recipe>`` to apply the model's recipe. Covers gemma/mpt ``w4a8_awq`` (``awq_lite`` ``alpha_step=1``), gemma ``int8_sq`` (SmoothQuant ``alpha=0.5``), phi4mm speech/audio/image/vision exclusions, and Nemotron VL vision-branch exclusions. All shipped recipes also enable FP8 KV-cache cast. MTP dynamic layer exclusion and ``is_nemotron_vl`` detection remain in Python.
- The Step3.5-Flash recipe moved from ``modelopt_recipes/models/Step3.5-Flash/nvfp4-mlp-only.yaml`` (0.44) to ``modelopt_recipes/huggingface/step3p5/Step3.5-Flash/ptq/nvfp4-mlp-only.yaml`` to match the ``huggingface/<model_type>/ptq/`` layout convention. Update ``--recipe`` paths accordingly.

**Deprecations**

- Deprecate the public ``QuantizationArgumentsWithConfig`` name in ``modelopt.torch.quantization.plugins.transformers_trainer``; it now aliases ``QuantizationArguments`` and will be removed in a future release.
- Deprecate ``examples/llm_autodeploy``. The AutoQuant + TensorRT-LLM AutoDeploy
  workflow it demonstrates will be removed in a future release; use TensorRT-LLM's
  `AutoDeploy <https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/auto_deploy>`_
  directly together with ModelOpt PTQ in ``examples/llm_ptq``.

**Bug Fixes**

- Support non-gated fused MoE experts in unified HuggingFace export. Nemotron-H MoE models (transformers 5.x ``NemotronHExperts``) store experts as fused 3-D ``up_proj`` / ``down_proj`` parameters with no ``gate_up_proj``; the fused-experts detection previously keyed on ``gate_up_proj``, so these were never wrapped as ``_QuantFusedExperts`` and export raised ``NotImplementedError: MoE model with experts type 'NemotronHExperts' is not supported``. The fused-experts path now also recognizes the non-gated layout (new ``_QuantNonGatedFusedExperts``) and exports a single ``up_proj`` per expert; the gated path is unchanged.
- In Megatron-Core only do EP amax sync for routed expert weights if ``sync_expert_weight_amax=True``. Previously EP amax sync would sync routed expert weights across EP ranks even when ``sync_expert_weight_amax`` was False.
- Fix Megatron-Core HF importer to load fused ``TELayerNormColumnParallelLinear.layer_norm_weight`` from HF for GPT-family models (Qwen3 etc.) under ``--export-default-te-spec``. Importer now prefers per-context keys ``fused_input_layernorm`` / ``fused_pre_mlp_layernorm`` (fallback ``fused_norm`` for Nemotron-H backward compatibility); ``mcore_qwen.py`` provides the new rules. Without this fix, post-prune MMLU sat at chance.
- Fix ONNX AutoCast ``keep_io_types=True`` sanity-check failure (``Unexpected type in I/O tensor ...``) when a network input/output is an empty tensor (a dimension of size 0). Such tensors were "fake-cast" (retyped in place) to the low precision type; because the value-info map aliases the ``graph.input``/``graph.output`` ``ValueInfoProto``, this silently changed the model's I/O type. AutoCast now inserts a real ``Cast`` for protected I/O tensors instead.
- Fix INT8 entropy calibration of fp16 ONNX models raising ``ValueError: Too many bins for data range`` on numpy >= 2.0. ``_collect_value`` in ``modelopt.onnx.quantization.ort_patching`` now casts the histogram range endpoints to Python float so bin edges are computed in float64, instead of inheriting the fp16 dtype of an activation tensor with a small range (which collapsed the 128-bin linspace under NEP-50 promotion).
- Fix the GPT-OSS MXFP4 → NVFP4 PTQ path in ``examples/llm_ptq/hf_ptq.py`` (used with ``--cast_mxfp4_to_nvfp4``). ``get_model`` now loads native MXFP4 checkpoints (``openai/gpt-oss-*``) dequantized to BF16 ``GptOssExperts`` via ``Mxfp4Config(dequantize=True)`` on a sequential device map. This fixes a CUDA illegal-memory access during the multi-GPU dequant load and the ``NotImplementedError`` for experts type ``Mxfp4GptOssExperts`` during unified HF export (the packed-kernel experts wrapper, used when the optional ``kernels`` package is installed, is unsupported by export); ``kernels`` is no longer required. The ``--cast_mxfp4_to_nvfp4`` step now also resolves a HF Hub ID ``--pyt_ckpt_path`` to its local snapshot directory instead of failing with ``FileNotFoundError``.
- Fix ``_QuantGptOssExperts`` / ``_QuantLlama4TextExperts`` static-block NVFP4 weight calibration raising ``ValueError: Input shape has changed`` during the calibration forward. These experts quantize their weights transposed (``_transposed_quantize``); ``iter_weights_for_calibration`` now yields the same transposed view so weight-only calibration and the forward agree on the block-quant shape (and the export ``_amax`` orientation).
- Fix unified HF checkpoint export for Llama4 MoE models. The uncalibrated-experts input-quantizer ``amax`` fallback in ``_export_transformers_checkpoint`` special-cased only ``QuantGptOssExperts``; ``QuantLlama4TextExperts`` uses the same fused ``gate_up_proj`` / ``down_proj`` layout and is now handled by the same branch, fixing the export failure.

0.44 (2026-05-14)
^^^^^^^^^^^^^^^^^

**New Features**

- Add model-agnostic `Liger kernel <https://github.com/linkedin/Liger-Kernel>`_ fused loss support in ``ModelOptHFTrainer`` for any HuggingFace causal LM, with distributed param gathering for FSDP2, DeepSpeed ZeRO-3, and DDP. Extends HuggingFace's built-in Liger integration which is limited to `a fixed set of model architectures <https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/transformers/monkey_patch.py>`_, FSDP only, and CrossEntropy loss. ModelOpt additionally supports Liger fused KD loss (JSD) for knowledge distillation.
- Add ``ModelOptTrainerArguments`` to ``ModelOptHFTrainer`` with ``--trainable_params``, ``--frozen_params``, ``--lr_config``, and ``--manual_gc`` flags. Add per-parameter learning rate support via YAML config. Saved checkpoints preserve the model's original dtype in ``config.json``.
- Simplify ``KDTrainer`` for HuggingFace knowledge distillation: remove ``mtd.convert()`` class-swap in favor of explicit teacher forwarding with logit-level distillation support.
- Support full Transformer Engine spec for Minitron pruning (``mcore_minitron``). Now we no longer need to use custom ModelOpt spec. Note that this does not affect the usage of the pruning workflow but makes pruning slightly faster and may result in slightly different pruned model because of different kernel and numerics.
- Add end-to-end tutorial for Minitron pruning + distillation + quantization + evaluation + vLLM deployment for Nemotron-Nano-9B-v2 → Pruned 7B along with data blend preparation steps (and ablation study). See `examples/pruning/minitron/README.md <https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/pruning/minitron/>`_ for details.
- Add Puzzletron - a new algorithm for heterogeneous pruning of LLM and VLM models. See `examples/puzzletron/README.md <https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/puzzletron>`_ for more details.
- Added iterator interface using CalibrationDataReader in ONNX quantization workflow.
- Add N:M sparse softmax support to the Triton flash attention kernel (``modelopt.torch.kernels.common.attention.triton_fa``). See `examples/llm_sparsity/attention_sparsity/README.md <https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/llm_sparsity/attention_sparsity>`_ for usage.
- Add skip-softmax skipping to the Triton flash attention kernel for both language models and video diffusion models (``modelopt.torch.kernels.common.attention.triton_fa``). See `examples/llm_sparsity/attention_sparsity/README.md <https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/llm_sparsity/attention_sparsity>`_ and `examples/diffusers/sparsity/ <https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/diffusers/sparsity>`_ for usage.
- Add Video Sparse Attention (VSA) method for video diffusion models (``modelopt.torch.sparsity.attention_sparsity``). VSA uses 3D block tiling with a two-branch architecture for attention speedup.
- Enable PTQ workflow for the Step3.5-Flash MoE model with NVFP4 W4A4 + FP8 KV cache quantization. See `modelopt_recipes/models/Step3.5-Flash/nvfp4-mlp-only.yaml <https://github.com/NVIDIA/Model-Optimizer/blob/main/modelopt_recipes/models/Step3.5-Flash/nvfp4-mlp-only.yaml>`_ for more details.
- Add support for vLLM fakequant reload using ModelOpt state for HF models. See `examples/vllm_serve/README.md <https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/vllm_serve#load-qatptq-model-and-serve-in-vllm-wip>`_ for more details.
- [Early Testing] Add Claude Code PTQ skill (``.claude/skills/ptq/``) for agent-assisted post-training quantization. The skill guides the agent through environment detection, model support checking, format selection, and execution via the launcher or manual SLURM/Docker/bare GPU paths. Includes handling for unlisted models with custom module patching. This feature is in early testing — use with caution.
- [Early Testing] Polish Claude Code evaluation skill (``.claude/skills/evaluation/``) for agent-assisted LLM accuracy benchmarking via NeMo Evaluator Launcher. Adds two companion skills vendored verbatim from `NVIDIA-NeMo/Evaluator <https://github.com/NVIDIA-NeMo/Evaluator>`_: ``launching-evals`` (run/check/debug/analyze NEL evaluations) and ``accessing-mlflow`` (query MLflow runs, compare metrics, fetch artifacts). Re-sync at a pinned upstream SHA via ``.claude/scripts/sync-upstream-skills.sh``. Also adds a shared ``skills/common/credentials.md`` covering HF / NGC / Docker token setup referenced by multiple skills. This feature is in early testing — use with caution.
- Add performant layerwise calibration for large models that don't fit on GPU (e.g. DeepSeek-R1, Kimi-K2). See `modelopt_recipes/general/ptq/nvfp4_experts_only-kv_fp8_layerwise.yaml <https://github.com/NVIDIA/Model-Optimizer/blob/main/modelopt_recipes/general/ptq/nvfp4_experts_only-kv_fp8_layerwise.yaml>`_ for usage. Layerwise calibration also supports PTQ with intermediate progress saving — useful when long PTQ runs get hit with Slurm timeouts. See `modelopt_recipes/general/ptq/nvfp4_default-kv_none-gptq.yaml <https://github.com/NVIDIA/Model-Optimizer/blob/main/modelopt_recipes/general/ptq/nvfp4_default-kv_none-gptq.yaml>`_ for usage.
- Add implicit GEMM CUDA kernel for Conv3D with fused NVFP4 fake quantization (``modelopt.torch.quantization.src.conv``). When NVFP4 quantization is applied to an ``nn.Conv3d`` layer via ModelOpt PTQ, the implicit GEMM path is used automatically instead of cuDNN. Uses BF16 WMMA tensor cores (SM80+) with FP32 accumulation and in-kernel FP4 (E2M1) activation quantization. Grouped convolution (``groups > 1``) falls back to the default cuDNN path. Inference only — training mode falls back to cuDNN with a warning.
- Add FP8 MHA quantization support for vision transformers. Adds an attention-aware ONNX post-processing pass (scale Mul / K-transpose move before Q, Q→DQ insertion on softmax output) in :class:`FP8QuantExporter <modelopt.onnx.export.fp8_exporter.FP8QuantExporter>`, per-instance nested-attention-wrapper skipping in the HF plugin, and ``nn.LayerNorm`` registration in ``QuantModuleRegistry`` so BMM input quantizers and LayerNorm output quantizers defined in FP8_DEFAULT_CFG are honored end-to-end. See `examples/torch_onnx/torch_quant_to_onnx.py <https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/torch_onnx/torch_quant_to_onnx.py>`_ for the general timm-model quantize→ONNX workflow.

**Backward Breaking Changes**

- The ``quant_cfg`` field in quantization configs is now an **ordered list** of ``QuantizerCfgEntry`` dicts instead of a flat dictionary. Each entry specifies a ``quantizer_name`` wildcard, an optional ``parent_class`` filter, a ``cfg`` dict of quantizer attributes, and/or an ``enable`` flag. Entries are applied in list order with later entries overriding earlier ones. The old dict-based format is still accepted and automatically converted via ``normalize_quant_cfg_list()``, but now emits a ``DeprecationWarning``; new code should use the list format. All built-in configs (e.g. ``FP8_DEFAULT_CFG``, ``INT4_AWQ_CFG``, ``NVFP4_DEFAULT_CFG``), examples, and YAML recipes have been updated. See the :ref:`quant-cfg` documentation for the new format reference and migration guide.
- Deprecated Mllama (Llama 3.2 Vision) support in the ``llm_ptq`` and ``vlm_ptq`` examples. The ``model_type == "mllama"`` branches and ``MllamaImageProcessor`` usage have been removed from ``hf_ptq.py`` and ``example_utils.py``. For image-text calibration of VLMs, use ``--calib_with_images`` with a supported VLM (see Nemotron VL section in ``examples/llm_ptq/README.md``).

**Bug Fixes**

- Fix Megatron utility functions for generation (with pipeline parallelism) and ~10x speedup in MMLU score evaluation (by batching prefill passes).
- Fix Minitron pruning (``mcore_minitron``) for MoE models. Importance estimation hooks were incorrectly registered for MoE modules and NAS step was hanging before this.
- Fix TRT support for remote autotuning in ONNX Autotune from 10.16+ to 10.15+ and fix TRT versioning check to the ``trtexec`` version instead of the TRT Python API when using ``trtexec`` backend.
- Exclude MatMul/Gemm nodes with K or N < 16 from ONNX INT8 and FP8 quantization. Such small-dimension GEMMs cannot efficiently use INT8/FP8 Tensor Cores and the added Q/DQ layers cause perf regressions in TensorRT. Honors Gemm ``transB`` when deriving K.
- Fix ``nvfp4_awq`` export ``AssertionError: Modules have different quantization formats`` for MoE models (e.g. Qwen3-30B-A3B) when some experts are not exercised by the calibration data. ``awq_lite`` now applies a neutral all-ones ``pre_quant_scale`` to any expert that ends up disabled (no cache-pass tokens, NaN scales, or no search-pass tokens) so its format remains ``nvfp4_awq``, consistent with the rest of the MoE block. A warning is emitted whenever this fallback fires.

**Misc**

- [Security] Changed the default of ``weights_only`` to ``True`` in ``torch.load`` for secure checkpoint loading. If you need to load a checkpoint that requires unpickling arbitrary objects, first register the class in ``torch.serialization.add_safe_globals([cls])`` before loading. Added :meth:`safe_save <modelopt.torch.utils.serialization.safe_save>` and :meth:`safe_load <modelopt.torch.utils.serialization.safe_load>` API to save and load checkpoints securely.
- Bump minimum required PyTorch version to 2.8.
- [Experimental] Add support for transformers>=5.0, including generic PTQ and unified HF checkpoint export for fused MoE expert modules (Mixtral, Qwen2-MoE, Qwen3-MoE, Qwen3.5-MoE, DeepSeek-V3, Jamba, OLMoE, etc.).
- Improve ``megatron_preprocess_data``: add ``--reasoning_content`` support for Nemotron v3 datasets, eliminate intermediate JSONL for HuggingFace datasets, return output file prefixes from the Python API, add gzip input support (``.jsonl.gz``), add ``--strip_newlines`` flag for plain-text pretraining data, add ``--hf_streaming`` for very large datasets (only consumed rows downloaded), and auto-shuffle when ``--hf_max_samples_per_split`` is set to avoid biased sampling.
- Add installation support for Python 3.14. Only basic unit tests are verified for now. Production usage still defaults to Python 3.12. Python 3.10 support will be dropped in the next release.

0.43 (2026-04-16)
^^^^^^^^^^^^^^^^^

**Bug Fixes**

- ONNX Runtime dependency upgraded to 1.24 to solve missing graph outputs when using the TensorRT Execution Provider.

**Backward Breaking Changes**

- Default ``--kv_cache_qformat`` in ``hf_ptq.py`` changed from ``fp8`` to ``fp8_cast``. Existing scripts that rely on the default will now skip KV cache calibration and use a constant amax instead. To restore the previous calibrated behavior, explicitly pass ``--kv_cache_qformat fp8``.
- Removed KV cache scale clamping (``clamp_(min=1.0)``) in the HF checkpoint export path. Calibrated KV cache scales below 1.0 are now exported as-is. If you observe accuracy degradation with calibrated KV cache (``--kv_cache_qformat fp8`` or ``nvfp4``), consider using the casting methods (``fp8_cast`` or ``nvfp4_cast``) instead.

**New Features**

- Add ``fp8_cast`` and ``nvfp4_cast`` modes for ``--kv_cache_qformat`` in ``hf_ptq.py``. These use a constant amax (FP8 E4M3 max, 448.0) without data-driven calibration, since the downstream engine uses FP8 attention math for both FP8 and NVFP4 quantization. A new ``use_constant_amax`` field in :class:`QuantizerAttributeConfig <modelopt.torch.quantization.config.QuantizerAttributeConfig>` controls this behavior.
- User does not need to manually register MOE modules to cover experts calibration coverage in PTQ workflow.
- ``hf_ptq.py`` now saves the quantization summary and moe expert token count table to the export directory.
- Add ``--moe_calib_experts_ratio`` flag in ``hf_ptq.py`` to specify the ratio of experts to calibrate during forward pass to improve expert coverage during calibration. Default to None (not enabled).
- Add sparse attention optimization for transformer models (``modelopt.torch.sparsity.attention_sparsity``). This reduces computational cost by skipping attention computation. Supports calibration for threshold selection on HuggingFace models. See `examples/llm_sparsity/attention_sparsity/README.md <https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/llm_sparsity/attention_sparsity>`_ for usage.
- Add support for rotating the input before quantization for RHT.
- Add support for advanced weight scale search for NVFP4 quantization and its export path.
- Enable PTQ workflow for Qwen3.5 MoE models.
- Enable PTQ workflow for the Kimi-K2.5 model.
- Add ``nvfp4_omlp_only`` quantization format for NVFP4 quantization. This is similar to ``nvfp4_mlp_only`` but also quantizes the output projection layer in attention.
- Add ``nvfp4_experts_only`` quantization config that targets only MoE routed expert layers (excluding shared) with NVFP4 quantization.
- ``pass_through_bwd`` in the quantization config is now default to True. Please set it to False if you want to use STE with zeroed outlier gradients for potentially better QAT accuracy.
- Add :meth:`compute_quantization_mse <modelopt.torch.quantization.model_quant.compute_quantization_mse>` API to measure per-quantizer mean-squared quantization error, with flexible wildcard and callable filtering.
- **Autotune**: New tool for automated Q/DQ (Quantize/Dequantize) placement optimization for ONNX models. Uses TensorRT latency measurements to choose insertion schemes that minimize inference time. Discovers regions automatically, groups them by structural pattern, and tests multiple Q/DQ schemes per pattern. Supports INT8 and FP8 quantization, pattern cache for warm-start on similar models, checkpoint/resume, and importing patterns from an existing QDQ baseline. CLI: ``python -m modelopt.onnx.quantization.autotune``. See the Autotune guide in the documentation.
- Add ``get_auto_quantize_config`` API to extract a flat quantization config from ``auto_quantize`` search results, enabling re-quantization at different effective bit targets without re-running calibration.
- Improve ``auto_quantize`` checkpoint/resume: calibration state is now saved and restored across runs, avoiding redundant calibration when resuming a search.
- Add support for Nemotron-3 (NemotronHForCausalLM) model quantization and support for NemotronH MoE expert support in ``auto_quantize`` grouping and scoring rules.
- Add support for block-granular RHT for non-power-of-2 dimensions.
- Replace modelopt FP8 QDQ nodes with native ONNX QDQ nodes.

**Deprecations**

- Removed MT-Bench (FastChat) support from ``examples/llm_eval``. The ``run_fastchat.sh`` and ``gen_model_answer.py`` scripts have been deleted, and the ``mtbench`` task has been removed from the ``llm_ptq`` example scripts.
- Remove deprecated NeMo-2.0 Framework references.

**Misc**

- Migrated project metadata from ``setup.py`` to a fully declarative ``pyproject.toml``.
- Enable experimental Python 3.13 wheel support and unit tests in CI/CD.

0.42 (2026-03-10)
^^^^^^^^^^^^^^^^^

**Bug Fixes**

- Fix calibration data generation with multiple samples in the ONNX workflow.

**New Features**

- Add standalone type inference option (``--use_standalone_type_inference``) in ONNX AutoCast as an alternative to ONNX's ``infer_shapes``. This experimental feature performs type-only inference without shape inference, useful as a workaround when shape inference fails or to avoid unnecessary shape inference overhead.
- Add support for Kimi K2 Thinking model quantization from the original int4 checkpoint.
- Add support for ``params`` constraint based automatic neural architecture search in Minitron pruning (``mcore_minitron``) as an alternative to manual pruning (using ``export_config``). See `examples/pruning/README.md <https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/pruning>`_ for more details on its usage.
- New example for Minitron pruning with Megatron-Bridge framework along with advanced pruning usage with new ``params`` constraint based pruning. Also add example for distillation with Megatron-Bridge framework. Check `examples/megatron_bridge/README.md <https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/megatron_bridge>`_ for example scripts.
- Add support for calibration data with multiple samples in ``npz`` format in the ONNX Autocast workflow.
- Add ``--opset`` option to ONNX quantization CLI to specify the target opset version for the quantized model.
- Add support for context parallelism in Eagle speculative decoding for huggingface and megatron core models.
- Add unified Hugging Face export support for diffusers pipelines/components.
- Add LTX-2 and Wan2.2 (T2V) support in the diffusers quantization workflow.
- Add PTQ support for GLM-4.7, including loading MTP layer weights from a separate ``mtp.safetensors`` file and export as-is.
- Add support for image-text data calibration in PTQ for Nemotron VL models.
- Add support for advanced weight scale search for NVFP4 quantization and its export path.
- Add PTQ support for Nemotron Parse.
- Add distillation support for LTX-2. See `examples/diffusers/distillation/README.md <https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/diffusers/distillation>`_ for more details.

0.41 (2026-01-19)
^^^^^^^^^^^^^^^^^

**Bug Fixes**

- Fix Megatron KV Cache quantization checkpoint restore for QAT/QAD (device placement, amax sync across DP/TP, flash_decode compatibility).

**New Features**

- Add support for Transformer Engine quantization for Megatron Core models.
- Add support for Qwen3-Next model quantization.
- Add support for dynamically linked TensorRT plugins in the ONNX quantization workflow.
- Add support for KV Cache Quantization for vLLM FakeQuant PTQ script. See `examples/vllm_serve/README.md <https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/vllm_serve#Calibrate-and-serve-fake-quant-model-in-vLLM>`__ for more details.
- Add support for subgraphs in ONNX autocast.
- Add support for parallel draft heads in Eagle speculative decoding.
- Add support to enable custom emulated quantization backend. See :meth:`register_quant_backend <modelopt.torch.quantization.nn.modules.tensor_quantizer.register_quant_backend>`` for more details. See an example in ``tests/unit/torch/quantization/test_custom_backend.py``.
- Add ``examples/llm_qad`` for QAD training with Megatron-LM.

**Deprecations**

- Deprecate ``num_query_groups`` parameter in Minitron pruning (``mcore_minitron``). You can use ModelOpt 0.40.0 or earlier instead if you need to prune it.

**Backward Breaking Changes**

- Remove ``torchprofile`` as a default dependency from ModelOpt as its used only for flops-based FastNAS pruning (computer vision models). It can be installed separately if needed.

**Windows Support**

- Fix ONNX 1.19 compatibility issues with CuPy during ONNX INT4 AWQ quantization. ONNX 1.19 uses ml_dtypes.int4 instead of numpy.int8 which caused CuPy failures.
- Add support for ONNX Mixed Precision Weight-only quantization using INT4 and INT8 precisions. Refer quantization `example for GenAI LLMs <https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/windows/onnx_ptq/genai_llm>`_.
- Add support for some diffusion models' quantization on Windows. Refer `example script <https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/windows/torch_onnx/diffusers>`_ for details.
- Add `Perplexity <https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/windows/accuracy_benchmark/perplexity_metrics>`_ and `KL-Divergence <https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/windows/accuracy_benchmark/kl_divergence_metrics>`_ accuracy benchmarks.

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

- Add general guidelines for Minitron pruning and distillation. See `pruning guidelines <https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/pruning#pruning-guidelines>`_ for more details.
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

**Windows Support**

- Model Optimizer for Windows now supports `NvTensorRtRtx <https://onnxruntime.ai/docs/execution-providers/TensorRTRTX-ExecutionProvider.html>`_ execution-provider.

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
- Add new attribute ``parallel_state`` to :class:`QuantModule <modelopt.torch.quantization.nn.modules.quant_module.QuantModule>` to support distributed parallelism such as data parallel and tensor parallel.
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

**Windows Support**

- New LLM models like DeepSeek etc. are supported with ONNX INT4 AWQ quantization on Windows. Refer `Windows Support Matrix <https://nvidia.github.io/Model-Optimizer/guides/0_support_matrix.html>`_ for details about supported features and models.
- Model Optimizer for Windows now supports ONNX INT8 and FP8 quantization (W8A8) of SAM2 and Whisper models. Check `example scripts <https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/windows/onnx_ptq>`_ for getting started with quantizing these models.

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

**Windows Support**

- This is the first official release of Model Optimizer for Windows
- **ONNX INT4 Quantization:** :meth:`modelopt.onnx.quantization.quantize_int4 <modelopt.onnx.quantization.int4.quantize>` now supports ONNX INT4 quantization for DirectML and TensorRT* deployment. See :ref:`Support_Matrix` for details about supported features and models.
- **LLM Quantization with Olive:** Enabled LLM quantization through Olive, streamlining model optimization workflows. Refer `Olive example <https://github.com/microsoft/Olive/tree/main/examples/phi3#quantize-models-with-nvidia-Model-Optimizer>`_.
- **DirectML Deployment Guide:** Added DML deployment guide. Refer :ref:`Onnxruntime_Deployment` deployment guide for details.
- **MMLU Benchmark for Accuracy Evaluations:** Introduced `MMLU benchmarking <https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/windows/accuracy_benchmark/README.md>`_ for accuracy evaluation of ONNX models on DirectML (DML).
- **Published quantized ONNX models collection:** Published quantized ONNX models at HuggingFace `NVIDIA collections <https://huggingface.co/collections/nvidia/optimized-onnx-models-for-nvidia-rtx-gpus>`_.


\* *This version includes experimental features such as TensorRT deployment of ONNX INT4 models, PyTorch quantization and sparsity. These are currently unverified on Windows.*


0.17 (2024-09-11)
^^^^^^^^^^^^^^^^^

**Backward Breaking Changes**

- Deprecated ``torch<2.0`` support.
- :meth:`modelopt.torch.utils.dataset_utils.get_dataset_dataloader` now returns a key value pair instead of the tensor.

**New Features**

- New APIs and examples: :mod:`modelopt.torch.prune` for pruning Conv, Linear, and Attention heads for
  NVIDIA Megatron-core GPT-style models (e.g. Llama 3), PyTorch Computer Vision models, and HuggingFace Bert/GPT-J models.
- New API: :mod:`modelopt.torch.distill` for knowledge distillation, along with guides and example.
- New Example: HF BERT Prune, Distill & Quantize
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
