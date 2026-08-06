# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Code that export quantized Hugging Face models for deployment."""

import json
import tempfile
import warnings
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

from modelopt.torch.quantization.utils.core_utils import has_accelerate_offload
from modelopt.torch.utils.distributed import is_fsdp2_model

# _HAS_DIFFUSERS is diffusers_utils' own probe; re-deriving it here would drift, since
# that module imports cleanly whether or not diffusers is installed.
from .diffusers_utils import _HAS_DIFFUSERS as HAS_DIFFUSERS
from .diffusers_utils import is_diffusers_object

try:
    from modelopt.torch.sparsity.attention_sparsity.conversion import export_sparse_attention_config
except ImportError:
    export_sparse_attention_config = None

# Importing the built-in handlers installs their entries in the two registries.
from . import hf_export_handlers as _hf_export_handlers  # noqa: F401
from .convert_hf_config import convert_hf_quant_config_format
from .hf_export_prep import (
    _add_mtp_exclusions,
    _patch_revert_weight_conversion,
    _prepare_moe_inputs,
    _resolve_export_dtype,
    _sanitize_generation_config_for_save,
    _unpatch_revert_weight_conversion,
    _warn_on_unsynced_moe_gate_up,
    requantize_resmooth_fused_llm_layers,
)
from .hf_weight_export import _process_quantized_modules
from .model_utils import _reorder_canonical_first
from .plugins import SpeculativeDecodingExporter, has_spec_opt, sanitize_hf_config_for_deployment
from .quant_aware_conversion import (
    build_reverse_name_mapper,
    revert_quant_config_names,
    revert_weight_conversion_quant_aware,
)
from .quant_utils import get_quant_config, postprocess_state_dict, sync_tied_input_amax
from .unified_export_diffusers import _export_diffusers_checkpoint
from .unified_export_hf_streaming import _export_transformers_checkpoint_streaming

__all__ = ["export_hf_checkpoint", "export_speculative_decoding"]


def _export_transformers_checkpoint(
    model: nn.Module,
    dtype: torch.dtype | None = None,
    is_modelopt_qlora: bool = False,
    **kwargs,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Exports the torch model to the packed checkpoint with original HF naming.

    The packed checkpoint will be consumed by the TensorRT-LLM unified converter.

    Builds the whole quantized state dict in memory, so it requires every weight to be
    resident. Models with accelerate CPU/disk offload are rejected here and handled by
    :func:`_export_transformers_checkpoint_streaming`, which materializes one layer at a
    time; :func:`export_hf_checkpoint` picks between the two.

    Args:
        model: the full torch model to export. The actual quantized model may be a submodule.
        dtype: the weights data type to export the unquantized layers or the default model data type if None.

    Returns:
        post_state_dict: Dict containing quantized weights
        quant_config: config information to export hf_quant_cfg.json

    Raises:
        NotImplementedError: if the model has accelerate offload hooks.
    """
    dtype = _resolve_export_dtype(model, dtype)
    _prepare_moe_inputs(model, dtype, is_modelopt_qlora)

    # Resmooth and requantize fused layers
    # TODO: Handle mixed precision
    requantize_resmooth_fused_llm_layers(model)

    # Offloaded models need their weights materialized layer-by-layer, which this
    # whole-state-dict path cannot do; export_hf_checkpoint() streams them instead.
    if has_accelerate_offload(model):
        raise NotImplementedError(
            "_export_transformers_checkpoint does not support disk/CPU-offloaded models. "
            "Use export_hf_checkpoint() which dispatches to _export_transformers_checkpoint_streaming."
        )

    # Remove all hooks from the model
    try:
        from accelerate.hooks import remove_hook_from_module

        remove_hook_from_module(model, recurse=True)
    except ImportError:
        pass  # no accelerate installed → no offload hooks exist to remove

    quant_config = get_quant_config(model, is_modelopt_qlora=is_modelopt_qlora)

    _add_mtp_exclusions(model, quant_config)

    _warn_on_unsynced_moe_gate_up(model)

    # Merge per-side input_quantizer amaxes BEFORE _process_quantized_modules,
    # so the merged value flows into input_scale derivation downstream.
    synced_input = sync_tied_input_amax(model)
    if synced_input:
        print(
            f"sync_tied_input_amax: max-merged input_quantizer amaxes across "
            f"{synced_input} tied module group(s)"
        )

    # Process all quantized modules and export weights
    from modelopt.torch.quantization.plugins.huggingface import _reconstruct_fused_moe_linear

    _process_quantized_modules(model, dtype, is_modelopt_qlora)
    _reconstruct_fused_moe_linear(model)

    if is_fsdp2_model(model):
        # FSDP2: gather the full (unsharded) state_dict to CPU on rank 0.
        quantized_state_dict = get_model_state_dict(
            model,
            options=StateDictOptions(full_state_dict=True, cpu_offload=True),
        )
    else:
        # Non-FSDP2: assumes a replicated model (rank 0 has the full state dict).
        quantized_state_dict = model.state_dict()

    # We define kv cache scale as amax / 448 for both FP8 and NVFP4 KV cache quantization.
    kv_cache_max_bound = 448
    kv_cache_format = quant_config["quantization"]["kv_cache_quant_algo"]

    # Reorder so canonical-side tied keys (per HF's _tied_weights_keys)
    # iterate first into postprocess_state_dict's first-wins data_ptr dedup.
    # Self-gated to DiffusionGemma inside _reorder_canonical_first; no-op
    # for every other model.
    quantized_state_dict = _reorder_canonical_first(quantized_state_dict, model)

    quantized_state_dict = postprocess_state_dict(
        quantized_state_dict, kv_cache_max_bound, kv_cache_format, is_modelopt_qlora
    )

    return quantized_state_dict, quant_config


def export_speculative_decoding(
    model: torch.nn.Module,
    dtype: torch.dtype | None = None,
    export_dir: Path | str = tempfile.gettempdir(),
) -> None:
    """Export speculative decoding HuggingFace model checkpoint."""
    assert has_spec_opt(model), "Model is not optimized for speculative decoding."

    exporter: SpeculativeDecodingExporter = model.get_exporter()
    exporter.export(export_dir, dtype)


def _write_hf_export_config(
    model: nn.Module,
    hf_quant_config: dict | None,
    export_dir: Path,
) -> None:
    """Write hf_quant_config.json (if quantized) and embed quantization_config into config.json."""
    quantization_details = (hf_quant_config or {}).get("quantization", {})
    is_quantized_export = (
        quantization_details.get("quant_algo") is not None
        or quantization_details.get("kv_cache_quant_algo") is not None
    )
    quantization_config = None
    if hf_quant_config is not None and is_quantized_export:
        with open(f"{export_dir}/hf_quant_config.json", "w") as file:
            json.dump(hf_quant_config, file, indent=4)
        quantization_config = convert_hf_quant_config_format(hf_quant_config)

    original_config = f"{export_dir}/config.json"
    with open(original_config) as file:
        config_data = json.load(file)
    sanitize_hf_config_for_deployment(config_data, model)
    if quantization_config is not None:
        config_data["quantization_config"] = quantization_config
    if export_sparse_attention_config is not None:
        sparse_attn_config = export_sparse_attention_config(model)
        if sparse_attn_config is not None:
            config_data["sparse_attention_config"] = sparse_attn_config
    with open(original_config, "w") as file:
        json.dump(config_data, file, indent=4)


def export_hf_checkpoint(
    model: Any,
    dtype: torch.dtype | None = None,
    export_dir: Path | str = tempfile.gettempdir(),
    save_modelopt_state: bool = False,
    components: list[str] | None = None,
    extra_state_dict: dict[str, torch.Tensor] | None = None,
    max_shard_size: int | str = "10GB",
    **kwargs,
):
    """Export quantized HuggingFace model checkpoint (transformers or diffusers).

    This function automatically detects whether the model is from transformers
    or diffusers and applies the appropriate export logic.

    Under ``torch.distributed`` (e.g. FSDP2), all ranks participate in the
    collective state-dict gather inside ``_export_transformers_checkpoint``;
    only rank 0 writes files. A final barrier syncs the other ranks.

    Args:
        model: The full torch model to export. The actual quantized model may be a submodule.
            Supports both transformers models (e.g., LlamaForCausalLM) and diffusers
            models/pipelines (e.g., StableDiffusionPipeline, UNet2DConditionModel).
        dtype: The weights data type to export the unquantized layers or the default
            model data type if None.
        export_dir: The target export path.
        save_modelopt_state: Whether to save the modelopt state_dict.
        components: Only used for diffusers pipelines. Optional list of component names
            to export. If None, all quantized components are exported.
        extra_state_dict: Extra state dictionary to add to the exported model.
        max_shard_size: Maximum size of each safetensors shard file. Defaults to "10GB".
        **kwargs: Runtime-specific post-processing options forwarded to
            :func:`_postprocess_safetensors` for diffusion model exports.
            See its docstring for supported keys.
    """
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    is_diffusers_obj = False
    if HAS_DIFFUSERS:
        is_diffusers_obj = is_diffusers_object(model)
    if is_diffusers_obj:
        _export_diffusers_checkpoint(
            model,
            dtype,
            export_dir,
            components,
            max_shard_size,
            **kwargs,
        )
        return

    is_distributed = (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
        and is_fsdp2_model(model)
    )
    # Offloaded models take the streaming path: it materializes one layer at a time and
    # writes each straight to a shard file, so peak memory is one layer plus one shard
    # buffer instead of the whole quantized state dict.
    _offloaded = has_accelerate_offload(model)

    try:
        if _offloaded:
            if save_modelopt_state:
                warnings.warn(
                    "save_modelopt_state=True is not supported in the streaming offload export "
                    "path and will be ignored."
                )
            _, hf_quant_config = _export_transformers_checkpoint_streaming(
                model,
                dtype,
                export_dir=export_dir,
                max_shard_size=max_shard_size,
                extra_state_dict=extra_state_dict,
                **kwargs,
            )
            if getattr(model, "hf_quantizer", None) is not None:
                model.hf_quantizer = None
            try:
                name_mapper = build_reverse_name_mapper(model)
                if name_mapper is not None and hf_quant_config:
                    revert_quant_config_names(hf_quant_config.get("quantization", {}), name_mapper)
            except Exception as exc:
                warnings.warn(
                    f"Quant-aware reverse weight conversion skipped ({exc}); exported tensor "
                    "names may not match the original HF hub checkpoint."
                )
            _write_hf_export_config(model, hf_quant_config, export_dir)
            return

        post_state_dict, hf_quant_config = _export_transformers_checkpoint(model, dtype, **kwargs)

        # Remove hf_quantizer from model so post_state_dict can be exported.
        if getattr(model, "hf_quantizer", None) is not None:
            model.hf_quantizer = None

        export_state_dict = {**post_state_dict, **(extra_state_dict or {})}

        # transformers may have applied a load-time conversion_mapping (fused gate_up_proj,
        # renamed MoE leaves, reordered model/language_model prefix), so the in-memory names
        # differ from the original hub checkpoint. Reverse it quantization-aware so exported
        # tensor names stay aligned with the hub checkpoint (the unified-checkpoint contract).
        # transformers' own revert_weight_conversion errors on 0-d scalar scale tensors, so we
        # do it here. The same rename is applied to the quant-config module references
        # (exclude_modules / quantized_layers keys) so a deployment loader matches them against
        # the reverted hub-named modules (otherwise an excluded BF16 layer is loaded as quantized
        # and fails). Best-effort and atomic: any failure (an op we cannot reverse yet,
        # transformers API drift, unexpected shapes) falls back to the in-memory names for BOTH
        # weights and config so they stay mutually consistent.
        try:
            name_mapper = build_reverse_name_mapper(model)
            export_state_dict = revert_weight_conversion_quant_aware(model, export_state_dict)
            if name_mapper is not None and hf_quant_config:
                revert_quant_config_names(hf_quant_config.get("quantization", {}), name_mapper)
        except Exception as exc:
            warnings.warn(
                f"Quant-aware reverse weight conversion skipped ({exc}); exported tensor "
                "names may not match the original HF hub checkpoint."
            )

        # Under torch.distributed only rank 0 writes; others sync at the finally barrier.
        if is_distributed and torch.distributed.get_rank() != 0:
            return

        # Keep transformers' own revert_weight_conversion disabled (the quant-aware reverse
        # above replaces it): it can't handle quantized state dicts (RuntimeError on 0-d scalar
        # scale tensors). Patch both the source and importing module since modeling_utils does
        # `from core_model_loading import revert_weight_conversion`.
        _patches = _patch_revert_weight_conversion()

        _sanitize_generation_config_for_save(model)

        # TODO: parallelize the disk write across ranks (avoid single-process speed + rank-0 OOM).
        try:
            model.save_pretrained(
                export_dir,
                state_dict=export_state_dict,
                save_modelopt_state=save_modelopt_state,
                max_shard_size=max_shard_size,
            )
        finally:
            _unpatch_revert_weight_conversion(_patches)

        _write_hf_export_config(model, hf_quant_config, export_dir)

    except Exception as e:
        warnings.warn(
            "Cannot export model to the model_config. The modelopt-optimized model state_dict"
            " can be saved with torch.save for further inspection."
        )
        raise e
    finally:
        if is_distributed:
            torch.distributed.barrier()
