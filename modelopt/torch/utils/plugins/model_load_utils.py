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

"""HuggingFace model loading: :func:`load_pretrained`, and the plan to distribute a model by."""

import logging
import os
import re
from itertools import chain
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from huggingface_hub import snapshot_download

if TYPE_CHECKING:
    from collections.abc import Iterable
from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict
from torch.distributed.tensor import DTensor
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, PretrainedConfig
from transformers.conversion_mapping import get_model_conversion_mapping
from transformers.core_model_loading import WeightConverter, dot_natural_key, rename_source_key

from modelopt.torch.utils import env
from modelopt.torch.utils.distributed import (
    DistributionPlan,
    barrier,
    broadcast_state_dict,
    fsdp2_wrap,
    is_initialized,
)
from modelopt.torch.utils.logging import print_rank_0
from modelopt.torch.utils.plugins.hf_checkpoint_utils import (
    checkpoint_tensor_files,
    ensure_local_checkpoint,
    indexed_weight_map,
    read_safetensors_subset,
)

logger = logging.getLogger(__name__)


def _resolve_checkpoint_dir(ckpt_path: str, rank: int) -> str:
    """Local dir for ``ckpt_path``; resolves an HF Hub ID (rank 0 downloads, others wait)."""
    if os.path.isdir(ckpt_path):
        return ckpt_path
    if rank == 0:
        snapshot_download(ckpt_path)
    if is_initialized():
        barrier()
    return snapshot_download(ckpt_path)


def _materialize_meta_model(model: nn.Module, device: torch.device) -> None:
    """Replace meta params/buffers with empty real ones on ``device``; move real buffers there.

    Goes through ``model._apply`` so FSDP2's override refreshes its internal
    ``_sharded_param_data`` pointers via ``reset_sharded_param``.
    """
    model._apply(lambda t: torch.empty_like(t, device=device) if t.is_meta else t.to(device))


def _promote_non_dtensor_to_gpu(model: nn.Module, device: torch.device) -> None:
    """Move all non-DTensor params + buffers in ``model`` to ``device`` in-place.

    Used after CPU-offload loading: decoder DTensor shards stay on CPU (FSDP2
    streams them to GPU per layer), while root-level plain params and buffers
    need to live on GPU so forwards work.
    """
    for module in model.modules():
        for name, param in list(module._parameters.items()):
            if param is None or isinstance(param, DTensor):
                continue
            module._parameters[name] = nn.Parameter(
                param.data.to(device), requires_grad=param.requires_grad
            )
        for name, buf in list(module._buffers.items()):
            if buf is None or isinstance(buf, DTensor):
                continue
            module._buffers[name] = buf.to(device)


def _conversion_plan(model: nn.Module, conversions: list | None = None) -> dict | None:
    """Transformers' own conversion mapping for ``model``, or ``None`` if nothing needs converting.

    ``legacy_renames`` covers a remote-code model's ``_checkpoint_conversion_mapping``; the
    ``renamings``/``converters`` from HF's engine drive renaming + MoE weight fusion directly.
    ``conversions`` replaces the model's stock mapping with a given list, such as the one
    ``from_pretrained`` records of what it ran; that record is complete, so no legacy renames are
    layered on top of it.
    """
    if conversions is None:
        legacy_renames = dict(getattr(model, "_checkpoint_conversion_mapping", None) or {})
        conversions = get_model_conversion_mapping(model)
    else:
        legacy_renames = {}
    renamings, converters = [], []
    for entry in conversions:
        (converters if isinstance(entry, WeightConverter) else renamings).append(entry)
    if not (legacy_renames or renamings or converters):
        return None
    return {
        "legacy_renames": legacy_renames,
        "renamings": renamings,
        "converters": converters,
        "prefix": getattr(model, "base_model_prefix", ""),
        "meta_state_dict": model.state_dict(),
    }


def _resolve_target(plan: dict, key: str) -> tuple[str, str | None]:
    """Resolve a checkpoint key to ``(target param name, matched converter source pattern)``.

    No tensors are read. ``source_pattern`` is ``None`` for a plain (non-fused) key.
    """
    for old, new in plan["legacy_renames"].items():
        key = re.sub(old, new, key)
    return rename_source_key(
        key, plan["renamings"], plan["converters"], plan["prefix"], plan["meta_state_dict"]
    )


def _conversion_plans(model: nn.Module) -> list[dict]:
    """The plans a load of ``model`` may have converted keys with; see :func:`unplaced_source_keys`.

    The stock mapping, and the record ``from_pretrained`` kept of what it ran when it built ``model``.
    """
    loaded = getattr(model, "_weight_conversions", None)
    plans = (_conversion_plan(model), _conversion_plan(model, loaded) if loaded else None)
    return [plan for plan in plans if plan is not None]


def _target_param(plans: list[dict], key: str, model_param_names: set[str]) -> str | None:
    """The parameter or buffer ``key`` loads into under any of ``plans``, or ``None`` for none."""
    if not plans:
        return key if key in model_param_names else None
    for plan in plans:
        target = _resolve_target(plan, key)[0]
        if target in model_param_names:
            return target
    return None


def _convert_keys(plan: dict, state: dict) -> dict:
    """Rename 1:1 keys and fuse per-expert keys by driving transformers' own conversion ops."""
    result: dict = {}
    collected: dict = {}  # target -> (converter, {source_pattern: [(sort_key, tensor)]})
    for key in sorted(state, key=dot_natural_key):
        renamed, source_pattern = _resolve_target(plan, key)
        if source_pattern is None:  # plain rename, no fusion
            result[renamed] = state[key]
            continue
        conv = next(c for c in plan["converters"] if source_pattern in c.source_patterns)
        collected.setdefault(renamed, (conv, {}))[1].setdefault(source_pattern, []).append(
            (dot_natural_key(key), state[key])
        )
    for target, (conv, by_src) in collected.items():
        tensors = {
            sp: [t for _, t in sorted(lst, key=lambda kt: kt[0])] for sp, lst in by_src.items()
        }
        for op in conv.operations:  # transformers runs the fusion math (any op, no whitelist)
            tensors = op.convert(
                tensors, source_patterns=conv.source_patterns, target_patterns=conv.target_patterns
            )
        if len(tensors) != 1:
            raise NotImplementedError(
                f"Only many-to-one conversions supported; got {list(tensors)}"
            )
        result[target] = next(iter(tensors.values()))
    return result


def build_meta_causal_lm(
    ckpt_path: str,
    trust_remote_code: bool,
    attn_implementation: str | None,
    hf_config=None,
    auto_class: Any = AutoModelForCausalLM,
):
    """Build a meta-init causal LM (no real storage allocated), or ``auto_class``'s model."""
    if hf_config is None:
        config_kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code}
        if attn_implementation is not None:
            config_kwargs["attn_implementation"] = attn_implementation
        hf_config = AutoConfig.from_pretrained(ckpt_path, **config_kwargs)
    elif attn_implementation is not None:
        # Honor the override even when the caller passed in a pre-fetched config.
        hf_config._attn_implementation = attn_implementation
    dtype = getattr(hf_config, "dtype", None) or torch.bfloat16
    from accelerate import init_empty_weights  # only real callers of this function need it

    with init_empty_weights(include_buffers=False):
        model = auto_class.from_config(hf_config, dtype=dtype, trust_remote_code=trust_remote_code)
    model.eval()
    return model


def _layers_for_rank(n_layers: int, world_size: int, r: int) -> list[int]:
    return [i for i in range(n_layers) if i % world_size == r]


def _read_and_convert(
    resolved_path: str, weight_map: dict, keyset: set[str], plan: dict | None
) -> dict:
    raw = read_safetensors_subset(resolved_path, weight_map, lambda k: k in keyset)
    return _convert_keys(plan, raw) if plan else raw


# One decoder layer's converted weights (param-name suffix -> tensor); the outer dict is keyed
# by decoder-layer index.
LayerStateDict = dict[str, torch.Tensor]
OwnedLayerStateDicts = dict[int, LayerStateDict]


def _read_owned_layers(
    resolved_path: str,
    weight_map: dict,
    layer_sources: dict,
    owned_layer_indices: list[int],
    plan: dict | None,
) -> OwnedLayerStateDicts:
    """Read + convert this rank's owned decoder layers from disk (ranks read in parallel)."""
    return {
        layer_idx: _read_and_convert(resolved_path, weight_map, set(layer_sources[layer_idx]), plan)
        for layer_idx in owned_layer_indices
    }


def _broadcast_load_group(
    layer_indices: list[int],
    source_rank: int,
    current_rank: int,
    owned_layer_state_dicts: OwnedLayerStateDicts,
    decoder_layers: list[nn.Module],
    layer_prefixes: list[str],
    device: torch.device,
    cpu_offload: bool,
) -> None:
    """Broadcast ``layer_indices`` from ``source_rank`` to all ranks and reshard into FSDP2 shards.

    The owner assembles the group's full tensors; every rank receives them, reshards its local slice,
    then frees the full copy (capping the transient GPU peak). The owner drops its read copy after.
    """
    group_state_dict: dict | None = None
    if current_rank == source_rank:
        group_state_dict = {}
        for layer_idx in layer_indices:
            group_state_dict.update(owned_layer_state_dicts[layer_idx])
    broadcasted_state_dict = broadcast_state_dict(group_state_dict, src=source_rank, device=device)
    for layer_idx in layer_indices:
        prefix = layer_prefixes[layer_idx]
        layer_state_dict = {
            k[len(prefix) :]: v for k, v in broadcasted_state_dict.items() if k.startswith(prefix)
        }
        if cpu_offload:
            layer_state_dict = {k: v.cpu() for k, v in layer_state_dict.items()}
        set_model_state_dict(
            decoder_layers[layer_idx],
            layer_state_dict,
            options=StateDictOptions(full_state_dict=True, broadcast_from_rank0=False),
        )
        del layer_state_dict
    del broadcasted_state_dict
    if current_rank == source_rank:
        for layer_idx in layer_indices:
            del owned_layer_state_dicts[layer_idx]


def _group_sources_by_layer(
    ckpt_keys: "Iterable[str]",
    plans: list[dict],
    model_param_names: set[str],
    layer_prefixes: list[str],
) -> tuple[dict[int, list[str]], list[str], list[str]]:
    """Bucket checkpoint keys by the decoder layer their converted target lives in.

    Returns ``(layer_sources, non_layer_sources, unplaced)``: ``layer_sources[i]`` holds the keys
    targeting decoder layer ``i``, ``non_layer_sources`` holds root (embed/lm_head/norm) keys, and
    ``unplaced`` NAMES the keys whose target isn't in the model -- weights the built model has no
    home for (an MTP head, an auxiliary tower). The names are kept, not just counted, so the export
    can copy them through: PTQ never touches them, but the exported checkpoint is still expected to
    contain them.
    """
    layer_sources: dict[int, list[str]] = {i: [] for i in range(len(layer_prefixes))}
    non_layer_sources: list[str] = []
    unplaced: list[str] = []
    for ckpt_key in ckpt_keys:
        target = _target_param(plans, ckpt_key, model_param_names)
        if target is None:
            unplaced.append(ckpt_key)
            continue
        for i, prefix in enumerate(layer_prefixes):
            if target.startswith(prefix):
                layer_sources[i].append(ckpt_key)
                break
        else:
            non_layer_sources.append(ckpt_key)
    return layer_sources, non_layer_sources, unplaced


def unplaced_source_keys(model: nn.Module, ckpt_path: str) -> list[str]:
    """Checkpoint tensors the loaded ``model`` has no parameter for, e.g. an MTP head.

    Every tensor in the weight files the loader opened
    (:func:`~modelopt.torch.utils.plugins.hf_checkpoint_utils.checkpoint_tensor_files`, which also
    sees tensors the index omits) is mapped through the conversions the load ran, and is placed if
    that names a parameter or buffer of ``model``. Architecture-agnostic by construction: nothing
    here knows what an MTP head looks like.

    The loader's own ``unexpected_keys`` cannot stand in for this. Transformers drops from it the
    keys an architecture's ``_keys_to_ignore_on_load_unexpected`` matches -- Qwen3-Next ``^mtp.*``,
    DeepSeek-V3 and GLM their MTP layers -- which are exactly the weights the export must carry, and
    it names some keys after conversion (a fused ``...experts.gate_up_proj``) where the export needs
    the source tensors on disk.

    Two conversion plans are replayed, and a key resolving through either is placed: the model's
    stock mapping, and ``_weight_conversions``, the record ``from_pretrained`` keeps of what it ran.
    Only the record has what loading adds to the stock mapping -- a dequantizing quantizer folds FP8
    ``weight_scale_inv`` and MXFP4 ``*_blocks``/``*_scales`` into the weights, an on-the-fly
    decompressor feeds ``weight_packed`` experts in -- but it lists only conversions some key used,
    and a model our own loader built from its config has none, so the stock mapping answers there.

    A checkpoint without safetensors (``pytorch_model.bin``) has nothing the export can carry: ``[]``.
    """
    tensor_files = checkpoint_tensor_files(ckpt_path)
    if not tensor_files:
        return []
    model_param_names = {n for n, _ in chain(model.named_parameters(), model.named_buffers())}
    plans = _conversion_plans(model)
    # A key that already names a parameter is placed as it is: a registered Qwen VLM's checkpoint
    # uses the model's own names, which its stock mapping would rename away.
    return [
        key
        for key in tensor_files
        if key not in model_param_names and _target_param(plans, key, model_param_names) is None
    ]


def record_unplaced_source_keys(
    model: nn.Module, ckpt_path: str, keys: "Iterable[str] | None" = None
) -> list[str]:
    """Record on ``model`` the checkpoint tensors it has no parameter for, for the export to carry.

    The load is what knows how the checkpoint maps onto the model, so the list is worked out once,
    at load time, and the export reads it back (``_modelopt_unplaced_source_keys``, beside the
    checkpoint in ``_modelopt_source_checkpoint``) rather than deriving it again. Pass ``keys`` when
    the loader already found them while placing the weights; otherwise :func:`unplaced_source_keys`
    derives them from the loaded model. Every loader of a model that may be exported calls this:
    :func:`load_pretrained` and :func:`parallel_load_and_prepare_fsdp2` do.
    """
    keys = sorted(unplaced_source_keys(model, ckpt_path) if keys is None else keys)
    model._modelopt_unplaced_source_keys = keys
    model._modelopt_source_checkpoint = str(ckpt_path)
    return keys


def load_pretrained(
    ckpt_path: str,
    auto_class: Any = None,
    *,
    model_type: str | None = None,
    **kwargs,
) -> nn.Module:
    """Load a Hugging Face model for ModelOpt to optimize and export.

    ``auto_class.from_pretrained(ckpt_path, **kwargs)`` (``auto_class`` defaults to
    ``AutoModelForCausalLM``), plus what ModelOpt needs around the load:

    * A Hub model ID is made local first (:func:`ensure_local_checkpoint`, which takes the
      ``revision``, ``token``, ``cache_dir`` and ``local_files_only`` ``kwargs``): the export reads
      the source checkpoint from disk. Under ``torch.distributed`` every rank must call this.
    * The model type's loading hook runs
      (:func:`~modelopt.torch.models.hf.prepare_model_for_loading`), e.g. to keep the MTP head a
      Nemotron-H checkpoint carries. ``model_type`` selects it, read from the checkpoint's config
      when omitted.
    * The checkpoint tensors the model has no parameter for are recorded
      (:func:`record_unplaced_source_keys`), so the export carries them over unchanged. The model
      is returned alone: ``output_loading_info=True`` is refused.
    """
    if kwargs.pop("output_loading_info", False):
        raise ValueError(
            "load_pretrained returns the model alone and does not take output_loading_info=True; "
            "the checkpoint tensors the model has no parameter for are recorded on it instead "
            "(see record_unplaced_source_keys)."
        )
    # Importing modelopt.torch.models registers every model spec; only callers of this need them.
    from modelopt.torch.models.hf import prepare_model_for_loading

    auto_class = auto_class or AutoModelForCausalLM
    _, ckpt_path = ensure_local_checkpoint(
        ckpt_path,
        **{
            k: kwargs.pop(k)
            for k in ("revision", "token", "cache_dir", "local_files_only")
            if k in kwargs
        },
    )
    trust_remote_code = kwargs.get("trust_remote_code", False)
    if model_type is None:
        model_type = AutoConfig.from_pretrained(
            ckpt_path, trust_remote_code=trust_remote_code
        ).model_type
    with prepare_model_for_loading(
        model_type,
        ckpt_path,
        trust_remote_code,
        model_class=None if auto_class in (AutoModel, AutoModelForCausalLM) else auto_class,
    ):
        model = auto_class.from_pretrained(ckpt_path, **kwargs)
    unplaced = record_unplaced_source_keys(model, ckpt_path)
    if unplaced:
        print_rank_0(
            f"{len(unplaced)} checkpoint tensor(s) the model has no parameter for "
            f"(e.g. {unplaced[0]}); the export will carry them over unchanged."
        )
    return model


def parallel_load_and_prepare_fsdp2(
    ckpt_path: str,
    device: torch.device,
    rank: int,
    world_size: int,
    trust_remote_code: bool = False,
    mp_policy=None,
    cpu_offload: bool = False,
    attn_implementation: str | None = None,
    hf_config=None,
    broadcast_chunk_size: int | None = 8,
) -> nn.Module:
    """Load and FSDP2-shard a HuggingFace causal LM via parallel safetensors reads.

    Round-robin assigns decoder layers to ranks; each rank reads only its owned
    layers' weights from disk in parallel, then broadcasts to the others. Non-decoder
    weights (embed, lm_head, norm) are read on rank 0 and broadcast.

    Requires an initialized ``torch.distributed`` process group (FSDP2's ``fully_shard``
    and the per-layer broadcasts both need it). A 1-rank PG (e.g. ``torchrun
    --nproc_per_node=1``) is allowed; bare single-process is not.

    Pass ``hf_config`` if the caller has already fetched it (skips a redundant fetch).

    ``broadcast_chunk_size`` sets how many of a source's owned layers are broadcast per collective:
    a smaller value lowers the peak transient GPU memory at the cost of more collectives (default 8;
    pass ``None`` to broadcast all of a source's layers at once).
    """
    resolved_path = _resolve_checkpoint_dir(ckpt_path, rank)
    weight_map = indexed_weight_map(resolved_path)
    if not weight_map:
        raise RuntimeError(
            f"No safetensors checkpoint at {resolved_path} "
            "(expected model.safetensors or model.safetensors.index.json)."
        )

    model = build_meta_causal_lm(resolved_path, trust_remote_code, attn_implementation, hf_config)

    # fsdp2_wrap shards each decoder layer + the root (embed/lm_head/norm sharded, not replicated).
    decoder_layers = fsdp2_wrap(model, mp_policy=mp_policy, cpu_offload=cpu_offload)
    module_to_name = {m: n for n, m in model.named_modules()}
    layer_prefixes = [module_to_name[layer] + "." for layer in decoder_layers]

    # transformers fuses/renames checkpoint keys so they no longer match param names 1:1
    # (None => nothing to convert).
    plan = _conversion_plan(model)

    # Valid targets; keys converting to anything else are aux weights (e.g. an MTP head) we skip.
    model_param_names = {n for n, _ in chain(model.named_parameters(), model.named_buffers())}

    # Bucket each checkpoint tensor by its target's decoder layer (root params go to
    # non_layer_sources). Every tensor in the files the index names, not only the ones it lists, so
    # the unplaced ones fall out of this same pass. A model built from its config ran no
    # conversions of its own, so the stock plan is the only one.
    layer_sources, non_layer_sources, unplaced = _group_sources_by_layer(
        checkpoint_tensor_files(resolved_path),
        [plan] if plan else [],
        model_param_names,
        layer_prefixes,
    )
    if unplaced:
        logger.debug(
            "%d checkpoint keys have no parameter in the built model (e.g. an MTP head); "
            "recorded for the export to copy through verbatim",
            len(unplaced),
        )
    # Recorded on the model so export can read them back from the source without being told where
    # it came from. Not a state dict: holding these tensors from load to export would waste the
    # memory this loader exists to save.
    record_unplaced_source_keys(model, resolved_path, unplaced)

    _materialize_meta_model(model, torch.device("cpu") if cpu_offload else device)

    owned_layer_indices = _layers_for_rank(len(decoder_layers), world_size, rank)
    owned_layer_state_dicts = _read_owned_layers(
        resolved_path, weight_map, layer_sources, owned_layer_indices, plan
    )

    # Smaller broadcast_chunk_size lowers the transient GPU peak (more, smaller collectives).
    for source_rank in range(world_size):
        source_layer_indices = _layers_for_rank(len(decoder_layers), world_size, source_rank)
        if not source_layer_indices:
            continue
        chunk = broadcast_chunk_size or len(source_layer_indices)
        for start in range(0, len(source_layer_indices), chunk):
            _broadcast_load_group(
                source_layer_indices[start : start + chunk],
                source_rank,
                rank,
                owned_layer_state_dicts,
                decoder_layers,
                layer_prefixes,
                device,
                cpu_offload,
            )

    # Non-decoder params: rank 0 reads + broadcasts; resharded into the root below.
    # TODO: layerwise support.
    non_layer = None
    if rank == 0:
        non_layer = _read_and_convert(resolved_path, weight_map, set(non_layer_sources), plan)
    non_layer = broadcast_state_dict(non_layer, src=0, device=device)
    if cpu_offload:
        non_layer = {k: v.cpu() for k, v in non_layer.items()}
    # shard_root=True makes the root params sharded DTensors, so reshard the full tensors via
    # set_model_state_dict. strict=False: decoder keys are absent here (loaded above).
    set_model_state_dict(
        model,
        non_layer,
        options=StateDictOptions(full_state_dict=True, broadcast_from_rank0=False, strict=False),
    )

    if cpu_offload:
        # Loaded on CPU for set_model_state_dict; FSDP2 streams decoder shards per forward, but
        # the unwrapped root must live on GPU, so promote it.
        _promote_non_dtensor_to_gpu(model, device)
    if hasattr(model, "tie_weights"):
        model.tie_weights()
    return model


def _auto_class(hf_config, require_lm_head: bool = True) -> Any:
    """The auto class that builds ``hf_config``'s whole model.

    A causal LM builds through ``AutoModelForCausalLM``. A VLM -- a nested text config, a vision
    config, or a ``*ForConditionalGeneration`` architecture -- builds through ``AutoModel``, which
    leaves its LM head out. ``require_lm_head`` builds it with the head: through the remote
    ``AutoModelForCausalLM`` when the checkpoint registers its ``*ForConditionalGeneration`` there
    and no image-text class, else through ``AutoModelForImageTextToText`` if that recognizes the
    config.
    """
    architectures = getattr(hf_config, "architectures", None) or []
    is_vlm = (
        getattr(hf_config, "text_config", None) is not None
        or getattr(hf_config, "vision_config", None) is not None
        or any("conditionalgeneration" in str(arch).lower() for arch in architectures)
    )
    if not is_vlm:
        return AutoModelForCausalLM
    if not require_lm_head:
        return AutoModel
    auto_map = getattr(hf_config, "auto_map", None) or {}
    remote_with_head = (
        "conditionalgeneration" in str(auto_map.get("AutoModelForCausalLM", "")).lower()
    )
    if remote_with_head and "AutoModelForImageTextToText" not in auto_map:
        return AutoModelForCausalLM
    try:
        from transformers import AutoModelForImageTextToText
    except ImportError:
        return AutoModel
    if type(hf_config) in AutoModelForImageTextToText._model_mapping:
        return AutoModelForImageTextToText
    return AutoModel


# The share of one GPU's memory a replicated dense part may take. The rest is left to calibration's
# activations and communication buffers, and to the rank's share of the experts.
_REPLICATED_SHARE = 0.6


def _gpu_memory() -> int | None:
    """This rank's GPU memory in bytes, or ``None`` without a GPU."""
    if not torch.cuda.is_available():
        return None
    return torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory


def _dense_from_env(world_size: int) -> str:
    """``MODELOPT_FSDP_SIZE`` as a dense layout, or ``"auto"``."""
    value = env.fsdp_size().lower()
    if value == "auto":
        return "auto"
    if value == "1":
        return "replicate"
    if value in ("world", "full", str(world_size)):
        return "fsdp"
    raise ValueError(
        f"MODELOPT_FSDP_SIZE={value!r}: expected auto, 1 (replicate the dense part) or world "
        f"(FSDP2-shard it over all {world_size} ranks)."
    )


def _decide_distribution(
    ckpt_path: str, hf_config, world_size: int, ep_size, dense, trust_remote_code: bool
) -> DistributionPlan:
    """The plan :func:`resolve_distribution_plan` returns, decided by this rank alone."""
    from modelopt.torch.utils.moe import (
        _ep_config,
        _routed_expert_modules,
        choose_ep_dp_degrees,
        num_experts_from_config,
    )

    if world_size == 1:
        # Nothing to shard over, so nothing about the model matters, nor do the knobs: a shell still
        # exporting a multi-GPU run's would otherwise reshape a single process's load. The caller's
        # own arguments still count, as dense="fsdp" for FSDP2 on one rank.
        return DistributionPlan(
            ep_size=1 if ep_size is None or str(ep_size) in ("auto", "0") else int(ep_size),
            dense="replicate" if dense in (None, "auto") else dense,
        )
    ep_size = env.ep_degree() if ep_size is None else ep_size
    ep_size = "auto" if str(ep_size) in ("auto", "0") else int(ep_size)
    dense = _dense_from_env(world_size) if dense is None else dense
    try:
        model = build_meta_causal_lm(
            ckpt_path, trust_remote_code, None, hf_config, auto_class=_auto_class(hf_config)
        )
    except Exception as e:
        raise RuntimeError(
            f"Deciding how to distribute {hf_config.model_type} needs its model built on the meta "
            f"device, which failed: {type(e).__name__}: {e}"
        ) from e
    ep_plan = getattr(_ep_config(hf_config), "base_model_ep_plan", None) or {}
    experts = _routed_expert_modules(model, ep_plan)
    num_experts = num_experts_from_config(hf_config) if experts else 0
    if ep_size == "auto":
        ep_size = choose_ep_dp_degrees(world_size, num_experts)[0]
    elif ep_size > 1 and (num_experts == 0 or num_experts % ep_size):
        raise ValueError(
            f"ep_size={ep_size} must divide the routed experts {hf_config.model_type} shards "
            f"({num_experts})."
        )
    if dense == "auto":
        # Every rank holds a whole copy of what expert parallelism does not shard.
        sharded = (
            {id(p) for m in experts.values() for p in m.parameters()} if ep_size > 1 else set()
        )
        replicated = sum(
            p.numel() * p.element_size() for p in model.parameters() if id(p) not in sharded
        )
        memory = _gpu_memory()
        too_big = memory is not None and replicated > _REPLICATED_SHARE * memory
        dense = "fsdp" if too_big else "replicate"
    return DistributionPlan(world_size, ep_size, dense)


def resolve_distribution_plan(
    ckpt_path: str,
    *,
    trust_remote_code: bool = False,
    ep_size: int | str | None = None,
    dense: str | None = None,
) -> tuple[DistributionPlan, PretrainedConfig]:
    """Decide how this run distributes ``ckpt_path``'s model over its ranks.

    The one place a run decides it. :func:`load_pretrained` loads the model by the plan this returns
    and records it on the model, and every later step reads it back from there
    (:func:`~modelopt.torch.utils.distributed.distribution_plan`). Returns ``(plan, hf_config)``,
    ``hf_config`` being the config a sharded load takes back as
    ``load_pretrained(ckpt_path, distribution=plan, hf_config=hf_config)``.

    Rank 0 decides, from the model built on the meta device: from its parameters rather than
    estimates of them, and from its routed experts as an expert-parallel load finds them. Every rank
    takes rank 0's plan (:func:`~modelopt.torch.utils.distributed.broadcast_from`): the inputs are
    read locally, ranks deciding for themselves could disagree, and ranks loading different plans
    hang. Collective: every rank must call it.

    * ``ep_size``: the expert-parallel degree, or ``"auto"``, the largest that divides both the
      world and the routed-expert count of a model whose config carries a ``base_model_ep_plan``.
    * ``dense``: ``"replicate"`` or ``"fsdp"`` for the rest of the model, or ``"auto"``, which
      FSDP2-shards it when a whole copy would take more than 60% of one GPU's memory. That GPU is
      rank 0's, by its total memory, so ``"auto"`` assumes the run's GPUs are alike.

    ``None`` reads ``MODELOPT_EP_DEGREE`` and ``MODELOPT_FSDP_SIZE`` (``1`` replicates; ``world``,
    or the world size as a number, shards) from rank 0's environment, both defaulting to
    ``"auto"``. A single process reads neither and holds the whole model, unless ``dense="fsdp"``
    asks for FSDP2 on one rank.
    """
    from modelopt.torch.utils import distributed as dist

    # Each rank reads the checkpoint for itself. One that cannot must fail every rank here, rather
    # than leave the others waiting in a broadcast below for a rank that already raised.
    read: list[tuple[str, PretrainedConfig]] = []

    def checkpoint_read() -> bool:
        _, local_path = ensure_local_checkpoint(ckpt_path)
        config = AutoConfig.from_pretrained(local_path, trust_remote_code=trust_remote_code)
        read.append((local_path, config))
        return True

    dist.all_reduce("checkpoint read", checkpoint_read, all)
    ckpt_path, hf_config = read[0]
    world_size = dist.size()
    plan = dist.broadcast_from(
        "plan",
        lambda: _decide_distribution(
            ckpt_path, hf_config, world_size, ep_size, dense, trust_remote_code
        ),
    )
    if plan != DistributionPlan() and dist.is_master():
        print(f"Distribution: {plan!r}", flush=True)
    return plan, hf_config
