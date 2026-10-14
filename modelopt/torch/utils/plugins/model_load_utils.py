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

"""HuggingFace model loading: :func:`load_pretrained`, by a plan to distribute the model."""

import contextlib
import copy
import logging
import os
import re
from itertools import chain
from typing import TYPE_CHECKING, Any
from warnings import warn

import torch
import torch.nn as nn
from huggingface_hub import snapshot_download

if TYPE_CHECKING:
    from collections.abc import Iterable
from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict
from torch.distributed.fsdp import CPUOffloadPolicy, fully_shard
from torch.distributed.tensor import DTensor
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, PretrainedConfig
from transformers.conversion_mapping import get_model_conversion_mapping
from transformers.core_model_loading import WeightConverter, dot_natural_key, rename_source_key

from modelopt.torch.utils import env
from modelopt.torch.utils.distributed import (
    DistributionPlan,
    _move_to_fsdp_device,
    _off_dtype_params,
    barrier,
    broadcast_state_dict,
    is_initialized,
    rank,
    set_distribution_plan,
)
from modelopt.torch.utils.logging import print_rank_0
from modelopt.torch.utils.plugins.hf_checkpoint_utils import (
    checkpoint_tensor_files,
    ensure_local_checkpoint,
    indexed_weight_map,
    read_safetensors_subset,
)

__all__ = [
    "build_meta_causal_lm",
    "fsdp2_wrap",
    "load_and_prepare_ep",
    "load_pretrained",
    "parallel_load_and_prepare_fsdp2",
    "record_unplaced_source_keys",
    "resolve_distribution_plan",
    "unplaced_source_keys",
]

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
    # Every name of a tied parameter: a checkpoint can store the one named_parameters() drops by
    # default (Qwen3-0.6B stores lm_head.weight beside the embedding it is tied to).
    model_param_names = {
        n
        for n, _ in chain(
            model.named_parameters(remove_duplicate=False),
            model.named_buffers(remove_duplicate=False),
        )
    }
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


_FSDP2 = "FSDP2"
# What each sharded load takes besides the checkpoint. These loads build the model class and place
# it themselves, so from_pretrained's own arguments (device_map, max_memory, ...) do not reach them.
_SHARDED_LOAD_OPTIONS = {
    _FSDP2: frozenset(
        {
            "trust_remote_code",
            "attn_implementation",
            "hf_config",
            "cpu_offload",
            "mp_policy",
            "broadcast_chunk_size",
        }
    ),
}


def _route(plan: DistributionPlan) -> str | None:
    """The sharded load ``plan`` takes, or ``None`` when every rank loads the whole model."""
    from modelopt.torch.utils import distributed as dist

    if plan.world_size != dist.size():
        raise ValueError(
            f"{plan!r} is for {plan.world_size} ranks, but this run has {dist.size()}."
        )
    if plan.is_expert_parallel:
        raise NotImplementedError(
            f"{plan!r} shards the routed experts, which load_pretrained has no load for. Plan "
            "ep_size=1 (MODELOPT_EP_DEGREE=1) to give every rank the whole model, or shard it "
            "with FSDP2."
        )
    return _FSDP2 if plan.is_fsdp_dense else None


def _load_sharded(route, ckpt_path, auto_class, model_type, kwargs, hub_options) -> nn.Module:
    """The sharded load ``route`` names, refusing what it cannot honor before it starts."""
    from modelopt.torch.models.hf import checkpoint_has_mtp
    from modelopt.torch.utils import distributed as dist

    if auto_class not in (None, AutoModelForCausalLM):
        raise NotImplementedError(
            f"The {route} load builds its model class itself; auto_class={auto_class.__name__} "
            "does not apply to it."
        )
    kwargs = dict(kwargs)
    if kwargs.get("dtype", "auto") == "auto":
        kwargs.pop("dtype", None)  # A sharded load keeps the checkpoint's dtype.
    unknown = sorted(set(kwargs) - _SHARDED_LOAD_OPTIONS[route])
    if unknown:
        raise TypeError(
            f"{unknown} do not apply to the {route} load, which places the model itself; it "
            f"takes {sorted(_SHARDED_LOAD_OPTIONS[route])}."
        )
    if checkpoint_has_mtp(model_type, ckpt_path):
        # prepare_model_for_loading builds the head on a whole model only; carried over, the
        # head would be exported unquantized.
        raise NotImplementedError(
            f"The {route} load cannot build the MTP head this {model_type} checkpoint carries; "
            "load it unsharded."
        )
    trust_remote_code = kwargs.pop("trust_remote_code", False)
    hf_config = kwargs.pop("hf_config", None)
    if hf_config is None:
        hf_config = AutoConfig.from_pretrained(
            ckpt_path, trust_remote_code=trust_remote_code, **hub_options
        )
    if getattr(hf_config, "quantization_config", None) is not None:
        # It reads the tensors as stored: quantized weights would load as if they were not.
        raise NotImplementedError(
            f"The FSDP2 load cannot dequantize this {model_type} checkpoint; load it unsharded."
        )
    if not dist.is_initialized():
        # FSDP2 shards through a process group even on one rank; a plan for more ranks than one
        # was refused without a group. A run (distributed_run) destroys the group on exit.
        dist.setup()
    return parallel_load_and_prepare_fsdp2(
        ckpt_path,
        torch.device("cuda", dist.local_rank()),
        dist.rank(),
        dist.size(),
        trust_remote_code=trust_remote_code,
        hf_config=hf_config,
        **kwargs,
    )


def load_pretrained(
    ckpt_path: str,
    auto_class: Any = None,
    *,
    model_type: str | None = None,
    distribution: DistributionPlan | None = None,
    **kwargs,
) -> nn.Module:
    """Load a Hugging Face model for ModelOpt to optimize and export, whole or distributed.

    ``distribution`` is the :class:`~modelopt.torch.utils.distributed.DistributionPlan` to spread
    the model over the run's ranks by, :func:`resolve_distribution_plan`'s by default. By the plan:

    * ``dense="fsdp"``: a causal LM FSDP2-sharded over the whole process group, each rank reading its
      own layers (:func:`parallel_load_and_prepare_fsdp2`, whose ``cpu_offload``, ``mp_policy`` and
      ``broadcast_chunk_size`` ``kwargs`` may set). A single process has no group to shard through,
      so the load sets up one of one rank, which
      :func:`~modelopt.torch.utils.distributed.distributed_run` destroys with the run. It reads the
      checkpoint's tensors as stored, so it refuses a quantized checkpoint.
    * otherwise ``auto_class.from_pretrained(ckpt_path, **kwargs)`` (``auto_class`` defaults to
      ``AutoModelForCausalLM``), placed as ``kwargs`` say. Every rank of a distributed run gets the
      whole model, ``device_map`` defaulting to the rank's GPU.

    A plan with ``ep_size > 1`` raises: no load here shards the routed experts. On a distributed run
    every rank loads the plan rank 0 was handed. The plan is recorded on the model
    (:func:`~modelopt.torch.utils.distributed.set_distribution_plan`), where calibration and export
    read it back.

    A sharded load builds the model class and places it itself. Besides its own options it takes
    ``trust_remote_code``, ``attn_implementation`` and ``hf_config``, and refuses any other keyword
    rather than ignore it. ``hf_config`` is the config :func:`resolve_distribution_plan` returned with
    the plan handed in as ``distribution``: the load builds from it as given, so without that plan
    it is refused. A ``config`` passed for the load is the one the plan is decided from, and a
    sharded load takes it as its ``hf_config``.

    Whatever the placement:

    * A Hub model ID is made local first (:func:`ensure_local_checkpoint`, which takes the
      ``revision``, ``token``, ``cache_dir`` and ``local_files_only`` ``kwargs``): the export reads
      the source checkpoint from disk. Under ``torch.distributed`` every rank must call this. The
      config and model loads still take all but ``revision``: an ``auto_map`` entry can name
      custom code in another Hub repo, which Transformers fetches with them. A ``subfolder`` is
      part of where the checkpoint is: the config, the loading hook and the record below read it,
      as the load does.
    * The model type's loading hook runs
      (:func:`~modelopt.torch.models.hf.prepare_model_for_loading`), e.g. to keep the MTP head a
      Nemotron-H checkpoint carries. ``model_type`` selects it; when omitted, the type of the
      config passed for the load, else of the checkpoint's config. A sharded load cannot run it, so
      it refuses a checkpoint the hook would build modules for.
    * The checkpoint tensors the model has no parameter for are recorded
      (:func:`record_unplaced_source_keys`), so the export carries them over unchanged. The model
      is returned alone: ``output_loading_info=True`` is refused. So is a ``variant``: the record
      and the export read the checkpoint's default weight files.
    """
    if kwargs.pop("output_loading_info", False):
        raise ValueError(
            "load_pretrained returns the model alone and does not take output_loading_info=True; "
            "the checkpoint tensors the model has no parameter for are recorded on it instead "
            "(see record_unplaced_source_keys)."
        )
    if kwargs.get("variant") is not None:
        raise ValueError(
            f"load_pretrained does not take variant={kwargs['variant']!r}: the checkpoint tensors "
            "the model has no parameter for are recorded from, and carried into the export from, "
            "the checkpoint's default weight files."
        )
    # Importing modelopt.torch.models registers every model spec; only callers of this need them.
    from modelopt.torch.models.hf import prepare_model_for_loading
    from modelopt.torch.utils import distributed as dist

    hub_options = {
        k: kwargs.pop(k) for k in ("token", "cache_dir", "local_files_only") if k in kwargs
    }
    revision = {"revision": kwargs.pop("revision")} if "revision" in kwargs else {}
    _, ckpt_path = ensure_local_checkpoint(ckpt_path, **revision, **hub_options)
    subfolder = kwargs.pop("subfolder", None)
    if subfolder:
        ckpt_path = os.path.join(ckpt_path, subfolder)
    trust_remote_code = kwargs.get("trust_remote_code", False)
    config = kwargs.get("config")
    config = config if isinstance(config, PretrainedConfig) else None
    if config is not None and "hf_config" in kwargs:
        raise TypeError("config and hf_config both name the config to load with; pass one.")
    if distribution is None and "hf_config" in kwargs:
        # The load builds from hf_config as given, so the plan must be the one it came with, not
        # one decided here from another config.
        raise TypeError(
            "hf_config is the config resolve_distribution_plan returned with a plan: pass that plan "
            "as distribution=, or pass config= to have the plan decided from your config."
        )
    hf_config = kwargs.get("hf_config")
    if distribution is None:
        plan, hf_config = resolve_distribution_plan(
            ckpt_path, config=config, trust_remote_code=trust_remote_code, **hub_options
        )
        if _route(plan) is not None:
            kwargs.setdefault("hf_config", hf_config)
    elif isinstance(distribution, DistributionPlan):
        # Every rank loads the plan rank 0 was handed, as it would one it decided: ranks loading
        # different plans would build mismatched meshes and hang. Rank 0 checks it before it sends,
        # so a plan it refuses raises on every rank instead of leaving them in the broadcast.
        handed = distribution

        def checked() -> DistributionPlan:
            _route(handed)
            return handed

        plan = dist.broadcast_from("plan", checked)
    else:
        raise TypeError(f"distribution={distribution!r}: expected a DistributionPlan or None.")
    route = _route(plan)
    if route is not None and config is not None:
        # The sharded load builds the model from hf_config: the plan's copy of this one, or it.
        del kwargs["config"]
        kwargs.setdefault("hf_config", config)
    if model_type is None:
        # The load uses the config passed for it, if any, so the loading hook follows that one.
        model_type = getattr(hf_config or config, "model_type", None) or (
            AutoConfig.from_pretrained(
                ckpt_path, trust_remote_code=trust_remote_code, **hub_options
            ).model_type
        )
    if route is not None:
        model = _load_sharded(route, ckpt_path, auto_class, model_type, kwargs, hub_options)
        # A sharded load finds the unplaced keys while it places the weights.
        keys = model._modelopt_unplaced_source_keys
    else:
        if "hf_config" in kwargs:
            raise TypeError("hf_config applies to a sharded load only.")
        if plan.world_size > 1:
            # Each rank takes the whole model onto its own GPU. Left to "auto", every rank's
            # accelerate could pick the same idle GPU.
            kwargs.setdefault("device_map", {"": f"cuda:{dist.local_rank()}"})
        auto_class = auto_class or AutoModelForCausalLM
        with prepare_model_for_loading(
            model_type,
            ckpt_path,
            trust_remote_code,
            model_class=None if auto_class in (AutoModel, AutoModelForCausalLM) else auto_class,
        ):
            model = auto_class.from_pretrained(ckpt_path, **hub_options, **kwargs)
        keys = None
    set_distribution_plan(model, plan)
    unplaced = record_unplaced_source_keys(model, ckpt_path, keys)
    if unplaced:
        print_rank_0(
            f"{len(unplaced)} checkpoint tensor(s) the model has no parameter for "
            f"(e.g. {unplaced[0]}); the export will carry them over unchanged."
        )
    return model


def fsdp2_wrap(model, shard_root=True, mp_policy=None, cpu_offload: bool = False):
    """Auto-detect a HF causal-LM's decoder layers and FSDP2 ``fully_shard`` each one.

    By default (``shard_root=True``) the root module is wrapped too, so embed/lm_head/norm are
    sharded instead of replicated per rank; pass ``shard_root=False`` to leave the root replicated
    (only decoder layers sharded). Returns the detected decoder layers so callers can reuse the
    detection result.

    Parameters whose dtype differs from the model's dominant one are excluded from the wrap (see
    :func:`~modelopt.torch.utils.distributed._off_dtype_params`), since FSDP2 rejects a shard group
    that mixes dtypes. They stay replicated and are moved onto the shards' device, which
    ``fully_shard`` does not do for the params it ignores.

    It records no distribution plan: :func:`load_pretrained` records the plan it loads by, and a
    caller that wraps a model itself records one
    (:func:`~modelopt.torch.utils.distributed.set_distribution_plan`) before calibrating or
    exporting it.
    """
    # Lazy import: layerwise_calib imports this module at top level (circular).
    from modelopt.torch.quantization.utils.layerwise_calib import LayerActivationCollector

    decoder_layers = LayerActivationCollector.get_decoder_layers(model)
    if decoder_layers is None:
        raise RuntimeError(
            "Could not auto-detect decoder layers; FSDP2 wrap requires a standard HF causal-LM layout."
        )

    fsdp_kwargs: dict[str, Any] = {"reshard_after_forward": True}
    if mp_policy is not None:
        fsdp_kwargs["mp_policy"] = mp_policy
    if cpu_offload:
        fsdp_kwargs["offload_policy"] = CPUOffloadPolicy()
    ignored_params = _off_dtype_params(model)
    if ignored_params:
        fsdp_kwargs["ignored_params"] = ignored_params

    # Snapshot/restore config.architectures: some HF builders mutate it during fully_shard.
    config = getattr(model, "config", None)
    architectures = list(getattr(config, "architectures", []) or [])
    for layer in decoder_layers:
        fully_shard(layer, **fsdp_kwargs)
    if shard_root:
        fully_shard(model, **fsdp_kwargs)
    if ignored_params:
        _move_to_fsdp_device(model, ignored_params)
    if config is not None and architectures:
        config.architectures = architectures
    return decoder_layers


def _map_ckpt_keys_to_model(model: "torch.nn.Module", ckpt_keys, model_keys) -> dict:
    """Map each checkpoint key to the model ``state_dict`` key it feeds.

    Map each CHECKPOINT key to the built model's ``state_dict`` key it feeds, reproducing what
    ``from_pretrained`` resolves on load -- so a DCP by-name load fills the right tensors.

    Three passes per checkpoint key (first hit wins): (1) exact match (the common case); (2) the model's
    ``_checkpoint_conversion_mapping`` regexes if it carries any (first pattern that fires); (3) longest
    UNIQUE trailing-component match. Pass 3 covers the transformers>=5 VLM prefix-nesting
    (``model.layers.*`` checkpoint -> ``model.language_model.layers.*`` model, ``visual.*`` -> ``model.visual.*``)
    WITHOUT hard-coding it: the leaf path is preserved, so the ckpt key's component suffix uniquely locates
    the model key. Ambiguous (>1 candidate) or absent suffixes are left unmapped -> caught by the caller's
    coverage guard rather than mis-assigned.

    ``model_keys`` must be the WHOLE model's keys, even when only some of them will be loaded: a suffix is
    unique only if no other model key shares it, and the caller drops the mappings it does not load. Each
    model key takes at most one checkpoint key -- a later pass never maps onto a key an earlier pass
    claimed, and two checkpoint keys claiming one key in the same pass are both left unmapped. Otherwise a
    key the model has no place for (an MTP layer's ``model.layers.61.shared_head.norm.weight``, a layer a
    truncated model lacks) lands on ``model.norm.weight`` by suffix alone and overwrites it, with set order
    picking the winner.
    """
    from collections import defaultdict

    key_map: dict = {}
    remaining = set(ckpt_keys)

    def claim(hits: dict) -> None:
        claimed = set(key_map.values())
        for mk, cks in hits.items():
            if len(cks) == 1 and mk not in claimed:
                key_map[cks[0]] = mk
            remaining.difference_update(cks)

    # Pass 1: exact
    claim({ck: [ck] for ck in remaining if ck in model_keys})
    # Pass 2: model-declared regex conversion mapping (rare; e.g. leaf renames)
    conv = getattr(model, "_checkpoint_conversion_mapping", None) or {}
    if conv and remaining:
        compiled = [(re.compile(p), r) for p, r in conv.items()]
        hits: dict = defaultdict(list)
        for ck in remaining:
            mk = ck
            for rx, repl in compiled:
                mk, n = rx.subn(repl, mk)
                if n:
                    break
            if mk in model_keys:
                hits[mk].append(ck)
        claim(hits)
    # Pass 3: longest unique component-suffix match. Candidates only grow as the suffix shortens, so the
    # longest suffix with any candidate decides.
    if remaining:
        suffix_idx: dict = defaultdict(list)
        for mk in model_keys:
            comps = mk.split(".")
            for k in range(1, len(comps) + 1):
                suffix_idx[tuple(comps[-k:])].append(mk)
        hits = defaultdict(list)
        for ck in remaining:
            comps = ck.split(".")
            for k in range(len(comps), 0, -1):
                cands = suffix_idx.get(tuple(comps[-k:]))
                if cands:
                    if len(cands) == 1:
                        hits[cands[0]].append(ck)
                    break
        claim(hits)
    return key_map


def _copy_tied_values(kept: torch.Tensor, src: torch.Tensor, name: str) -> None:
    """Copy ``name``'s checkpoint values from ``src`` into ``kept``, the tied param that replaced it.

    The copy is local: plain tensors, or DTensors laid out alike (one mesh, the same placements), as
    FSDP2 shards tied params. Any other layout would need a redistribution and is refused instead.
    """
    from torch.distributed.tensor import DTensor

    alike = kept.shape == src.shape and (
        not (isinstance(kept, DTensor) or isinstance(src, DTensor))
        or (
            isinstance(kept, DTensor)
            and isinstance(src, DTensor)
            and kept.device_mesh == src.device_mesh
            and kept.placements == src.placements
        )
    )
    if not alike:
        raise RuntimeError(
            f"meta+DCP load: tie_weights replaced {name}, which the checkpoint filled, with a param "
            "laid out differently, so its values cannot be carried over"
        )
    with torch.no_grad():
        if isinstance(kept, DTensor):
            kept, src = kept.to_local(), src.to_local()
        kept.copy_(src)


def _meta_dcp_load(
    model: "torch.nn.Module",
    local_dir: str,
    device,
    *,
    load_names: set,
) -> None:
    """Materialize a meta-built model's ``load_names`` params and fill them through torch DCP.

    Each rank materializes the params on ``device`` and fills them from the HF safetensors via DCP's
    ``HuggingFaceStorageReader``.

    PLACEMENT is decided entirely by the CALLER, not here: ``dcp.load`` reads a FULL tensor into a plain
    (replicated) param and only the LOCAL SLICE into a ``Shard`` ``DTensor``. The EP hybrid ``fully_shard``-s
    its dense modules first, so each rank loads only its dense shard, and no full copy is made.

    ``load_names`` restricts the checkpoint load to those ``state_dict`` names -- the dense params in the EP
    hybrid, whose experts are ALREADY materialized by ``from_pretrained`` and must not be touched.

    Materialization is SELECTIVE (module-by-module, only where params are meta) so already-real params (EP
    experts) are never re-emptied. ``to_empty`` also re-empties BUFFERS (rotary ``inv_freq`` etc., + any the
    caller already loaded), so they are snapshotted and restored -- a params-only checkpoint load would
    otherwise leave them garbage and the calibration forward would drift (weights load fine but activation
    amax is wrong). Persistent buffers ``load_names`` names are (re)loaded from the checkpoint via DCP below.

    Every target param ends up filled -- from the checkpoint, by ``tie_weights``, or with the value it
    already had -- or the load fails: nothing is left as the uninitialized memory ``to_empty`` hands out.
    """
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint import HuggingFaceStorageReader

    # Snapshot real buffers (to_empty re-empties them), materialize params, restore buffers. Materialize
    # selectively, module by module, so the real experts are not re-emptied. That is safe only because no
    # root is fully_shard-ed: a per-module loop corrupts a ROOT-sharded FSDP state (its state_dict then
    # returns wrong keys).
    saved_buffers = {n: b.detach().clone() for n, b in model.named_buffers() if not b.is_meta}
    # A module's ``to_empty`` re-empties every direct param, so a param already real beside a meta one
    # (a router's ``e_score_correction_bias`` next to its meta ``weight``) is kept here too: nothing
    # outside ``load_names`` would refill it.
    saved_params: dict[str, torch.Tensor] = {}
    emptied: set[str] = set()
    for name, m in model.named_modules():
        direct = dict(m.named_parameters(recurse=False))
        if any(p.is_meta for p in direct.values()):
            prefix = f"{name}." if name else ""
            saved_params.update(
                (prefix + n, p.detach().clone()) for n, p in direct.items() if not p.is_meta
            )
            emptied.update(prefix + n for n in direct)
            m.to_empty(device=device, recurse=False)
    with torch.no_grad():
        for n, b in model.named_buffers():
            src = saved_buffers.get(n)
            if src is not None:
                b.copy_(src.to(device=b.device, dtype=b.dtype))
        for n, src in saved_params.items():
            p = model.get_parameter(n)
            p.copy_(src.to(device=p.device, dtype=p.dtype))

    # ``dcp.load`` matches state_dict entries to checkpoint entries BY NAME, but a checkpoint's stored key
    # names can differ from the built model's ``state_dict`` names, and from_pretrained bridges the gap on
    # load. E.g. Qwen2.5-VL's checkpoint stores ``model.layers.*`` / ``visual.*`` while transformers>=5 nests
    # them as ``model.language_model.layers.*`` / ``model.visual.*`` -- a pure prefix-insertion with the leaf
    # path preserved, and there is NO ``_checkpoint_conversion_mapping`` on the model to read. A raw by-name
    # match would SILENTLY skip every renamed weight (they stay ``to_empty`` garbage -> a wrecked model that
    # only surfaces as a confusing downstream crash). Reproduce from_pretrained's result generically via
    # :func:`_map_ckpt_keys_to_model`: exact -> conversion-mapping regexes -> longest UNIQUE suffix.
    reader = HuggingFaceStorageReader(str(local_dir))
    ckpt_keys = set(reader.read_metadata().state_dict_metadata.keys())
    if not ckpt_keys:
        raise RuntimeError(
            f"meta+DCP load: the checkpoint at {local_dir} has NO tensors -- the resolved snapshot carries no "
            "safetensors (a config-only / incomplete HF revision; e.g. refs/main pointing at a weightless "
            "snapshot). Re-download the checkpoint or pass a complete local snapshot dir / pinned revision."
        )
    full_sd = (
        model.state_dict()
    )  # (sharded ``DTensor`` or plain) params + replicated persistent buffers
    target_names = set(load_names) & set(full_sd)
    # {ckpt_key: model_key}, mapped against the WHOLE model and then cut to what this load fills: a suffix
    # unique among the targets alone can still belong to a key outside them (an expert's, when only the
    # dense part loads here).
    key_map = {
        ck: mk
        for ck, mk in _map_ckpt_keys_to_model(model, ckpt_keys, set(full_sd)).items()
        if mk in target_names
    }
    # Build load_sd in the model's state_dict ORDER (identical on every rank) -- NOT key_map/set iteration
    # order, which is hash-randomized per process. dcp.load's collective load planner deadlocks if ranks
    # present the same keys in different orders, so a deterministic, rank-consistent order is required.
    inv = {mk: ck for ck, mk in key_map.items()}  # model_key -> ckpt_key (1:1 on model keys)
    load_sd = {inv[mk]: full_sd[mk] for mk in full_sd if mk in inv}
    mapped_names = set(key_map.values())
    # Fail fast, BEFORE the load, if the mapping left over half of the TARGET WEIGHT unloaded -- a key scheme
    # this path did not resolve. Weight by NUMEL, not param COUNT: a fused-expert MoE (Qwen3-MoE: per-expert
    # checkpoint -> grouped ``experts.gate_up_proj``/``down_proj`` model params DCP can't fuse) leaves only
    # ~2 params/layer unloaded but they are ~95% of the weight, so a count-based guard misses it. A smaller
    # shortfall can be tied weights, which ``tie_weights`` fills after the load; whatever it does not fill
    # fails there.
    numel = {n: p.numel() for n, p in model.named_parameters() if n in target_names}
    want_params = set(numel)
    unloaded = want_params - mapped_names
    total_nel = max(1, sum(numel.values()))
    unloaded_nel = sum(numel[n] for n in unloaded)
    if 2 * unloaded_nel > total_nel:
        raise RuntimeError(
            f"meta+DCP load: {len(unloaded)}/{len(want_params)} params ({100 * unloaded_nel / total_nel:.0f}% "
            f"of the target weight) matched no checkpoint key -- the checkpoint's key naming was not resolved "
            f"to this model's state_dict (exact + conversion-mapping + suffix match all missed), or it needs a "
            f"weight conversion DCP can't do (FUSED EXPERTS, split qkv, on-the-fly dequant). Unloaded e.g.: "
            f"{sorted(unloaded)[:5]}. Replicating the dense part (MODELOPT_FSDP_SIZE=1) loads it through "
            "from_pretrained instead."
        )
    if is_initialized() and rank() == 0:
        loaded_bufs = mapped_names - {n for n, _ in model.named_parameters()}
        if loaded_bufs:
            warn(
                f"meta+DCP load: loaded {len(loaded_bufs)} persistent buffer(s) from checkpoint: "
                f"{sorted(loaded_bufs)[:5]}"
            )
    dcp.load(load_sd, storage_reader=reader)
    # Taken before tying, which can swap a filled param out of its module.
    filled = {n: model.get_parameter(n) for n in want_params & mapped_names}
    if hasattr(model, "tie_weights"):
        model.tie_weights()  # re-establish tied weights the checkpoint omitted
    # Tying keeps one param of each tied group, not necessarily the one the checkpoint filled: a tied
    # checkpoint storing only ``lm_head.weight`` loses its values to the embedding's empty param. The kept
    # param takes the filled one's values, as from_pretrained leaves it.
    loaded = {id(p) for p in filled.values()}
    for name, src in filled.items():
        kept = model.get_parameter(name)
        if id(kept) not in loaded:
            _copy_tied_values(kept, src, name)
            loaded.add(id(kept))
    # A re-emptied target param no checkpoint tensor filled, and that tying did not make a loaded param,
    # still holds ``to_empty``'s memory. from_pretrained would initialize it; this loader can't run the
    # model's init on shards, so it fails rather than return the model. Every rank reaches the same verdict.
    unfilled = sorted(
        n
        for n in (unloaded & emptied) - saved_params.keys()
        if id(model.get_parameter(n)) not in loaded
    )
    if unfilled:
        raise RuntimeError(
            f"meta+DCP load: {len(unfilled)} param(s) have no checkpoint tensor and are tied to no loaded "
            f"param, so they would stay uninitialized: {unfilled[:5]}"
        )


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

    It records no distribution plan, as :func:`fsdp2_wrap` does not; :func:`load_pretrained` does.
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
    # The expert-parallel load shards the dense part through Distributed Checkpoint, which cannot
    # dequantize a block-fp8 checkpoint's F8_E8M0 scales.
    unshardable = ep_size > 1 and _ckpt_quant_method(hf_config) == "fp8"
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
        if too_big and unshardable:
            print(
                f"{hf_config.model_type}: the dense part would take more than "
                f"{_REPLICATED_SHARE:.0%} of a GPU, but a block-fp8 checkpoint's cannot be sharded "
                "expert-parallel; replicating it.",
                flush=True,
            )
        dense = "fsdp" if too_big and not unshardable else "replicate"
    elif dense == "fsdp" and unshardable:
        raise NotImplementedError(
            f"The dense part of a block-fp8 {hf_config.model_type} checkpoint cannot be sharded "
            "expert-parallel: Distributed Checkpoint cannot dequantize its F8_E8M0 scales."
        )
    return DistributionPlan(world_size, ep_size, dense)


def resolve_distribution_plan(
    ckpt_path: str,
    *,
    config: PretrainedConfig | None = None,
    trust_remote_code: bool = False,
    ep_size: int | str | None = None,
    dense: str | None = None,
    token: str | bool | None = None,
    cache_dir: str | os.PathLike | None = None,
    local_files_only: bool = False,
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

    ``token``, ``cache_dir`` and ``local_files_only`` are what resolving a Hub model ID
    (:func:`ensure_local_checkpoint`) and reading the config take, the config for custom code its
    ``auto_map`` names in another Hub repo.

    ``config``, the config the load will use when the caller has one, is decided from instead of
    the checkpoint's, which is then not read: each rank takes a copy, as deciding may add to it.
    """
    from modelopt.torch.utils import distributed as dist

    # Each rank reads the checkpoint for itself. One that cannot must fail every rank here, rather
    # than leave the others waiting in a broadcast below for a rank that already raised.
    read: list[tuple[str, PretrainedConfig]] = []

    def checkpoint_read() -> bool:
        _, local_path = ensure_local_checkpoint(
            ckpt_path, token=token, cache_dir=cache_dir, local_files_only=local_files_only
        )
        if config is not None:
            read.append((local_path, copy.deepcopy(config)))
            return True
        hf_config = AutoConfig.from_pretrained(
            local_path,
            trust_remote_code=trust_remote_code,
            token=token,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        read.append((local_path, hf_config))
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


def _make_expert_param_matcher(ep_plan: dict[str, Any]):
    """Build ``is_expert(param_name) -> bool`` from the model's ``base_model_ep_plan``.

    the plan that actually drives EP sharding -- instead of a fragile ``"expert" in name`` substring heuristic.

    A param is EP-sharded (only its ``1/ep_degree`` shard resides on this rank) iff its name, with layer
    indices wildcarded exactly as HuggingFace does (``replace_layer_number_by_wildcard``) and any trailing
    ``.weight`` / ``.bias`` stripped, matches an expert plan key as a SUFFIX (tolerating the ``model.`` /
    ``language_model.model.`` prefixes the base-relative plan keys omit) OR is NESTED under one. This
    covers both per-parameter shard keys (``...experts.gate_up_proj: grouped_gemm``) and experts-container
    keys (``...experts: torch_a2a_experts`` / ``nemotron_experts_ep``). Router entries
    (``ep_router`` / ``torch_a2a_router``) are REPLICATED, not sharded, so they're excluded. If the plan is
    empty the matcher is always ``False`` (-> no EP credit -> stock warmup sizing).
    """
    from transformers.integrations.tensor_parallel import replace_layer_number_by_wildcard

    _router_styles = {"ep_router", "torch_a2a_router"}
    keys = [k for k, v in (ep_plan or {}).items() if v not in _router_styles]

    def is_expert(pname: str) -> bool:
        generic = replace_layer_number_by_wildcard(pname)
        for suffix in (".weight", ".bias"):
            if generic.endswith(suffix):
                generic = generic[: -len(suffix)]
                break
        padded = "." + generic + "."
        return any(
            generic == k or generic.endswith("." + k) or ("." + k + ".") in padded for k in keys
        )

    return is_expert


@contextlib.contextmanager
def _ep_aware_caching_allocator_warmup(
    ep_degree: int, ep_plan: dict[str, Any], dense_on_meta: bool = False
):
    """Make transformers' ``caching_allocator_warmup`` EP-aware for the duration of ``from_pretrained``.

    transformers sizes the load-time allocator warmup via ``get_total_byte_count``, which credits the
    ``// world_size`` sharding division only to params matched in ``model._tp_plan``. Under the EP load
    (``device_mesh`` + ``enable_expert_parallel``; sharding driven by ``base_model_ep_plan``) ``_tp_plan``
    is EMPTY, so every param -- experts are ~95% of a large MoE -- is counted at FULL global size. The
    warmup then pre-allocates one ``torch.empty`` clamped to ``device_total - 1.2 GiB`` (~the whole GPU)
    on every rank; torch keeps that as reserved cache, which later starves the NCCL a2a comm buffers.

    This temporarily replaces ``get_total_byte_count`` with a version that additionally credits EP-sharded
    expert params their per-rank shard (``// ep_degree``), identifying them from ``ep_plan`` (the model's
    ``base_model_ep_plan``), sizing the warmup to the real per-rank footprint (e.g. ~91 GiB instead of
    ~275 GiB for Nemotron-3-Ultra-550B EP32) -- keeping the warmup's load-speed benefit without the
    ~device-max transient.

    Always on (no disable knob): it only *sizes* an allocator pre-reservation, never model correctness; it
    falls back to stock sizing on any incompatibility (``try/except``); and a mis-estimate only nudges load
    speed (an over-credit under-sizes -> the allocator just grows during load; an under-credit sizes up to
    the same device-max clamp as stock) -- so it can never OOM worse than, or load differently from, stock.
    No-op when ``ep_degree <= 1``.
    """
    if ep_degree <= 1:
        yield
        return
    try:
        import transformers.modeling_utils as _mu
    except Exception:
        yield
        return
    orig = getattr(_mu, "get_total_byte_count", None)
    if orig is None:
        yield
        return

    is_expert = _make_expert_param_matcher(ep_plan)

    def _ep_aware(model, accelerator_device_map, hf_quantizer=None):
        try:
            from collections import defaultdict

            try:
                from transformers.integrations.tensor_parallel import _get_parameter_tp_plan
            except Exception:
                _get_parameter_tp_plan = None
            ws = (
                torch.distributed.get_world_size()
                if torch.distributed.is_available() and torch.distributed.is_initialized()
                else 1
            )
            tp_plan = (getattr(model, "_tp_plan", None) or {}) if ws > 1 else {}
            tied = set(getattr(model, "all_tied_weights_keys", {}) or {})
            total: dict = defaultdict(int)
            for pname, dev in accelerator_device_map.items():
                if pname in tied:
                    continue
                try:
                    p = model.get_parameter_or_buffer(pname)
                except Exception:
                    continue
                dsize = (
                    hf_quantizer.param_element_size(model, pname, p)
                    if hf_quantizer is not None
                    else p.element_size()
                )
                nb = p.numel() * dsize
                if is_expert(pname):
                    nb //= ep_degree  # EP-sharded across the ep submesh: only the local shard resides here
                elif dense_on_meta and pname.endswith((".weight", ".bias")):
                    # A sharded dense part: _ep_dense_meta_load forces every dense .weight/.bias to
                    # materialize on META during from_pretrained (they are FSDP2-sharded + DCP-loaded
                    # afterwards), so they occupy NO device memory in the window this warmup sizes.
                    # Counting them at full global size is what pushed Kimi-K3 (EP64, dense=fsdp) to a
                    # 185.8 GiB estimate on a 184.31 GiB card -- over the device, so the warmup fell back
                    # to the device-max clamp it exists to avoid. Persistent buffers are NOT forced to
                    # meta by that context, so they keep their full credit here.
                    continue
                elif (
                    tp_plan
                    and _get_parameter_tp_plan is not None
                    and _get_parameter_tp_plan(pname, tp_plan, is_weight=True) is not None
                ):
                    nb //= ws  # preserve stock TP sharding credit for any tp-planned params
                total[dev] += nb
            if (
                not (torch.distributed.is_available() and torch.distributed.is_initialized())
                or torch.distributed.get_rank() == 0
            ):
                gib = 1024**3
                sizes = ", ".join(f"{d}: {b / gib:.1f} GiB" for d, b in total.items())
                print(
                    f"[MEM] EP-aware warmup sizing (experts // ep_degree={ep_degree}"
                    f"{', dense on meta -> uncounted' if dense_on_meta else ''}) -> {sizes} "
                    f"(was full-model ~device-max; EP-aware plan-driven sizing)",
                    flush=True,
                )
            return total
        except Exception:
            return orig(model, accelerator_device_map, hf_quantizer)

    try:
        _mu.get_total_byte_count = _ep_aware
        yield
    finally:
        _mu.get_total_byte_count = orig


def _refuse_unquantizable_experts(expert_modules) -> None:
    """Refuse routed experts that expert-parallel PTQ would quantize wrong.

    It quantizes the routed experts as they are, through the grouped GEMM of their ``torch_a2a``
    forward (as for DeepSeek, GLM and MiniMax). Experts stored transposed, ``(num_experts, in_dim,
    out_dim)`` like GPT-OSS's, that would leave unquantized. Experts ModelOpt quantizes through a
    wrapper of their own, like Llama4's, ``mtq.quantize`` would convert to it, and its forward drops
    the process group the all-to-all transport passes in. The generic fused-experts wrapper is no
    bar: it quantizes through the grouped GEMM, and ``mtq.quantize`` registers it, for the whole
    process, for every fused-experts class it meets, so a second load of the same class sees it.
    """
    transposed = sorted(
        {type(m).__name__ for m in expert_modules if getattr(m, "is_transposed", False)}
    )
    if transposed:
        raise NotImplementedError(
            f"Expert-parallel PTQ cannot quantize {', '.join(transposed)}: these experts store their "
            "weights transposed, which the grouped quantization leaves unquantized. Quantize the "
            "model without expert parallelism."
        )
    from modelopt.torch.quantization.nn.modules.quant_module import QuantModuleRegistry
    from modelopt.torch.quantization.plugins.huggingface import _QuantFusedExperts

    wrapped = sorted(
        {
            type(m).__name__
            for m in expert_modules
            if type(m) in QuantModuleRegistry
            and not issubclass(QuantModuleRegistry[type(m)], _QuantFusedExperts)
        }
    )
    if wrapped:
        raise NotImplementedError(
            f"Expert-parallel PTQ cannot quantize {', '.join(wrapped)}: ModelOpt quantizes these "
            "experts through a wrapper of their own, whose forward drops the expert-parallel "
            "transport. Quantize the model without expert parallelism."
        )


def _seed_parallel_states(model: nn.Module, expert_names, dense_state, expert_state) -> None:
    """Seed every module's ``_parallel_state`` for calibration's amax sync.

    The routed-expert containers named in ``expert_names`` and everything under them get
    ``expert_state``; every other module gets ``dense_state``. Seeding the plain modules works because
    quantization keeps a ``_parallel_state`` that is already present when it converts a module -- the same
    way Megatron's expert MLPs seed theirs.
    """
    prefixes = tuple(name + "." for name in expert_names)
    for name, module in model.named_modules():
        module._parallel_state = (
            expert_state if prefixes and (name + ".").startswith(prefixes) else dense_state
        )


def _fsdp2_shard_dense(model: nn.Module, fsdp_mesh, expert_modules) -> int:
    """FSDP2-shard the model's dense modules, leaving the routed experts to EP.

    FSDP2 ``fully_shard`` the model's DENSE (non-expert) modules over ``fsdp_mesh``, leaving the routed
    experts (already EP-sharded on ``ep`` + the a2a transport) untouched -- the "FSDP2 dense + EP MoE"
    hybrid. In the EDP=1 layout ``fsdp_mesh`` is the FULL world (dense shards over every rank, the same
    ranks the experts are EP-sharded over); a 2D ``(dp, fsdp)`` ``fsdp_mesh`` gives the EDP>1 hybrid
    (dense HSDP-sharded over ``fsdp`` and replicated over ``dp``, aligned with the experts' ``(edp, ep)``).

    Wraps at the DECODER-LAYER boundary (exactly like the proven pure-FSDP2 path,
    :func:`~modelopt.torch.utils.distributed._fsdp2_wrap_modules` via HF ``_no_split_modules``) and passes
    the routed-expert params as ``ignored_params`` so FSDP shards the layer's dense params (attention,
    layernorms, router) while leaving the experts EP-sharded on the ``ep`` mesh + a2a transport. Wrapping
    the whole layer (not ``self_attn`` or individual leaves) is essential: ``fully_shard`` swaps a wrapped
    module's class (``Foo`` -> ``FSDPFoo``), and modelopt inspects classes by name/source -- wrapping
    ``self_attn`` breaks ``mtq``'s KV-cache AST registration (``inspect.getsource`` on the synthetic
    class), and wrapping the leaf ``QuantLinear``s breaks the export's name-based ``is_quantlinear``.
    Wrapping only the LAYER leaves ``self_attn`` and the q/k/v/o ``QuantLinear``s PLAIN, while
    ``_get_enclosing_fsdp_module`` still finds the layer FSDP wrapper for the weight-access/writeback
    contexts. The root (embed/lm_head/final norm) is intentionally NOT wrapped: those stay replicated
    (and unswizzled, so their quantization-exclude still applies), matching the dense-replicated baseline.

    Returns the number of layers wrapped.
    """
    from torch.distributed.fsdp import fully_shard

    # Every routed-expert param is EP-sharded on the ``ep`` mesh; FSDP must IGNORE them so it never tries
    # to reshard them onto ``fsdp_mesh``. Ignored params not present in a given layer are simply no-ops.
    ignored_params: set = set()
    for em in expert_modules:
        ignored_params.update(em.parameters(recurse=True))

    from torch.distributed.tensor import DTensor, Replicate

    ignored_param_ids = {id(p) for p in ignored_params}  # id-based: Tensor.__eq__ is elementwise

    # Wrap only the OUTERMOST _no_split_modules. Some models (e.g. Kimi's MoonViT VLM) list BOTH a
    # container (the vision model) and its blocks in ``_no_split_modules``; ``fully_shard`` on the parent
    # already shards the children, so wrapping a child afterward hits ``DeviceMesh._concatenate`` ("Cannot
    # concatenate overlapping meshes"). Drop any wrap module nested under another.
    wrap_modules = _fsdp2_wrap_modules(model)
    wrap_ids = {id(m) for m in wrap_modules}
    nested_ids: set = set()
    for m in wrap_modules:
        for c in m.modules():
            if id(c) != id(m) and id(c) in wrap_ids:
                nested_ids.add(id(c))
    wrap_modules = [m for m in wrap_modules if id(m) not in nested_ids]

    n = 0
    for layer in wrap_modules:
        layer_ignored = set(ignored_params)  # experts + any pre-sharded dense (below)
        has_shardable = False
        for name, p in list(layer.named_parameters(recurse=True)):
            if id(p) in ignored_param_ids:
                continue
            if isinstance(p.data, DTensor):
                if all(isinstance(pl, Replicate) for pl in p.data.placements):
                    # Replicate -> plain (a free view, no gather), re-registered as a plain Parameter:
                    # assigning .data would keep the DTensor parameter, which fully_shard then tries
                    # to place on both its own mesh and the FSDP one.
                    owner_name, _, leaf = name.rpartition(".")
                    owner = layer.get_submodule(owner_name) if owner_name else layer
                    setattr(
                        owner,
                        leaf,
                        nn.Parameter(p.data.to_local(), requires_grad=p.requires_grad),
                    )
                else:
                    layer_ignored.add(
                        p
                    )  # already sharded by the loader (TP) -> leave it distributed
                    continue
            has_shardable = True
        if not has_shardable:
            continue  # nothing left to FSDP-shard in this layer (all expert / already-sharded)
        fully_shard(
            layer,
            mesh=fsdp_mesh,
            reshard_after_forward=True,
            ignored_params=layer_ignored,
        )
        n += 1
    return n


# The checkpoint formats load_and_prepare_ep dequantizes as it loads, beside unquantized ones (None).
_EP_SOURCE_FORMATS = frozenset({None, "mxfp4"})


def _ckpt_quant_method(cfg):
    """Return the checkpoint's native quantization method, or ``None``.

    Return the checkpoint's native quantization method (e.g. ``"fp8"`` for block-fp8 DeepSeek-V3),
    or ``None`` if the checkpoint isn't natively quantized. Checks the top config and the VLM ``text_config``.
    """
    for c in (cfg, getattr(cfg, "text_config", None)):
        qc = getattr(c, "quantization_config", None)
        m = qc.get("quant_method") if isinstance(qc, dict) else getattr(qc, "quant_method", None)
        if m:
            # transformers' config objects carry a QuantizationMethod enum, whose str() is not its value.
            return str(getattr(m, "value", m))
    return None


def _transformers_mesh_builder():
    """The module whose ``initialize_distributed_mesh`` ``from_pretrained`` calls, or ``None``.

    Newer transformers drops ``from_pretrained``'s ``device_mesh`` once ``distributed_config`` is
    given: it builds its own world-sized ``tp`` mesh from the config's sizes. That shards the experts
    over every rank instead of this rank's EP group whenever the EP degree is below the world size.
    Older releases honor ``device_mesh`` and have no such builder.
    """
    try:
        import transformers.distributed.mixin as mixin
    except ImportError:
        return None
    return mixin if hasattr(mixin, "initialize_distributed_mesh") else None


@contextlib.contextmanager
def _experts_sharded_over(ep_mesh):
    """Make ``from_pretrained`` shard the experts over ``ep_mesh`` on releases that build their own mesh."""
    mixin = _transformers_mesh_builder()
    if mixin is None:
        yield
        return

    def _ep_mesh(distributed_config):
        index = torch.cuda.current_device() if ep_mesh.device_type == "cuda" else None
        return torch.device(ep_mesh.device_type, index), ep_mesh

    original = mixin.initialize_distributed_mesh
    mixin.initialize_distributed_mesh = _ep_mesh
    try:
        yield
    finally:
        mixin.initialize_distributed_mesh = original


@contextlib.contextmanager
def _ep_dense_meta_load(is_expert):
    """Fully-distributed dense load for the EP hybrid.

    during ``from_pretrained``, force every non-expert weight/bias to materialize on ``meta`` (i.e. NOT materialize on
    the GPU) by making transformers' per-param ``get_device`` return ``"meta"`` for dense keys.

    transformers 5.x's ``convert_and_load_state_dict_in_model`` places EXPERT params via the tp /
    ``distributed_operation`` branch (which fuses + EP-shards them and never consults ``get_device``), and
    every OTHER (dense) param via ``param_device = get_device(device_map, key)`` -> ``spawn_materialize``.
    Returning ``"meta"`` there leaves dense params unmaterialized while experts still EP-shard normally. The
    skipped dense params are then FSDP2-sharded + DCP-loaded (:func:`_load_ep_dense_distributed`), so dense is
    never replicated per rank. Only ``.weight``/``.bias`` are forced meta -- persistent buffers (small) load
    normally. Patches ``get_device`` on both ``transformers.modeling_utils`` and the module that actually
    defines the loader (it's looked up as a module global there).
    """
    import sys as _sys

    import transformers.modeling_utils as _mu

    loader = getattr(_mu, "convert_and_load_state_dict_in_model", None)
    mods = {_mu}
    loader_module = getattr(loader, "__module__", None)
    dm = _sys.modules.get(loader_module) if loader_module else None
    if dm is not None:
        mods.add(dm)
    targets = [m for m in mods if hasattr(m, "get_device")]
    if not targets:
        yield
        return

    def _dense_weight(key) -> bool:
        k = str(key)
        return k.endswith((".weight", ".bias")) and not is_expert(k)

    saved = [(m, m.get_device) for m in targets]

    def _make(orig):
        def patched(device_map, key, *args, **kwargs):
            if _dense_weight(key):
                return "meta"
            return orig(device_map, key, *args, **kwargs)

        return patched

    try:
        for m, orig in saved:
            m.get_device = _make(orig)
        yield
    finally:
        for m, orig in saved:
            m.get_device = orig


def _load_ep_dense_distributed(model, fsdp_mesh, local_dir, is_expert, expert_modules):
    """Fully-distributed dense load for the EP hybrid.

    FSDP2-shard the (meta) dense modules and DCP-load each rank's shard directly from the checkpoint -- dense is NEVER
    materialized replicated. Experts are already EP-sharded (loaded by ``from_pretrained``) and left untouched. Returns
    the number of dense layers sharded, like :func:`_fsdp2_shard_dense`.

    Requires the dense ``.weight``/``.bias`` params to be on ``meta`` on entry (see
    :func:`_ep_dense_meta_load`). Delegates the materialize + DCP load to the SHARED
    :func:`~modelopt.torch.utils.distributed._meta_dcp_load` (``load_names`` restricted to the dense params),
    the same loader that backs the DP and FSDP2-full paths -- here just fed a partially-meta model (dense
    meta, experts already real) and a dense-only name set.
    """
    from modelopt.torch.utils.distributed import is_master, local_rank

    device = torch.device(f"cuda:{local_rank()}")
    # FSDP2-shard the dense modules. Their params are meta plain tensors -> fully_shard makes them meta Shard
    # DTensors; experts are passed as ignored_params and stay EP-sharded.
    n_dense = _fsdp2_shard_dense(model, fsdp_mesh, expert_modules)
    # Restrict the shared loader to the dense params (experts already materialized by from_pretrained must not
    # be touched). It selectively materializes only the meta dense modules (experts skipped), preserves
    # buffers across to_empty, and DCP-loads each rank's dense shard.
    dense_names = {
        n
        for n, _ in model.named_parameters()
        if n.endswith((".weight", ".bias")) and not is_expert(n)
    }
    if is_master():
        print(
            f"[ep] dense-distributed load: FSDP2-sharded {n_dense} dense module(s); "
            f"meta+DCP-loading {len(dense_names)} dense param(s) (experts left EP-sharded)",
            flush=True,
        )
    _meta_dcp_load(model, local_dir, device, load_names=dense_names)
    return n_dense


def load_and_prepare_ep(
    ckpt_path: str,
    plan: DistributionPlan,
    trust_remote_code: bool = False,
    attn_implementation: str | None = None,
    freeze: bool = True,
    hf_config=None,
    require_lm_head: bool = True,
) -> nn.Module:
    """Load + shard a MoE causal LM for expert-parallel PTQ via the DP=EP all-to-all transport.

    ``plan`` is the expert-parallel :class:`~modelopt.torch.utils.distributed.DistributionPlan` to load
    by (``ep_size > 1``), for the whole process group. It records no plan on the model: the caller
    does (:func:`~modelopt.torch.utils.distributed.set_distribution_plan`), as
    :func:`load_pretrained` does for every load.

    Builds a ``(dp, ep)`` device mesh and loads experts EP-sharded over the ``ep`` submesh
    (``enable_expert_parallel=True`` -- each rank owns ``num_experts / ep`` experts as grouped-GEMM
    shards), while ATTENTION and dense params stay REPLICATED (so EP has no KV-head-divisibility cap).
    The model's ``base_model_ep_plan`` is swapped to the ``torch_a2a`` styles: the router emits GLOBAL
    unmasked expert ids and the experts forward routes each token to its expert-owning rank by a NCCL
    all-to-all (dispatch) -> quantizable ``_grouped_linear`` -> all-to-all (combine), rather than every
    rank computing all tokens followed by an EP all-reduce. Every rank is data-parallel (sees DIFFERENT
    tokens), so calibration MUST shard the data per-rank.

    The ``dp`` dimension (present when ``plan.ep_size < world_size``) is additional data parallelism across
    EP groups: non-expert params stay REPLICATED per rank and each EP group calibrates on different
    data; experts stay EP-sharded (resident). No FSDP and no concurrent ``mesh_dp``/``mesh_ep``
    collectives -> no deadlock by construction. Costs more per-rank memory (non-experts + experts
    replicated across dp) but is the reliable path at scale.

    With ``plan.dense == "fsdp"`` the dense part is instead FSDP2-sharded over the ``ep`` group and each
    rank loads only its shard from the checkpoint, never the whole of it -- for a model whose dense part
    does not fit one GPU. A block-fp8 checkpoint cannot be loaded that way: Distributed Checkpoint
    cannot dequantize its F8_E8M0 scales.

    EP CANNOT be applied to a meta-init model (transformers' ``GroupedGemmParallel.shard_tensor``
    slices materialized tensors), so this loads real weights via ``from_pretrained`` over the ``ep``
    submesh. Requires the transformers MoE EP framework (ExpertsInterface / grouped-GEMM EP styles /
    enable_expert_parallel), first present in transformers >= 5.6. (Per-model plugins enforce a higher
    floor where their stock support landed later, e.g. qwen3_moe's stock base_model_ep_plan in >= 5.7.)

    Assumes the caller has already initialized (and device-bound) the process group -- distributed
    bring-up is the launcher's job, not this loader's (hf_ptq.main() does it via ``distributed_run``).
    The expert-parallel mesh stays recoverable from the model
    (:func:`~modelopt.torch.utils.moe.expert_parallel_mesh`).
    """
    from modelopt.torch.utils.distributed import ParallelState, is_initialized, is_master, size
    from modelopt.torch.utils.moe import (
        _ep_config,
        _routed_expert_keys,
        _routed_expert_modules,
        register_torch_a2a,
        require_transformers,
    )

    if not plan.is_expert_parallel:
        raise ValueError(f"{plan!r} shards no routed experts, which is what this load is for.")
    # This loader assumes the caller has ALREADY initialized (and device-bound) the process group --
    # distributed bring-up belongs to the launcher, not this library function. The example launcher does
    # it in hf_ptq.main() via distributed_run() -> modelopt.torch.utils.distributed.setup() before
    # the load. A live, device-bound PG is required here so the mesh subgroups built below create their
    # NCCL comms eagerly (the synchronized warmup all-reduce per subgroup further down) instead of lazily
    # on first collective (which deadlocks at DP>1 because ranks reach the lazy init in different phases).
    if not is_initialized():
        raise RuntimeError(
            "load_and_prepare_ep requires a live, device-bound process group; set it up first, e.g. "
            "with modelopt.torch.utils.distributed.distributed_run() under torchrun."
        )
    if plan.world_size != size():
        raise ValueError(f"{plan!r} is for {plan.world_size} ranks, but this run has {size()}.")
    # Hard floor: the transformers MoE expert-parallel framework. Below 5.6 there is no ExpertsInterface
    # / grouped-GEMM EP styles, and enable_expert_parallel silently shards nothing (no error) -> fail loud.
    require_transformers(
        "5.6", "ModelOpt DP=EP expert-parallel PTQ (transformers MoE EP framework)"
    )
    from torch.distributed.device_mesh import init_device_mesh
    from transformers import AutoConfig
    from transformers.distributed import DistributedConfig

    world_size, ep_degree, dp = plan.world_size, plan.ep_size, plan.edp
    if hf_config is None:
        config_kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code}
        if attn_implementation is not None:
            config_kwargs["attn_implementation"] = attn_implementation
        hf_config = AutoConfig.from_pretrained(ckpt_path, **config_kwargs)
    if plan.is_fsdp_dense and _ckpt_quant_method(hf_config) == "fp8":
        raise NotImplementedError(
            f"{plan!r}: the dense part of a block-fp8 checkpoint cannot be sharded, since Distributed "
            "Checkpoint cannot dequantize its F8_E8M0 scales. Plan dense='replicate'."
        )
    quant_method = _ckpt_quant_method(hf_config)
    if quant_method not in _EP_SOURCE_FORMATS:
        # transformers' own quantizer would load it as stored, for PTQ to treat as plain weights.
        dequantized = ", ".join(sorted(f for f in _EP_SOURCE_FORMATS if f))
        raise NotImplementedError(
            f"The expert-parallel load cannot dequantize a {quant_method!r} checkpoint: it loads "
            f"unquantized checkpoints and dequantizes {dequantized}. Start from an unquantized copy "
            "of the model."
        )
    dtype = getattr(hf_config, "torch_dtype", None) or torch.bfloat16
    # The weights load, and the checkpoint recorded for the export to read the unplaced tensors back
    # from, take the local snapshot: a Hub ID recorded there would leave the export nothing to read.
    # Resolved once the plan and the config pass, so a refused load downloads nothing.
    ckpt_path = _resolve_checkpoint_dir(ckpt_path, torch.distributed.get_rank())

    # Two INDEPENDENT meshes (NeMo-AutoModel style), named by the model part they govern -- two
    # factorizations of the same world:
    #   sparse_moe_mesh = (edp, ep): experts are SHARDED over the ``ep`` axis and REPLICATED over ``edp``;
    #       the ``ep`` submesh also drives from_pretrained's expert sharding + the a2a transport.
    #   dense_mesh = (dp,): dense/non-expert params are data-parallel-REPLICATED over the WHOLE world.
    # This subsumes the old a2a-vs-partial split: a2a is simply dp == 1 (``edp`` size 1), and dense amax
    # always reduces over the whole world (fixing the DP>1 case, where the old dp submesh missed the ep dim).
    sparse_moe_mesh = init_device_mesh("cuda", (dp, ep_degree), mesh_dim_names=("edp", "ep"))
    dense_mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("dp",))
    ep_submesh = sparse_moe_mesh["ep"]

    # NO TENSOR PARALLELISM IS REQUESTED HERE, despite the name. transformers implements expert
    # parallelism THROUGH the TP application path: with enable_expert_parallel=True the model's
    # ``tp_plan`` PROPERTY returns ``_ep_plan`` (distributed/mixin.py), so what actually gets applied
    # is the expert plan. ``tp_plan="auto"`` is only the SWITCH that makes ``tp_size > 1`` in
    # DistributedConfig.__post_init__ and therefore reaches ``apply_tensor_parallelism`` at all --
    # drop it and __post_init__ early-returns with tp_size=None, which then TypeErrors on
    # ``tp_size > 1`` in prepare_distribute_model. Dense params stay REPLICATED (= our DP) because
    # ``swapped_plan`` below keeps only expert entries; attention/embeds match no plan key.
    #
    # WHERE it goes is version-dependent: transformers >=5.16 raises if tp_plan is passed alongside
    # distributed_config ("Set tensor-parallel options on DistributedConfig when using it"); older
    # releases have no tp_plan field on DistributedConfig at all. Passing tp_plan ALONE is not an
    # option either -- the >=5.16 back-compat path rebuilds the config as
    # DistributedConfig(tp_plan=..., tp_size=...) and DROPS enable_expert_parallel, silently losing
    # EP. So decide from the dataclass fields, which tell the two apart on any release.
    import dataclasses as _dc

    _dc_has_tp_plan = any(f.name == "tp_plan" for f in _dc.fields(DistributedConfig))
    if _transformers_mesh_builder() is not None:
        # This release sizes its own mesh from tp_size; _experts_sharded_over swaps in the EP mesh.
        distributed_config = DistributedConfig(
            enable_expert_parallel=True, tp_plan="auto", tp_size=ep_degree
        )
    elif _dc_has_tp_plan:
        distributed_config = DistributedConfig(enable_expert_parallel=True, tp_plan="auto")
    else:
        distributed_config = DistributedConfig(enable_expert_parallel=True)
    from_pretrained_kwargs: dict[str, Any] = {
        "dtype": dtype,
        "device_mesh": ep_submesh,
        "distributed_config": distributed_config,
        "trust_remote_code": trust_remote_code,
    }
    if not _dc_has_tp_plan:
        from_pretrained_kwargs["tp_plan"] = "auto"
    if attn_implementation is not None:
        from_pretrained_kwargs["attn_implementation"] = attn_implementation
    if _ckpt_quant_method(hf_config) == "mxfp4":
        # Native mxfp4 checkpoints (e.g. gpt-oss: fused `gate_up_proj_blocks`/`_scales` + biases) must be
        # DEQUANTIZED to bf16 on load so ModelOpt can insert nvfp4/fp8 quantizers on the fused bf16 experts
        # (the a2a grouped_gemm expects bf16). transformers dequantizes blocks->bf16 during from_pretrained.
        # NB: the single-process loader needed device_map="sequential" to dodge a CUDA illegal-access in the
        # mxfp4 dequant under multi-GPU device_map="auto"; the EP path uses device_mesh (no device_map), so
        # watch for an EP-shard-awareness issue in the dequant convert-op.
        from transformers import Mxfp4Config

        from_pretrained_kwargs["quantization_config"] = Mxfp4Config(dequantize=True)
        if is_master():
            print(
                "[ep] native mxfp4 checkpoint (e.g. gpt-oss) -> Mxfp4Config(dequantize=True) -> bf16",
                flush=True,
            )
    # The swapped plan is this load's own: the caller's config keeps the plan its checkpoint declares.
    hf_config = copy.deepcopy(hf_config)
    ep_cfg = _ep_config(hf_config)  # for VLMs the plan lives on .text_config
    ep_plan = dict(getattr(ep_cfg, "base_model_ep_plan", None) or {})
    if not ep_plan:
        raise ValueError("expert-parallel PTQ requires the model to declare base_model_ep_plan")
    # DP=EP all-to-all: swap the ep_plan to our torch_a2a styles (torch_a2a_router emits GLOBAL ids
    # unmasked; torch_a2a_experts injects the EP process_group + skips the all-reduce) and select the
    # `torch_a2a` experts forward (NCCL all-to-all dispatch/combine around the quantizable
    # _grouped_linear). Tokens are routed to the expert-owning rank by all-to-all instead of every rank
    # computing all tokens + all-reduce. Requires per-rank (data-parallel) calib data. nemotron_h's
    # logits-only gate has no ep_router entry, so its nemotron_experts_ep also maps to torch_a2a_experts.
    register_torch_a2a()  # idempotent; register the experts forward + the two styles, right before use
    swap = {
        "ep_router": "torch_a2a_router",
        "moe_tp_experts": "torch_a2a_experts",
        "nemotron_experts_ep": "torch_a2a_experts",
    }
    # Keep expert-related entries (grouped_gemm expert params + the router/experts styles); DROP the TP
    # styles (a2a builds no TP group for them -> would shard attention across the wrong mesh).
    swapped_plan = {
        k: swap.get(v, v) for k, v in ep_plan.items() if v in swap or v == "grouped_gemm"
    }
    # Tag every routed-experts CONTAINER torch_a2a_experts (injects the EP process_group + marks the
    # routed expert), found as the distribution plan's decision finds them: some plans tag only the
    # expert PARAMS grouped_gemm, and name the container through their parents.
    swapped_plan.update(dict.fromkeys(_routed_expert_keys(ep_plan), "torch_a2a_experts"))
    from_pretrained_kwargs["experts_implementation"] = "torch_a2a"
    ep_cfg.base_model_ep_plan = swapped_plan
    from_pretrained_kwargs["config"] = hf_config
    if is_master():
        print(f"[ep] ep_plan={swapped_plan} experts_implementation=torch_a2a", flush=True)
    # The class that builds the whole checkpoint -- with its lm_head, so the head is calibrated and
    # exported and the generation preview works -- chosen as the distribution plan's decision chose
    # it. The per-model plugin (models/) has already installed any on-the-fly weight decompression as
    # a transformers monkeypatch.
    loader = _auto_class(hf_config, require_lm_head)
    if is_master():
        print(f"[ep] loader={loader.__name__} (require_lm_head={require_lm_head})", flush=True)
    # Size the load-time caching-allocator warmup to the real per-rank (EP-sharded) footprint instead of
    # the full model -- otherwise transformers pre-allocates ~device-max on every rank and torch holds it
    # as reserved cache, starving the NCCL a2a buffers. Expert params are identified from `swapped_plan`
    # (the actual EP plan). Always on: falls back to stock sizing on any issue and a mis-size only nudges
    # load speed, never correctness. (See _ep_aware_caching_allocator_warmup.)
    is_expert_fn = _make_expert_param_matcher(swapped_plan)
    # The plan places the DENSE (non-expert) modules; the experts are always EP-sharded by from_pretrained.
    # A replicated dense part loads with them, on every rank. A sharded one is left on meta during
    # from_pretrained (get_device->meta via _ep_dense_meta_load), then FSDP2-sharded + DCP-loaded, so it
    # is never replicated anywhere.
    _dense_ctx = (
        _ep_dense_meta_load(is_expert_fn) if plan.is_fsdp_dense else contextlib.nullcontext()
    )
    with (
        _ep_aware_caching_allocator_warmup(
            ep_degree, swapped_plan, dense_on_meta=plan.is_fsdp_dense
        ),
        _dense_ctx,
        _experts_sharded_over(ep_submesh),
    ):
        model = loader.from_pretrained(ckpt_path, **from_pretrained_kwargs)
    model.eval()
    if plan.is_fsdp_dense and is_master():
        _nmeta = [n for n, p in model.named_parameters() if p.is_meta]
        _dense_meta = [n for n in _nmeta if not is_expert_fn(n)]
        print(
            f"[ep] dense-meta load: {len(_nmeta)} meta param(s) after from_pretrained "
            f"({len(_dense_meta)} dense, {len(_nmeta) - len(_dense_meta)} expert); experts materialized. "
            f"e.g. dense-meta {_dense_meta[:3]}",
            flush=True,
        )
    # Force the experts forward to dispatch to `torch_a2a`: from_pretrained's `experts_implementation`
    # kwarg resolves non-fp8 impls back to "grouped_mm", but @use_experts_implementation reads
    # `config._experts_implementation` at FORWARD time, so set it on every (sub)config now. (Without
    # this the torch_a2a_experts style passes a 5th process_group arg to grouped_mm_experts_forward ->
    # TypeError.)
    for _m in [model, *model.modules()]:
        _cfg = getattr(_m, "config", None)
        if _cfg is not None and hasattr(_cfg, "_experts_implementation"):
            _cfg._experts_implementation = "torch_a2a"

    # The checkpoint tensors the model has no parameter for (an MTP head), worked out once, at load,
    # for the export to carry.
    record_unplaced_source_keys(model, ckpt_path)

    # DP=EP all-to-all: no FSDP. Non-experts stay replicated per rank (from_pretrained already placed
    # them on this rank's GPU); experts stay EP-sharded on the ep submesh. The model carries
    # model._device_mesh = ep_submesh (sparse_moe_mesh["ep"]), which calibration re-installs the
    # all-to-all forwards over (expert_parallel_mesh).
    if hasattr(model, "tie_weights"):
        model.tie_weights()
    if freeze:
        model.requires_grad_(False)

    # The routed-expert containers, from the SAME base_model_ep_plan that drove the sharding. A MoE model
    # loaded WITHOUT EP (dense DP) never reaches this loader, so its experts stay dense.
    expert_modules = _routed_expert_modules(model, ep_plan)
    _refuse_unquantizable_experts(expert_modules.values())

    # "FSDP2 dense + EP MoE" hybrid: experts are EP-sharded over ``ep`` and the DENSE (non-expert) modules
    # are FSDP2-sharded over the SAME ``ep`` group, replicated over ``edp``. At EDP=1 that is the full world
    # (dense shards over every rank); at EDP>1 it is the 2D (edp, ep) HSDP layout -- dense shards over ``ep``
    # (== ``mp``) and replicates over ``edp`` (== ``dp``), byte-for-byte aligned with the experts'
    # ``sparse_moe_mesh``. Dense all-gather then expert a2a run sequentially within a forward (same
    # ``ep`` group), so there is no concurrent-collective deadlock. Weight amax is computed on the
    # FSDP-gathered full weight (identical per rank); activation amax syncs over the whole world via the
    # dense ``dp`` group registered below.
    #
    # A replicated dense part stays data-parallel and needs no post-load step.
    if plan.is_fsdp_dense:
        # Fully-distributed dense: dense was left on meta during from_pretrained; FSDP2-shard it and
        # DCP-load each rank's shard directly (never replicated anywhere). EDP=1 -> the 1D full-world dense
        # mesh (the original, validated layout); EDP>1 -> the 2D (edp, ep) mesh (fully_shard replicates dim0
        # ``edp``, shards dim1 ``ep``), so the meta+DCP loader fills [Replicate(edp), Shard(ep)] DTensors.
        dense_fsdp_mesh = dense_mesh["dp"] if dp == 1 else sparse_moe_mesh
        # Collective: rank 0 makes the checkpoint local (downloading a Hub ID once); the others wait.
        _, local_dir = ensure_local_checkpoint(ckpt_path)
        n_dense = _load_ep_dense_distributed(
            model, dense_fsdp_mesh, local_dir, is_expert_fn, expert_modules.values()
        )
        # The keep-sharded writer reads each weight's own mesh/placements, so it dedups the ``edp``
        # replicas at EDP>1.
        if is_master():
            _dims = tuple(dense_fsdp_mesh.shape)
            print(
                f"[ep] dense sharded: FSDP2-sharded {n_dense} dense module(s) on mesh "
                f"{dense_fsdp_mesh.mesh_dim_names}={_dims} (shard over ep={ep_degree}, replicate over "
                f"edp={dp}); experts EP-sharded on ep",
                flush=True,
            )

    # Each module's ParallelState for calibration's amax sync: dense modules reduce over the whole world;
    # routed experts reduce over their expert-data-parallel replicas, and share one input amax across the
    # layer's experts over the expert-parallel group. A size-1 axis is left unused (-1): no collective.
    def _axis(mesh):
        return mesh.get_group() if mesh.size() > 1 else -1

    _seed_parallel_states(
        model,
        expert_modules,
        dense_state=ParallelState(data_parallel_group=_axis(dense_mesh["dp"])),
        expert_state=ParallelState(
            data_parallel_group=_axis(sparse_moe_mesh["edp"]),
            expert_model_parallel_group=_axis(sparse_moe_mesh["ep"]),
        ),
    )
    if is_master():
        print(
            f"[ep] seeded the expert-parallel state of {len(expert_modules)} routed-expert "
            "module(s) from base_model_ep_plan",
            flush=True,
        )
    # Eagerly create each amax collective's NCCL comm in a globally-consistent order (barrier around
    # each), so the sub-communicators are built in lockstep rather than lazily mid-calibration -- which
    # deadlocks when submesh groups are phase-skewed at DP>1. NO forward warmup (variable per-group time
    # would desync ranks before comm creation). Skip size-1 axes (no collective). The EP group is already
    # live from from_pretrained's expert sharding; re-warming it is a harmless redundant all-reduce. Same
    # order on every rank -> lockstep.
    warm_device = next(model.parameters()).device
    for _sub in (dense_mesh["dp"], sparse_moe_mesh["ep"], sparse_moe_mesh["edp"]):
        if _sub.size() <= 1:
            continue
        torch.distributed.barrier()
        torch.distributed.all_reduce(torch.ones(1, device=warm_device), group=_sub.get_group())
        torch.cuda.synchronize()
    torch.distributed.barrier()
    return model
