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
import dataclasses
import logging
import os
import re
import warnings
from collections import defaultdict
from contextlib import contextmanager
from itertools import chain
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed.checkpoint as dcp
import torch.nn as nn
from huggingface_hub import snapshot_download

if TYPE_CHECKING:
    from collections.abc import Iterable
from torch.distributed.checkpoint import HuggingFaceStorageReader
from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict
from torch.distributed.fsdp import FSDPModule
from torch.distributed.tensor import DTensor
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoModelForSeq2SeqLM,
    AutoModelForSpeechSeq2Seq,
    PretrainedConfig,
)
from transformers.conversion_mapping import get_model_conversion_mapping
from transformers.core_model_loading import WeightConverter, dot_natural_key, rename_source_key

from modelopt.torch.utils import distributed as dist
from modelopt.torch.utils import env
from modelopt.torch.utils.distributed import (
    DistributionPlan,
    barrier,
    broadcast_state_dict,
    is_initialized,
    set_distribution_plan,
)
from modelopt.torch.utils.fsdp2_utils import fsdp2_wrap
from modelopt.torch.utils.logging import print_rank_0, warn_rank_0
from modelopt.torch.utils.moe import (
    _choose_ep_degree,
    _ep_config,
    _num_experts_from_config,
    _routed_expert_keys,
    _routed_expert_modules,
)
from modelopt.torch.utils.plugins.hf_checkpoint_utils import (
    checkpoint_tensor_files,
    ensure_local_checkpoint,
    indexed_weight_map,
    read_safetensors_subset,
)

__all__ = [
    "build_meta_causal_lm",
    "fsdp2_wrap",
    "is_fsdp2_text_only_vlm",
    "load_and_prepare_ep",
    "load_pretrained",
    "parallel_load_and_prepare_fsdp2",
    "record_unplaced_source_keys",
    "resolve_distribution_plan",
    "unplaced_source_keys",
]

logger = logging.getLogger(__name__)


def is_fsdp2_text_only_vlm(config) -> bool:
    """Whether the VLM has a supported text-only FSDP2 decoder layout."""
    return getattr(config, "model_type", None) in {"qwen3_5", "qwen3_5_moe"}


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
    auto_class: Any = None,
):
    """Build a meta-init causal LM (no real storage allocated), or ``auto_class``'s model.

    It leaves ``auto_class`` as it found it, and the auto classes a remote model's ``__init__``
    builds its parts through (``AutoModel``, ``AutoModelForCausalLM``). Building a remote-code
    model through an auto class registers its class there, as it is, and a later load through that
    auto class builds the registered class, without the ``GenerationMixin`` a first load adds: a
    rank that built one more meta model than its peers, as rank 0 deciding a distribution does,
    would load its model without the ``generate`` theirs have. That assumes a remote model registers
    its classes while it is built: one its module registered when imported is undone as well, and a
    later load in the process, which finds the module imported, does not register it again.
    """
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

    model_factory = auto_class
    if model_factory is None:
        model_factory = AutoModelForCausalLM
        if is_fsdp2_text_only_vlm(hf_config):
            model_factory = AutoModelForImageTextToText
    # Where transformers records the class a remote-code build registers: in the auto class built
    # through, and in those a remote VLM builds its towers through. Read without a fallback: if a
    # release moves it, this fails here rather than leave one rank's model without generate.
    kept = [
        (cls._model_mapping._extra_content, dict(cls._model_mapping._extra_content))
        for cls in (model_factory, AutoModel, AutoModelForCausalLM)
    ]
    try:
        with init_empty_weights(include_buffers=False):
            model = model_factory.from_config(
                hf_config, dtype=dtype, trust_remote_code=trust_remote_code
            )
    finally:
        for registered, saved in kept:
            registered.clear()
            registered.update(saved)
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
_FSDP2_ONLY = frozenset({"cpu_offload", "mp_policy", "broadcast_chunk_size"})
_SHARDED_LOAD_OPTIONS = {
    _FSDP2: _FSDP2_ONLY | {"trust_remote_code", "attn_implementation", "hf_config"}
}
# The FSDP2-only values that ask nothing of a whole load: the FSDP2 load's defaults, and None where
# that leaves the option unset (a None broadcast_chunk_size asks for all of a source's layers).
_ASKS_NOTHING = {"cpu_offload": (False, None), "mp_policy": (None,), "broadcast_chunk_size": (8,)}


def _route(plan: DistributionPlan) -> str | None:
    """The sharded load ``plan`` takes, or ``None`` when every rank loads the whole model."""
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


@contextmanager
def _refused_together(label: str, alone: bool):
    """Raise on every rank if the block raises on any, so no rank waits for one that refused."""
    error = None
    try:
        yield
    except BaseException as e:  # A SystemExit too: the others would wait for this rank as well.
        error = e

    def refuse() -> None:
        if error is not None:
            raise error

    if alone:  # A process of a launch that set no group up has no peers to tell.
        refuse()
    else:
        dist.all_reduce(label, refuse, combine=lambda _: None)


def _read_config(path, auto_class, trust_remote_code: bool = False, **hub_options):
    """``path``'s config as ``auto_class.from_pretrained`` reads it: config_class, or AutoConfig."""
    if (reader := getattr(auto_class, "config_class", None)) is not None:
        return reader.from_pretrained(path, **hub_options)
    return AutoConfig.from_pretrained(path, trust_remote_code=trust_remote_code, **hub_options)


def _init_quantized_weights() -> bool:
    """Whether ``init_quantized_weights`` patched ``from_pretrained`` to quantize and place."""
    return getattr(AutoModelForCausalLM.from_pretrained, "_modelopt_init_quantized_weights", False)


def _whole_instead(decided: bool) -> str:
    """How to have every rank load the model whole: the knob counts only for a decided plan."""
    plan = f"distribution=DistributionPlan({dist.size()}, dense='replicate')"
    return f"MODELOPT_FSDP_SIZE=1 or {plan}" if decided else plan


def _refuse_sharded_load_options(
    route, auto_class, kwargs, decided: bool, measured: bool = False
) -> None:
    """Refuse what the sharded load ``route`` cannot honor; ``measured``: GPU memory chose it."""
    # An option passed as None, from_pretrained's own default (device_map=None, say), asks nothing.
    passed = {name for name, value in kwargs.items() if value is not None}
    unknown = sorted(passed - _SHARDED_LOAD_OPTIONS[route] - {"dtype", "torch_dtype"})
    # torch_dtype is from_pretrained's older name for dtype, and None means "auto" to it.
    dtypes = [name for name in ("dtype", "torch_dtype") if kwargs.get(name) not in (None, "auto")]
    if measured and (refused := sorted({*unknown, *dtypes})):
        # Say so, rather than refuse the options below as if the caller had asked for this load.
        raise TypeError(
            f"{refused} do not apply to a sharded load, but no plan or MODELOPT_FSDP_SIZE asked for "
            f"a layout, and by rank 0's GPU memory the model does not fit one GPU, so the {route} "
            f"load shards it. To load it whole on every rank: {_whole_instead(decided)}."
        )
    if _init_quantized_weights():
        raise NotImplementedError(
            f"The {route} load builds the model without from_pretrained, so it would skip "
            "init_quantized_weights' quantization; load the model whole on every rank: "
            f"{_whole_instead(decided)}."
        )
    if auto_class not in (None, AutoModelForCausalLM):
        name = getattr(auto_class, "__name__", repr(auto_class))
        raise NotImplementedError(
            f"The {route} load builds its model class itself, a causal LM's or a text-only VLM's "
            f"image-text class, so it takes no auto_class={name}: leave it out to have the load "
            f"pick, or load the model whole with it: {_whole_instead(decided)}."
        )
    if dtypes:  # A sharded load keeps the checkpoint's dtype.
        name = dtypes[0]
        raise TypeError(
            f"The {route} load reads the checkpoint's tensors as stored and keeps their dtype, so "
            f"{name}={kwargs[name]!r} does not apply to it; FSDP2's mp_policy sets the dtype it "
            "computes in."
        )
    if unknown:
        raise TypeError(
            f"{unknown} do not apply to the {route} load, which places the model itself; it "
            f"takes {sorted(_SHARDED_LOAD_OPTIONS[route])}. To place the model yourself, have "
            f"every rank load it whole: {_whole_instead(decided)}."
        )


def _sharded_load_kwargs(
    route, ckpt_path, auto_class, model_type, kwargs, hub_options, decided: bool, measured: bool
) -> dict:
    """What the sharded load ``route`` names takes, refusing what it cannot honor."""
    from modelopt.torch.models.hf import checkpoint_has_mtp

    _refuse_sharded_load_options(route, auto_class, kwargs, decided, measured)
    # Past the refusal, anything else is a dtype the load keeps or an option at its default, None.
    kwargs = {k: v for k, v in kwargs.items() if k in _SHARDED_LOAD_OPTIONS[route]}
    if checkpoint_has_mtp(model_type, ckpt_path):
        # The head is built on a whole model only; carried over, it would export unquantized.
        raise NotImplementedError(
            f"The {route} load cannot build the MTP head this {model_type} checkpoint carries. To "
            f"load it whole on every rank: {_whole_instead(decided)}."
        )
    trust_remote_code = kwargs.pop("trust_remote_code", False)
    hf_config = kwargs.pop("hf_config", None)
    if hf_config is None:
        hf_config = AutoConfig.from_pretrained(
            ckpt_path, trust_remote_code=trust_remote_code, **hub_options
        )
    nested = (getattr(hf_config, name, None) for name in ("text_config", "llm_config"))
    if any(getattr(c, "quantization_config", None) is not None for c in (hf_config, *nested)):
        # Read as stored, quantized weights (a VLM's language model's too) would load as if not.
        raise NotImplementedError(
            f"The {route} load cannot dequantize this {model_type} checkpoint. To load it whole on "
            f"every rank: {_whole_instead(decided)}."
        )
    # As build_meta_causal_lm builds: AutoModelForCausalLM refuses most VLMs' configs, and builds
    # some as the language model alone, whose other towers the export would carry unquantized.
    builds = (
        AutoModelForImageTextToText if is_fsdp2_text_only_vlm(hf_config) else AutoModelForCausalLM
    )
    if _auto_class(hf_config, trust_remote_code=trust_remote_code) is not builds:
        raise NotImplementedError(
            f"The {route} load builds a causal LM, or a text-only VLM it supports, which this "
            f"{model_type} checkpoint is not. To load it whole on every rank, pass its model class "
            f"as auto_class, with {_whole_instead(decided)}."
        )
    if not dist.is_initialized() and "RANK" not in os.environ:
        # FSDP2 shards through a process group even on one rank.
        raise RuntimeError(
            f"The {route} load shards through a process group, even on one rank, which a process "
            "started without a launcher cannot set up; launch it with `torchrun --nproc_per_node=1`."
        )
    if not _has_own_gpu():
        # Refused with the other ranks, not inside the load, where they would wait for this one.
        raise RuntimeError(
            f"The {route} load shards onto each rank's own GPU, and this rank has none: it sees "
            f"{torch.cuda.device_count()} GPU(s) as local rank {dist.local_rank()}. To load it "
            f"whole on every rank: {_whole_instead(decided)}."
        )
    return {"trust_remote_code": trust_remote_code, "hf_config": hf_config, **kwargs}


def _own_gpu_index() -> int | None:
    """This rank's GPU, as torch's DeviceMesh takes it, or ``None`` for a rank with none there.

    The device CUDA is set to once it is set up, else the one the rank's local rank numbers.
    """
    if not torch.cuda.is_available():
        return None
    if torch.cuda.is_initialized():
        return torch.cuda.current_device()
    index = dist.local_rank()
    return index if index < torch.cuda.device_count() else None


def _has_own_gpu() -> bool:
    """Whether this rank has a GPU of its own (:func:`_own_gpu_index`)."""
    return _own_gpu_index() is not None


def _load_sharded(ckpt_path, load_kwargs) -> nn.Module:
    """The FSDP2 load, by the arguments :func:`_sharded_load_kwargs` checked."""
    if not dist.is_initialized():
        # A single process's group of one, which the sharded model needs while it lives.
        dist.setup()
    return parallel_load_and_prepare_fsdp2(
        ckpt_path,
        torch.device("cuda", _own_gpu_index()),
        dist.rank(),
        dist.size(),
        **load_kwargs,
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

    ``distribution`` is the :class:`~modelopt.torch.utils.distributed.DistributionPlan` to load by:
    every rank takes rank 0's or, when rank 0 passes none, :func:`resolve_distribution_plan`'s.

    * ``dense="fsdp"``: :func:`parallel_load_and_prepare_fsdp2` shards a causal LM, or a text-only
      VLM it supports, over the process group (one process needs ``torchrun --nproc_per_node=1``).
      It builds and places the model itself, so it takes its own options but no ``auto_class``,
      placement or ``dtype``, and refuses a quantized checkpoint.
    * otherwise ``auto_class.from_pretrained(ckpt_path, **kwargs)`` (``auto_class`` defaults to
      ``AutoModelForCausalLM``); on a distributed run every rank loads the whole model, onto its own
      GPU unless ``device_map`` is passed or ``init_quantized_weights`` places it.

    No load here shards the routed experts: a plan with ``ep_size > 1`` is refused. A refusal on any
    rank raises on every rank, before the load. The plan is recorded on the model
    (:func:`~modelopt.torch.utils.distributed.set_distribution_plan`) for calibration and export, so
    :func:`~modelopt.torch.utils.fsdp2_utils.fsdp2_wrap` refuses it: to shard a model yourself,
    load it with ``from_pretrained``. :doc:`/guides/_distributed_loading` has the routing in full.
    Around the load:

    * A Hub model ID is made local first (:func:`ensure_local_checkpoint`, which takes the
      ``revision``, ``token``, ``cache_dir`` and ``local_files_only`` ``kwargs``): the export reads
      the source checkpoint from disk. Under ``torch.distributed`` every rank must call this. The
      config read and a whole-model load still take all but ``revision``: an ``auto_map`` entry can
      name custom code in another Hub repo, which Transformers fetches with them. A sharded load
      builds the model with ``from_config``, which takes none. A ``subfolder`` is part of where the
      checkpoint is: the config, the loading hook and the record below read it, as the load does.
    * The model type's loading hook runs
      (:func:`~modelopt.torch.models.hf.prepare_model_for_loading`), e.g. to keep the MTP head a
      Nemotron-H checkpoint carries. ``model_type`` selects it; when omitted, the type of the
      ``config`` passed for the load, else of the checkpoint's config. A sharded load cannot run
      it, so it refuses a checkpoint the hook would build modules for.
    * The checkpoint tensors the model has no parameter for are recorded
      (:func:`record_unplaced_source_keys`), so the export carries them over unchanged. The model
      is returned alone: ``output_loading_info=True`` is refused. So is a ``variant``: the record
      and the export read the checkpoint's default weight files.
    """
    # Importing modelopt.torch.models registers every model spec; only callers of this need them.
    from modelopt.torch.models.hf import prepare_model_for_loading

    # A launch whose processes set no group up (hf_ptq without --use_fsdp2): each loads alone.
    alone = dist.is_distributed_launch() and not dist.is_initialized()
    if alone and isinstance(distribution, DistributionPlan) and distribution.world_size > 1:
        raise RuntimeError(
            f"{distribution!r} is for {distribution.world_size} ranks, whose process group this "
            f"{os.environ['WORLD_SIZE']}-rank launch has not set up; load inside distributed_run(), "
            "which sets it up."
        )
    if alone and isinstance(distribution, DistributionPlan) and _route(distribution) is not None:
        # One rank cannot set the launch's group up, and a one-rank plan is no plan for the launch.
        launch = os.environ["WORLD_SIZE"]
        raise RuntimeError(
            f"The {_route(distribution)} load shards through this {launch}-rank launch's process "
            "group, which is not set up; load inside distributed_run(), which sets it up, with a "
            f"plan for the launch: DistributionPlan({launch}, dense='fsdp')."
        )
    if alone and distribution is None:
        # Knobs asking to shard over the launch are refused, as is a value that is no EP degree.
        ep, world = _ep_degree(None), int(os.environ["WORLD_SIZE"])
        experts = ep != "auto" and ep > 1
        asked = [f"MODELOPT_EP_DEGREE={env.ep_degree()}"] if experts else []
        if _dense_from_env(world) == "fsdp":
            asked.append(f"MODELOPT_FSDP_SIZE={env.fsdp_size()}")
        if asked:
            ep_way = ", with MODELOPT_EP_DEGREE=1: no load here shards the routed experts"
            raise RuntimeError(
                f"{' and '.join(asked)} ask to shard the model over this "
                f"{os.environ['WORLD_SIZE']}-rank launch, whose process group is not set up; load "
                f"inside distributed_run(), which sets it up{ep_way if experts else ''}."
            )
    # None is these options' own default: passed explicitly, it is no option to refuse.
    for name in ("config", "variant", "hf_config"):
        if name in kwargs and kwargs[name] is None:
            del kwargs[name]
    # The config to load with, or, as from_pretrained also takes it, a path to read it from.
    config = kwargs.get("config")
    config = config if isinstance(config, (PretrainedConfig, str, os.PathLike)) else None

    def handed() -> DistributionPlan | None:
        if distribution is None:
            return None
        if not isinstance(distribution, DistributionPlan):
            raise TypeError(f"distribution={distribution!r}: expected a DistributionPlan or None.")
        _route(distribution)
        return distribution

    # Every rank loads rank 0's distribution, which rank 0 checks before it sends it.
    plan = handed() if alone else dist.broadcast_from("distribution", handed)
    hub_options = {
        k: kwargs.pop(k) for k in ("token", "cache_dir", "local_files_only") if k in kwargs
    }
    revision = {"revision": kwargs.pop("revision")} if "revision" in kwargs else {}
    decided = plan is None and dist.size() > 1
    # Each rank checks its own arguments; a refusal on one fails every rank before the fetch.
    with _refused_together("load arguments", alone):
        # No load here shards the routed experts yet, so an EP degree left to the decision is 1.
        # Parsed in this check, a knob one rank cannot parse fails every rank.
        ep_size = 1 if decided and _ep_degree(None) == "auto" else None
        if kwargs.pop("output_loading_info", False):
            raise ValueError(
                "load_pretrained returns the model alone and does not take "
                "output_loading_info=True; the checkpoint tensors the model has no parameter for "
                "are recorded on it instead (see record_unplaced_source_keys)."
            )
        if kwargs.get("variant") is not None:
            raise ValueError(
                f"load_pretrained does not take variant={kwargs['variant']!r}: the checkpoint "
                "tensors the model has no parameter for are recorded from, and carried into the "
                "export from, the checkpoint's default weight files."
            )
        if config is not None and "hf_config" in kwargs:
            raise TypeError("config and hf_config both name the config to load with; pass one.")
        if "hf_config" in kwargs and (plan is None or _route(plan) is None):
            # Only a sharded load builds from hf_config, so it needs the plan it came with.
            raise TypeError(
                "hf_config is the config resolve_distribution_plan returned with a plan that "
                "shards the model: pass that plan as distribution=, or pass config= to have the "
                "plan decided from your config."
            )
        if plan is not None and _route(plan) is not None:
            # A config passed becomes the load's hf_config; subfolder is where the checkpoint is.
            _refuse_sharded_load_options(
                _route(plan),
                auto_class,
                {
                    k: v
                    for k, v in kwargs.items()
                    if k != "subfolder" and not (k == "config" and config is not None)
                },
                decided=False,
            )
    # Rank 0 fetches the checkpoint; a rank that cannot see it then fails every rank.
    with _refused_together("checkpoint fetch", alone):
        _, ckpt_path = ensure_local_checkpoint(ckpt_path, **revision, **hub_options)
        if config is not None and not isinstance(config, PretrainedConfig):
            # Read as the load reads it, so the plan and the loading hook follow the model's config.
            trust = kwargs.get("trust_remote_code", False)
            config = _read_config(config, auto_class, trust, **hub_options)
    subfolder = kwargs.pop("subfolder", None)
    if subfolder:
        ckpt_path = os.path.join(ckpt_path, subfolder)
    trust_remote_code = kwargs.get("trust_remote_code", False)
    hf_config = kwargs.get("hf_config")
    if decided:
        plan, hf_config = resolve_distribution_plan(
            ckpt_path,
            config=config,
            auto_class=auto_class,
            trust_remote_code=trust_remote_code,
            ep_size=ep_size,
            **hub_options,
        )
        if _route(plan) is not None:
            kwargs.setdefault("hf_config", hf_config)
    elif plan is None:
        # One rank holds the whole model, whatever the model: there is nothing to decide or read.
        plan = DistributionPlan()
    route = _route(plan)
    # Each rank reads the config itself, so a refusal of it fails every rank before the load.
    with _refused_together("load checks", alone):
        # Options only the FSDP2 load takes would reach from_pretrained, which has no use for them.
        if route is None:
            for name, nothing in _ASKS_NOTHING.items():
                if name in kwargs and kwargs[name] in nothing:
                    del kwargs[name]
        if route is None and (sharded := sorted(set(kwargs) & _FSDP2_ONLY)):
            raise TypeError(
                f"{sharded} only apply to the FSDP2 load, but {plan!r} gives every rank the whole "
                "model; pass a plan with dense='fsdp' to shard it."
            )
        measured = decided and route == _FSDP2 and _dense_from_env(plan.world_size) == "auto"
        if route is not None and config is not None:
            # The sharded load builds from hf_config, a copy, as building writes to the config.
            del kwargs["config"]
            if "hf_config" not in kwargs:
                kwargs["hf_config"] = copy.deepcopy(config)
        if model_type is None:
            # The load uses the config passed for it, if any, so the loading hook follows that one.
            if route == _FSDP2 and hf_config is None and config is None:
                # The FSDP2 load builds from the checkpoint's config too: read it once, for both.
                hf_config = AutoConfig.from_pretrained(
                    ckpt_path, trust_remote_code=trust_remote_code, **hub_options
                )
                kwargs["hf_config"] = hf_config
            chosen = hf_config if hf_config is not None else config
            model_type = getattr(chosen, "model_type", None) or (
                _read_config(ckpt_path, auto_class, trust_remote_code, **hub_options).model_type
            )
        load_kwargs = (
            {}
            if route is None
            else _sharded_load_kwargs(
                route, ckpt_path, auto_class, model_type, kwargs, hub_options, decided, measured
            )
        )
    if route is not None:
        model = _load_sharded(ckpt_path, load_kwargs)
        # A sharded load finds the unplaced keys while it places the weights.
        keys = model._modelopt_unplaced_source_keys
    else:
        # init_quantized_weights computes the device map itself, and refuses one it is passed.
        spread = plan.world_size > 1 and not _init_quantized_weights()
        if spread and (index := _own_gpu_index()) is not None:
            # Each rank takes the whole model onto its own GPU: under "auto", every rank's
            # accelerate could pick the same idle GPU.
            kwargs.setdefault("device_map", {"": f"cuda:{index}"})
        elif spread and "device_map" not in kwargs:
            # This rank says so itself: rank 0 may have a GPU of its own, and nothing to say.
            warnings.warn(
                f"Rank {dist.rank()} has no GPU of its own: it sees {torch.cuda.device_count()} "
                f"GPU(s) as local rank {dist.local_rank()}, so it loads the whole model where "
                "from_pretrained places it without a device_map, on the CPU.",
                stacklevel=2,
            )
        auto_class = auto_class or AutoModelForCausalLM
        # It runs no collective, so a rank failing in it alone would leave the others in the next.
        with (
            _refused_together("whole-model load", alone),
            prepare_model_for_loading(
                model_type,
                ckpt_path,
                trust_remote_code,
                model_class=None if auto_class in (AutoModel, AutoModelForCausalLM) else auto_class,
            ),
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

    It does not resolve keys through :func:`_target_param`: a model built on meta has no record of what
    ``from_pretrained`` ran, and the stock mapping lacks a remote-code model's renames. The loader instead
    refuses a mapping that fills a tensor the model's load recorded as unplaced, which the export carries.
    """
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
                cands = suffix_idx[tuple(comps[-k:])]
                if len(cands) < 2:  # only "exactly one candidate" is read below
                    cands.append(mk)
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

    Every target param ends up filled -- from the checkpoint, by ``tie_weights`` with a param the
    checkpoint filled, or with the value it already had -- or the load fails: nothing is left as the
    uninitialized memory ``to_empty`` hands out.
    """
    # Read what the checkpoint holds before changing the model, so a refusal leaves it as it was.
    reader = HuggingFaceStorageReader(str(local_dir))
    ckpt_md = reader.read_metadata().state_dict_metadata
    ckpt_keys = set(ckpt_md)
    if not ckpt_keys:
        raise RuntimeError(
            f"meta+DCP load: the checkpoint at {local_dir} has NO tensors -- the resolved snapshot carries no "
            "safetensors (a config-only / incomplete HF revision; e.g. refs/main pointing at a weightless "
            "snapshot). Re-download the checkpoint or pass a complete local snapshot dir / pinned revision."
        )
    if isinstance(model, FSDPModule):
        raise RuntimeError(
            "meta+DCP load: the root module is fully_shard-ed. The load materializes module by module, "
            "so already-real params survive, which a root-sharded FSDP2 state does not take: shard the "
            "decoder layers only."
        )
    meta_buffers = sorted(n for n, b in model.named_buffers() if b.is_meta)
    if meta_buffers:
        raise RuntimeError(
            f"meta+DCP load: {len(meta_buffers)} buffer(s) are on meta, which a checkpoint of params "
            f"cannot fill (a non-persistent buffer is not in it): {meta_buffers[:3]}. Build the model "
            "with its buffers materialized, as accelerate's init_empty_weights does."
        )

    # ``dcp.load`` matches state_dict entries to checkpoint entries BY NAME, but a checkpoint's stored key
    # names can differ from the built model's ``state_dict`` names, and from_pretrained bridges the gap on
    # load. E.g. Qwen2.5-VL's checkpoint stores ``model.layers.*`` / ``visual.*`` while transformers>=5 nests
    # them as ``model.language_model.layers.*`` / ``model.visual.*`` -- a pure prefix-insertion with the leaf
    # path preserved, which the model carries no ``_checkpoint_conversion_mapping`` attribute for. A raw by-name
    # match would SILENTLY skip every renamed weight (they stay ``to_empty`` garbage -> a wrecked model that
    # only surfaces as a confusing downstream crash). Reproduce from_pretrained's result generically via
    # :func:`_map_ckpt_keys_to_model`: exact -> conversion-mapping regexes -> longest UNIQUE suffix.
    # (sharded or plain) params + persistent buffers; a DTensor's shape is its global one.
    model_shapes = {k: t.shape for k, t in model.state_dict().items()}
    target_names = set(load_names) & set(model_shapes)
    # {ckpt_key: model_key}, mapped against the WHOLE model and then cut to what this load fills: a suffix
    # unique among the targets alone can still belong to a key outside them (an expert's, when only the
    # dense part loads here).
    key_map = {
        ck: mk
        for ck, mk in _map_ckpt_keys_to_model(model, ckpt_keys, set(model_shapes)).items()
        if mk in target_names
    }
    # The export carries the checkpoint tensors the model has no parameter for over raw, as the load
    # that built the model recorded them (record_unplaced_source_keys): filling one here too would
    # write it twice, once quantized.
    carried = sorted(
        set(key_map) & set(getattr(model, "_modelopt_unplaced_source_keys", None) or ())
    )
    if carried:
        raise RuntimeError(
            f"meta+DCP load: {len(carried)} checkpoint tensor(s) map onto this model's params by name, "
            f"but its load recorded them as having no param, for the export to carry over raw: "
            f"{carried[:3]}. The two would write them twice."
        )
    # dcp.load refuses a checkpoint tensor whose shape is not its model tensor's only once to_empty has
    # re-emptied the model; refuse it while the model is untouched.
    mismatched = sorted(
        f"{ck} {tuple(ckpt_md[ck].size)} -> {mk} {tuple(model_shapes[mk])}"
        for ck, mk in key_map.items()
        if ckpt_md[ck].size != model_shapes[mk]
    )
    if mismatched:
        raise RuntimeError(
            f"meta+DCP load: {len(mismatched)} checkpoint tensor(s) differ in shape from the model "
            f"tensor they map onto, as when the model's config does not match the checkpoint: "
            f"{mismatched[:3]}"
        )
    # Build load_sd in the model's state_dict ORDER (identical on every rank) -- NOT key_map/set iteration
    # order, which is hash-randomized per process. dcp.load's collective load planner deadlocks if ranks
    # present the same keys in different orders, so a deterministic, rank-consistent order is required.
    inv = {mk: ck for ck, mk in key_map.items()}  # model_key -> ckpt_key (1:1 on model keys)
    mapped_names = set(key_map.values())
    # Fail fast, BEFORE the load, if the mapping left over half of the TARGET WEIGHT unloaded -- a key scheme
    # this path did not resolve. Weight by NUMEL, not param COUNT: a fused-expert MoE (Qwen3-MoE: per-expert
    # checkpoint -> grouped ``experts.gate_up_proj``/``down_proj`` model params DCP can't fuse) leaves only
    # ~2 params/layer unloaded but they are ~95% of the weight, so a count-based guard misses it. A smaller
    # shortfall can be tied weights, which ``tie_weights`` fills after the load; whatever it does not fill
    # fails there. Every alias of a tied param counts, as in unplaced_source_keys: the checkpoint can
    # store the one named_parameters() drops by default. Only meta params weigh in: one already real
    # keeps its value without a checkpoint tensor.
    want_params: set[str] = set()
    numel: dict[str, int] = {}
    for n, p in model.named_parameters(remove_duplicate=False):
        if n in target_names:
            want_params.add(n)
            if p.is_meta:
                numel[n] = p.numel()
    unloaded = numel.keys() - mapped_names
    total_nel = max(1, sum(numel.values()))
    unloaded_nel = sum(numel[n] for n in unloaded)
    if 2 * unloaded_nel > total_nel:
        raise RuntimeError(
            f"meta+DCP load: {len(unloaded)}/{len(numel)} meta params ({100 * unloaded_nel / total_nel:.0f}% "
            f"of their weight) matched no checkpoint key -- the checkpoint's key naming was not resolved "
            f"to this model's state_dict (exact + conversion-mapping + suffix match all missed), or it needs a "
            f"weight conversion DCP can't do (FUSED EXPERTS, split qkv, on-the-fly dequant). Unloaded e.g.: "
            f"{sorted(unloaded)[:5]}. Replicating the dense part (MODELOPT_FSDP_SIZE=1) loads it through "
            "from_pretrained instead."
        )
    loaded_bufs = mapped_names - {n for n, _ in model.named_parameters(remove_duplicate=False)}
    if loaded_bufs:
        warn_rank_0(
            f"meta+DCP load: loaded {len(loaded_bufs)} persistent buffer(s) from checkpoint: "
            f"{sorted(loaded_bufs)[:5]}"
        )
    # Snapshot real buffers (to_empty re-empties them), materialize params, restore buffers. Materialize
    # selectively, module by module, so the real experts are not re-emptied. That is safe only because no
    # root is fully_shard-ed: a per-module loop corrupts a ROOT-sharded FSDP state (its state_dict then
    # returns wrong keys).
    saved_buffers = {n: b.detach().clone() for n, b in model.named_buffers() if not b.is_meta}
    # A module's ``to_empty`` re-empties every direct param, so a param already real beside a meta one
    # (a router's ``e_score_correction_bias`` next to its meta ``weight``) is kept here too: nothing
    # outside ``load_names`` would refill it. Every name counts: ``to_empty`` gives a param the module
    # registers twice a tensor under each name.
    saved_params: dict[str, torch.Tensor] = {}
    emptied: set[str] = set()
    for name, m in model.named_modules():
        direct = dict(m.named_parameters(recurse=False, remove_duplicate=False))
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
    # The clones are restored; only the names of the params they kept are needed from here on.
    kept_names = set(saved_params)
    del saved_buffers, saved_params

    full_sd = model.state_dict()  # the materialized tensors: to_empty replaced the meta params
    load_sd = {inv[mk]: full_sd[mk] for mk in full_sd if mk in inv}
    dcp.load(load_sd, storage_reader=reader)
    # Taken before tying, which can swap a filled param out of its module; in state_dict order, the
    # same on every rank, so every rank keeps the same param of a tied group.
    filled = {n: model.get_parameter(n) for n in full_sd if n in want_params and n in mapped_names}
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
    # A re-emptied param no checkpoint tensor filled, and that tying did not make a loaded param, still
    # holds ``to_empty``'s memory, whether or not ``load_names`` named it. from_pretrained would
    # initialize it; this loader can't run the model's init on shards, so it fails rather than return
    # the model. Every rank reaches the same verdict.
    unfilled = sorted(
        n for n in emptied - mapped_names - kept_names if id(model.get_parameter(n)) not in loaded
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

    It records no distribution plan: a caller records its own, as :func:`load_pretrained` does
    (:func:`~modelopt.torch.utils.distributed.set_distribution_plan`).
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


def _auto_class(hf_config, require_lm_head: bool = True, trust_remote_code: bool = False) -> Any:
    """The auto class that builds ``hf_config``'s whole model.

    A causal LM builds through ``AutoModelForCausalLM``, head included: ``require_lm_head`` only
    applies to a model whose architecture reads as a VLM's -- a nested text config, a vision
    config, or a ``*ForConditionalGeneration`` architecture -- or that is an encoder-decoder model
    (``is_encoder_decoder``), whether or not it names its architecture. Without ``require_lm_head``
    such a model builds through ``AutoModel``, its base model, when that can build it (the config
    is a native one, or a trusted remote one registering ``AutoModel``): most base models leave the
    head out, since it is not required, though some keep it (Llama4's). Otherwise it builds through
    the remote ``AutoModelForCausalLM`` when the checkpoint registers its
    ``*ForConditionalGeneration`` there, its code is trusted and it registers no image-text class,
    else through its seq2seq class if it is an encoder-decoder model, else through
    ``AutoModelForImageTextToText`` if that recognizes the config or, its code trusted, the
    checkpoint registers one, else through ``AutoModelForCausalLM`` if that recognizes it (as it
    does Phi-4-multimodal), else still through ``AutoModel``.
    """
    architectures = getattr(hf_config, "architectures", None) or []
    is_vlm = (
        getattr(hf_config, "text_config", None) is not None
        or getattr(hf_config, "vision_config", None) is not None
        or any("conditionalgeneration" in str(arch).lower() for arch in architectures)
    )
    if not is_vlm and not getattr(hf_config, "is_encoder_decoder", False):
        return AutoModelForCausalLM
    auto_map = getattr(hf_config, "auto_map", None) or {}
    if not require_lm_head and (
        type(hf_config) in AutoModel._model_mapping
        or (trust_remote_code and "AutoModel" in auto_map)
    ):
        return AutoModel
    remote_with_head = (
        trust_remote_code
        and "conditionalgeneration" in str(auto_map.get("AutoModelForCausalLM", "")).lower()
    )
    if remote_with_head and "AutoModelForImageTextToText" not in auto_map:
        return AutoModelForCausalLM
    for seq2seq in (AutoModelForSeq2SeqLM, AutoModelForSpeechSeq2Seq):
        if type(hf_config) in seq2seq._model_mapping:
            return seq2seq
    if type(hf_config) in AutoModelForImageTextToText._model_mapping or (
        trust_remote_code and "AutoModelForImageTextToText" in auto_map
    ):
        return AutoModelForImageTextToText
    if type(hf_config) in AutoModelForCausalLM._model_mapping:
        return AutoModelForCausalLM
    return AutoModel


# The share of one GPU's memory a rank may take with the dense part replicated: that whole copy and
# the rank's share of the routed experts. The rest is left to calibration's activations and
# communication buffers.
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
    if value in ("world", str(world_size)):
        return "fsdp"
    raise ValueError(
        f"MODELOPT_FSDP_SIZE={value!r}: expected auto, 1 (replicate the dense part) or world "
        f"(FSDP2-shard it over all {world_size} ranks)."
    )


def _ep_degree(ep_size) -> int | str:
    """``ep_size`` as an EP degree or ``"auto"``; ``None`` reads ``MODELOPT_EP_DEGREE``."""
    value = env.ep_degree() if ep_size is None else ep_size
    if str(value).lower() in ("auto", "0"):
        return "auto"
    try:
        degree = int(value)
    except (ValueError, TypeError):  # TypeError: an argument that is no number at all, as [2]
        degree = 0
    if isinstance(value, float) and value != degree:  # int() truncates a fraction
        degree = 0
    if degree < 1:
        name = "MODELOPT_EP_DEGREE" if ep_size is None else "ep_size"
        raise ValueError(f"{name}={value!r}: expected auto, 0 (auto) or an EP degree.")
    return degree


def _decide_distribution(
    ckpt_path: str, hf_config, world_size: int, ep_size, dense, trust_remote_code: bool, auto_class
) -> DistributionPlan:
    """The plan :func:`resolve_distribution_plan` returns, decided by this rank alone."""
    # What the caller set the EP degree with, for a refusal to name it.
    knob = "MODELOPT_EP_DEGREE" if ep_size is None else "ep_size"
    if world_size == 1:
        # Nothing to shard over, so nothing about the model matters, nor do the knobs: a shell still
        # exporting a multi-GPU run's would otherwise reshape a single process's load. The caller's
        # own arguments still count, as dense="fsdp" for FSDP2 on one rank.
        single = 1 if ep_size is None else _ep_degree(ep_size)
        return DistributionPlan(
            ep_size=single if isinstance(single, int) else 1,
            dense="replicate" if dense in (None, "auto") else dense,
        )
    ep_size = _ep_degree(ep_size)
    dense = _dense_from_env(world_size) if dense is None else dense
    if (
        (auto_class not in (None, AutoModelForCausalLM) or _init_quantized_weights())
        and ep_size in ("auto", 1)
        and dense in ("auto", "replicate")
    ):
        # The sharded loads refuse this class and init_quantized_weights: load it whole, unbuilt.
        return DistributionPlan(world_size)
    if auto_class not in (None, AutoModelForCausalLM):
        # Refused here, rather than build the decision's own pick of class, which may not build.
        name = getattr(auto_class, "__name__", repr(auto_class))
        raise NotImplementedError(
            f"A sharded load builds its model class itself, a causal LM's or a text-only VLM's "
            f"image-text class, so it takes no auto_class={name} while ep_size={ep_size!r}, "
            f"dense={dense!r} ask to shard. To give every rank the whole model, plan ep_size=1 and "
            "dense='replicate' (MODELOPT_EP_DEGREE=1, MODELOPT_FSDP_SIZE=1)."
        )
    # Building writes to the config it builds from, and only rank 0 decides: build from a copy, so
    # the config returned with the plan, which a sharded load builds from, is alike on every rank.
    hf_config = copy.deepcopy(hf_config)
    try:
        model = build_meta_causal_lm(
            ckpt_path,
            trust_remote_code,
            None,
            hf_config,
            auto_class=_auto_class(hf_config, trust_remote_code=trust_remote_code),
        )
        # init_empty_weights re-wraps every parameter it registers, which unties what from_config
        # tied: re-tie, so a tied embedding and LM head count once, as the loaded model holds them.
        model.tie_weights()
    except Exception as e:
        raise RuntimeError(
            f"Deciding how to distribute {hf_config.model_type} needs its model built on the meta "
            f"device, which failed: {type(e).__name__}: {e}"
        ) from e
    ep_plan = getattr(_ep_config(hf_config), "base_model_ep_plan", None) or {}
    experts = _routed_expert_modules(model, ep_plan)
    num_experts = _num_experts_from_config(_ep_config(hf_config)) if experts else 0
    if ep_plan and not experts and ep_size != 1 and _num_experts_from_config(_ep_config(hf_config)):
        # The plan names routed experts the built model has none of under those names: deciding
        # without them would replicate every expert on every rank, and the expert-parallel load
        # finds them the same way.
        containers = ", ".join(sorted(_routed_expert_keys(ep_plan))) or "none"
        raise ValueError(
            f"{hf_config.model_type} has routed experts and a base_model_ep_plan, but none of the "
            f"plan's expert containers ({containers}) names a module of the built model, so they "
            f"can be neither counted nor sharded. Set {knob}=1 to load it without expert "
            "parallelism."
        )
    if experts and not num_experts and ep_size != 1:
        # The mirror case: the containers are there, but no count to choose a degree by, which
        # would leave every expert replicated on every rank.
        raise ValueError(
            f"{hf_config.model_type}'s base_model_ep_plan names routed experts the built model has, "
            "but its config gives no expert count (num_experts, num_local_experts, "
            f"n_routed_experts or num_routed_experts) to shard them by. Set {knob}=1 to load it "
            "without expert parallelism."
        )
    if ep_size == "auto":
        ep_size = _choose_ep_degree(world_size, num_experts)
    elif ep_size > 1 and world_size % ep_size:
        raise ValueError(f"{knob}={ep_size} must divide the world size ({world_size}).")
    elif ep_size > 1 and (num_experts == 0 or num_experts % ep_size):
        raise ValueError(
            f"{knob}={ep_size} must divide the routed experts {hf_config.model_type} shards "
            f"({num_experts})."
        )
    if dense == "auto":
        # A rank holds a whole copy of what expert parallelism does not shard, and its 1/ep_size
        # share of the routed experts it does.
        sharded = (
            {id(p) for m in experts.values() for p in m.parameters()} if ep_size > 1 else set()
        )
        held = sum(
            p.numel() * p.element_size() // (ep_size if id(p) in sharded else 1)
            for p in model.parameters()
        )
        memory = _gpu_memory()
        too_big = memory is not None and held > _REPLICATED_SHARE * memory
        dense = "fsdp" if too_big else "replicate"
    return DistributionPlan(world_size, ep_size, dense)


def resolve_distribution_plan(
    ckpt_path: str,
    *,
    config: PretrainedConfig | None = None,
    auto_class: Any = None,
    trust_remote_code: bool = False,
    ep_size: int | str | None = None,
    dense: str | None = None,
    revision: str | None = None,
    token: str | bool | None = None,
    cache_dir: str | os.PathLike | None = None,
    local_files_only: bool = False,
    subfolder: str | None = None,
) -> tuple[DistributionPlan, PretrainedConfig]:
    """Decide how this run distributes ``ckpt_path``'s model over its ranks.

    The one place a run decides it: a distributed load loads the model by the plan this returns and
    records it on the model (:func:`~modelopt.torch.utils.distributed.set_distribution_plan`), and
    every later step reads it back from there
    (:func:`~modelopt.torch.utils.distributed.distribution_plan`). Returns ``(plan, hf_config)``,
    ``hf_config`` being this rank's own copy of the config (rank 0 decides the plan from its copy),
    for a sharded load to build from: :func:`load_pretrained` takes it only with a plan that shards.

    Rank 0 decides, from the model built on the meta device: from its parameters rather than
    estimates of them, and from its routed experts as an expert-parallel load finds them. Every rank
    takes rank 0's plan (:func:`~modelopt.torch.utils.distributed.broadcast_from`): the inputs are
    read locally, ranks deciding for themselves could disagree, and ranks loading different plans
    hang. Collective: every rank must call it.

    * ``ep_size``: the expert-parallel degree, or ``"auto"``, the largest that divides both the
      world and the routed-expert count of a model whose config carries a ``base_model_ep_plan``.
    * ``dense``: ``"replicate"`` or ``"fsdp"`` for the rest of the model, or ``"auto"``, which
      FSDP2-shards it when a whole copy of it and the rank's share of the routed experts would take
      more than 60% of one GPU's memory. That GPU is rank 0's, by its total memory, so ``"auto"``
      assumes the run's GPUs are alike; a rank 0 without a GPU replicates it.

    ``None`` reads ``MODELOPT_EP_DEGREE`` and ``MODELOPT_FSDP_SIZE`` (``1`` replicates; ``world``,
    or the world size as a number, shards) from rank 0's environment, both defaulting to
    ``"auto"``. A single process reads neither and holds the whole model, unless ``dense="fsdp"``
    asks for FSDP2 on one rank.

    ``revision``, ``token``, ``cache_dir``, ``local_files_only`` and ``subfolder`` locate the
    checkpoint as :func:`load_pretrained` takes them, so the plan is decided for what the load reads:
    all but ``subfolder`` resolve a Hub model ID (:func:`ensure_local_checkpoint`), and the config
    read takes all but ``revision``, for custom code its ``auto_map`` names in another Hub repo. It
    returns no path, so a caller that loads the checkpoint resolves it first, with
    :func:`ensure_local_checkpoint`, and passes the local directory: the plan is then decided for
    the snapshot the load reads.

    ``config``, the caller's own config, is decided from instead of the checkpoint's. ``auto_class``
    is the class the load builds with: with one the sharded loads refuse, or inside
    ``init_quantized_weights``, and nothing asking to shard, the plan loads the model whole, decided
    without a build.
    """
    # Each rank reads the checkpoint for itself. One that cannot must fail every rank here, rather
    # than leave the others waiting in a broadcast below for a rank that already raised.
    read: list[tuple[str, PretrainedConfig]] = []

    def checkpoint_read() -> bool:
        _, local_path = ensure_local_checkpoint(
            ckpt_path,
            revision=revision,
            token=token,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        if subfolder:
            local_path = os.path.join(local_path, subfolder)
        if config is not None:
            read.append((local_path, copy.deepcopy(config)))
            return True
        hf_config = _read_config(
            local_path,
            auto_class,
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
            ckpt_path, hf_config, world_size, ep_size, dense, trust_remote_code, auto_class
        ),
    )
    if plan != DistributionPlan():
        print_rank_0(f"Distribution: {plan!r}")
    return plan, hf_config


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


# The checkpoint formats load_and_prepare_ep dequantizes as it loads, beside unquantized ones (None).
# mxfp4 is for experts stored in the standard layout.
_EP_SOURCE_FORMATS = frozenset({None, "mxfp4"})


def _ckpt_quant_method(cfg):
    """Return the checkpoint's native quantization method, or ``None``.

    Return the checkpoint's native quantization method (e.g. ``"fp8"`` for block-fp8 DeepSeek-V3),
    or ``None`` if the checkpoint isn't natively quantized. Checks the top config, then a VLM's
    ``text_config`` or ``llm_config``.
    """
    for c in (cfg, getattr(cfg, "text_config", None), getattr(cfg, "llm_config", None)):
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

    def _ep_mesh(*args, **kwargs):
        index = torch.cuda.current_device() if ep_mesh.device_type == "cuda" else None
        return torch.device(ep_mesh.device_type, index), ep_mesh

    original = mixin.initialize_distributed_mesh
    mixin.initialize_distributed_mesh = _ep_mesh
    try:
        yield
    finally:
        mixin.initialize_distributed_mesh = original


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
    :func:`load_pretrained` does for every load. The model builds with its LM head where an auto
    class builds one; ``require_lm_head=False`` builds a VLM without it, through ``AutoModel`` (a
    causal LM always keeps its head).

    Builds an ``(edp, ep)`` device mesh and loads experts EP-sharded over the ``ep`` submesh
    (``enable_expert_parallel=True`` -- each rank owns ``num_experts / ep`` experts as grouped-GEMM
    shards), while ATTENTION and dense params stay REPLICATED (so EP has no KV-head-divisibility cap).
    The model's ``base_model_ep_plan`` is swapped to the ``torch_a2a`` styles: the router emits GLOBAL
    unmasked expert ids and the experts forward routes each token to its expert-owning rank by a NCCL
    all-to-all (dispatch) -> quantizable ``_grouped_linear`` -> all-to-all (combine), rather than every
    rank computing all tokens followed by an EP all-reduce. Every rank is data-parallel (sees DIFFERENT
    tokens), so calibration MUST shard the data per-rank.

    The ``edp`` dimension (present when ``plan.ep_size < world_size``) is additional data parallelism
    across EP groups: non-expert params stay REPLICATED per rank and each EP group calibrates on
    different data; experts stay EP-sharded (resident). No FSDP and no concurrent ``edp``/``ep``
    collectives -> no deadlock by construction. Costs more per-rank memory (non-experts + experts
    replicated across edp) but is the reliable path at scale.

    EP CANNOT be applied to a meta-init model (transformers' ``GroupedGemmParallel.shard_tensor``
    slices materialized tensors), so this loads real weights via ``from_pretrained`` over the ``ep``
    submesh. Requires transformers >= 5.7: the MoE EP framework (ExpertsInterface / grouped-GEMM EP
    styles / enable_expert_parallel) is in 5.6, but 5.6 builds a model only with the experts
    implementations it ships, never ``torch_a2a``. (A model whose expert-parallel support landed
    later states the release it needs in its spec.)

    Assumes the caller has already initialized (and device-bound) the process group -- distributed
    bring-up is the launcher's job, not this loader's (hf_ptq.main() does it via ``distributed_run``).
    The expert-parallel mesh stays recoverable from the model
    (:func:`~modelopt.torch.utils.moe.expert_parallel_mesh`), once the caller has recorded the plan.
    A refusal on any rank -- of its transformers, the config, the checkpoint, or in the load --
    raises on every rank, so no rank is left waiting in a collective for one that already failed.
    Without ``hf_config`` every rank reads the checkpoint's config itself, from the Hub for a Hub ID,
    before rank 0 makes the checkpoint local: on many ranks, pass it (as ``hf_ptq`` does).
    """
    # Imported here, so looked up when the load runs: its tests stand in a process group and a
    # transformers release by patching these two modules.
    from modelopt.torch.utils.distributed import ParallelState, is_initialized, size
    from modelopt.torch.utils.moe import (
        _EXPERTS_STYLES,
        _ep_config,
        _num_experts_from_config,
        _routed_expert_keys,
        _routed_expert_modules,
        register_torch_a2a,
        require_transformers,
    )

    if not plan.is_expert_parallel:
        raise ValueError(f"{plan!r} shards no routed experts, which is what this load is for.")
    if plan.is_fsdp_dense:
        raise NotImplementedError(f"{plan!r}: this load replicates the dense part on every rank.")
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
    world_size, ep_degree, edp = plan.world_size, plan.ep_size, plan.edp
    # Each rank checks its own transformers, the config it reads or was handed, and the plan against
    # it: one refusing fails every rank here, before the checkpoint's fetch and the meshes.
    with _refused_together("the expert-parallel load's checks", alone=False):
        # Hard floor: below 5.6 there is no ExpertsInterface / grouped-GEMM EP styles, and
        # enable_expert_parallel silently shards nothing (no error); 5.6 checks a model's experts
        # implementation against the ones it ships, so from_pretrained would refuse torch_a2a.
        require_transformers(
            "5.7", "ModelOpt DP=EP expert-parallel PTQ (transformers MoE EP framework)"
        )
        if hf_config is None:
            config_kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code}
            if attn_implementation is not None:
                config_kwargs["attn_implementation"] = attn_implementation
            hf_config = AutoConfig.from_pretrained(ckpt_path, **config_kwargs)
        quant_method = _ckpt_quant_method(hf_config)
        if quant_method not in _EP_SOURCE_FORMATS:
            # transformers' own quantizer would load it as stored, for PTQ to treat as plain weights.
            dequantized = ", ".join(sorted(f for f in _EP_SOURCE_FORMATS if f))
            raise NotImplementedError(
                f"The expert-parallel load cannot dequantize a {quant_method!r} checkpoint: it loads "
                f"unquantized checkpoints and dequantizes {dequantized}. Start from an unquantized copy "
                "of the model."
            )
        # The swapped plan is this load's own: the caller's config keeps the plan its checkpoint declares.
        hf_config = copy.deepcopy(hf_config)
        ep_cfg = _ep_config(hf_config)  # for VLMs the plan lives on .text_config
        ep_plan = dict(getattr(ep_cfg, "base_model_ep_plan", None) or {})
        if not ep_plan:
            raise ValueError("expert-parallel PTQ requires the model to declare base_model_ep_plan")
        if not _routed_expert_keys(ep_plan):
            raise ValueError(
                f"base_model_ep_plan={ep_plan} styles no routed experts ({', '.join(_EXPERTS_STYLES)} "
                "or grouped_gemm), so this load would leave them whole on every rank."
            )
        # Only grouped_gemm entries shard an expert's weights here: the torch_a2a styles the
        # containers take for the load shard nothing themselves (packed_* entries are dropped).
        unsharded = sorted(
            key
            for key in _routed_expert_keys(ep_plan)
            if not any(
                style == "grouped_gemm" and name.startswith(key + ".")
                for name, style in ep_plan.items()
            )
        )
        if unsharded:
            raise ValueError(
                f"base_model_ep_plan={ep_plan} shards no weights of {', '.join(unsharded)} through "
                "grouped_gemm, the only expert-weight style this load shards by, so every rank would "
                "hold every expert."
            )
        # The load drops any other style an expert weight takes, so that weight would load whole
        # beside its sharded siblings, and the grouped forward would pair slices of different experts.
        containers = _routed_expert_keys(ep_plan)
        dropped = sorted(
            name
            for name, style in ep_plan.items()
            if style not in ("grouped_gemm", "ep_router", *_EXPERTS_STYLES)
            and name not in containers
            and any(name.startswith(key + ".") for key in containers)
        )
        if dropped:
            raise ValueError(
                f"base_model_ep_plan={ep_plan} styles {', '.join(dropped)} by a style this load "
                "drops, so those expert weights would load whole on every rank beside their sharded "
                "siblings."
            )
        # The all-to-all transport finds an expert's rank by dividing its id by this rank's share of
        # the experts, which holds only when every rank holds as many. Counted in the config that holds
        # the plan, as the distribution plan's decision counts them. A count the config names none of
        # the ways this reads is checked once the experts load, from their weights.
        num_experts = _num_experts_from_config(ep_cfg)
        if num_experts % ep_degree:
            raise ValueError(
                f"{plan!r}: ep_size={ep_degree} must divide the {num_experts} routed experts, which the "
                "all-to-all transport shares out equally over the expert-parallel ranks."
            )
    # transformers 5.6 on, which the floor above makes sure of.
    from torch.distributed.device_mesh import init_device_mesh
    from transformers.distributed import DistributedConfig

    dtype = getattr(hf_config, "dtype", None) or torch.bfloat16
    # The weights load, and the checkpoint recorded for the export to read the unplaced tensors back
    # from, take the local snapshot: a Hub ID recorded there would leave the export nothing to read.
    # Resolved once the plan and the config pass, so a refused load downloads nothing. Rank 0 makes
    # it local; a rank that cannot see where fails every rank.
    with _refused_together("the local checkpoint", alone=False):
        _, ckpt_path = ensure_local_checkpoint(ckpt_path)

    # Two INDEPENDENT meshes (NeMo-AutoModel style), named by the model part they govern -- two
    # factorizations of the same world:
    #   sparse_moe_mesh = (edp, ep): experts are SHARDED over the ``ep`` axis and REPLICATED over ``edp``;
    #       the ``ep`` submesh also drives from_pretrained's expert sharding + the a2a transport.
    #   dense_mesh = (dp,): dense/non-expert params are data-parallel-REPLICATED over the WHOLE world.
    # This subsumes the old a2a-vs-partial split: a2a is simply edp == 1, and dense amax
    # always reduces over the whole world (fixing the DP>1 case, where the old dp submesh missed the ep dim).
    sparse_moe_mesh = init_device_mesh("cuda", (edp, ep_degree), mesh_dim_names=("edp", "ep"))
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
    # WHERE it goes is version-dependent: from transformers 5.17 from_pretrained refuses tp_plan
    # beside distributed_config ("Set tensor-parallel options on DistributedConfig when using it"),
    # while DistributedConfig has no tp_plan field before 5.10, nor in 5.11 and 5.12. Passing
    # tp_plan ALONE is not an option either -- the 5.17+ back-compat path rebuilds the config as
    # DistributedConfig(tp_plan=..., tp_size=...) and DROPS enable_expert_parallel, silently losing
    # EP. So each argument goes on DistributedConfig where it has a field for it, and tp_plan
    # otherwise to from_pretrained: the dataclass fields tell them apart on any release.
    _dc_fields = {f.name for f in dataclasses.fields(DistributedConfig)}
    _dc_has_tp_plan = "tp_plan" in _dc_fields
    dc_kwargs: dict[str, Any] = {"enable_expert_parallel": True}
    if _dc_has_tp_plan:
        dc_kwargs["tp_plan"] = "auto"
    if _transformers_mesh_builder() is not None and "tp_size" in _dc_fields:
        # This release sizes its own mesh from tp_size; _experts_sharded_over swaps in the EP mesh.
        dc_kwargs["tp_size"] = ep_degree
    distributed_config = DistributedConfig(**dc_kwargs)
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
    if quant_method == "mxfp4":
        # Native mxfp4 checkpoints (fused `gate_up_proj_blocks`/`_scales` + biases) must be
        # DEQUANTIZED to bf16 on load so ModelOpt can insert nvfp4/fp8 quantizers on the fused bf16 experts
        # (the a2a grouped_gemm expects bf16). transformers dequantizes blocks->bf16 during from_pretrained.
        # NB: the single-process loader needed device_map="sequential" to dodge a CUDA illegal-access in the
        # mxfp4 dequant under multi-GPU device_map="auto"; the EP path uses device_mesh (no device_map), so
        # watch for an EP-shard-awareness issue in the dequant convert-op.
        from transformers import Mxfp4Config

        from_pretrained_kwargs["quantization_config"] = Mxfp4Config(dequantize=True)
        print_rank_0("[ep] native mxfp4 checkpoint -> Mxfp4Config(dequantize=True) -> bf16")
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
    # The checkpoint's own plan, for the config the model keeps (below); None: a class default.
    declared_plan = vars(ep_cfg).get("base_model_ep_plan")
    ep_cfg.base_model_ep_plan = swapped_plan
    from_pretrained_kwargs["config"] = hf_config
    print_rank_0(f"[ep] ep_plan={swapped_plan} experts_implementation=torch_a2a")
    # A rank failing in the load (out of memory, a file it cannot read) fails every rank here,
    # before the warmup's collectives.
    with _refused_together("the expert-parallel load", alone=False):
        # The class that builds the whole checkpoint -- with its lm_head where an auto class builds one,
        # so the head is calibrated and exported and the generation preview works -- chosen as the
        # distribution plan's decision chose it, unless the caller asked for a VLM without its head
        # (require_lm_head=False). The per-model plugin (models/) has already installed any on-the-fly
        # weight decompression as a transformers monkeypatch.
        loader = _auto_class(hf_config, require_lm_head, trust_remote_code=trust_remote_code)
        print_rank_0(f"[ep] loader={loader.__name__} (require_lm_head={require_lm_head})")
        with _experts_sharded_over(ep_submesh):
            model = loader.from_pretrained(ckpt_path, **from_pretrained_kwargs)
        model.eval()
        # The swapped styles drove the sharding, and the model keeps the plan it was sharded by
        # (``model.tp_plan``, copied when it was built). The config it keeps goes back to the
        # checkpoint's plan: an export writes the config out, and nothing without ModelOpt knows the
        # torch_a2a styles.
        kept_config = getattr(model, "config", None)
        if kept_config is not None:
            kept_ep_config = _ep_config(kept_config)
            if declared_plan is None:
                vars(kept_ep_config).pop("base_model_ep_plan", None)
            else:
                kept_ep_config.base_model_ep_plan = declared_plan
        # Force the experts forward to dispatch to `torch_a2a`: from_pretrained's `experts_implementation`
        # kwarg resolves non-fp8 impls back to "grouped_mm", but @use_experts_implementation reads
        # `config._experts_implementation` at FORWARD time, so set it on every (sub)config now. (Without
        # this the torch_a2a_experts style passes a 5th process_group arg to grouped_mm_experts_forward ->
        # TypeError.) Unlike the plan it stays set, and an export still writes no torch_a2a: transformers
        # keeps it as ``_experts_implementation_internal``, which it leaves out of a serialized config.
        for _m in model.modules():
            _cfg = getattr(_m, "config", None)
            if _cfg is not None and hasattr(_cfg, "_experts_implementation"):
                _cfg._experts_implementation = "torch_a2a"

        # The checkpoint tensors the model has no parameter for (an MTP head), worked out once, at load,
        # for the export to carry.
        record_unplaced_source_keys(model, ckpt_path)

        # DP=EP all-to-all: no FSDP. Non-experts stay replicated per rank (from_pretrained already placed
        # them on this rank's GPU); experts stay EP-sharded on the ep submesh, which the model carries as
        # ``_device_mesh`` for calibration to re-install the all-to-all forwards over (expert_parallel_mesh).
        # transformers sets it from the mesh this load hands it, through internals that differ by release;
        # set here, the model carries it on every release.
        model._device_mesh = ep_submesh
        if hasattr(model, "tie_weights"):
            model.tie_weights()
        if freeze:
            model.requires_grad_(False)

        # The routed-expert containers, from the SAME base_model_ep_plan that drove the sharding. A MoE model
        # loaded WITHOUT EP (dense DP) never reaches this loader, so its experts stay dense.
        expert_modules = _routed_expert_modules(model, ep_plan)
        if not expert_modules:
            raise ValueError(
                f"base_model_ep_plan={ep_plan} names routed-expert containers that no module of "
                f"{type(model).__name__} matches, so calibration would reduce the experts' amax over "
                "ranks holding different experts. Quantize the model without expert parallelism."
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
    print_rank_0(
        f"[ep] seeded the expert-parallel state of {len(expert_modules)} routed-expert "
        "module(s) from base_model_ep_plan"
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
