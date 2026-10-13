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

"""HuggingFace model loading: :func:`load_pretrained`, whole or distributed."""

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
):
    """Build a meta-init causal LM (no real storage allocated)."""
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
        model = AutoModelForCausalLM.from_config(
            hf_config, dtype=dtype, trust_remote_code=trust_remote_code
        )
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


def _distributed_session():
    """The active ``ModelOptSession`` when it runs distributed, else ``None``."""
    from modelopt.torch.utils.distributed import get_active_mo_session

    session = get_active_mo_session(required=False)
    return session if session.initialized and session.distributed else None


def _decide_distribution(ckpt_path: str, trust_remote_code: bool, ep_size, fsdp_size):
    """:func:`resolve_distribution_plan` for a local ``ckpt_path``."""
    from modelopt.torch.utils import env
    from modelopt.torch.utils.moe import decide_distribution_plan

    session = _distributed_session()
    if session is None:
        return None, None
    hf_config = AutoConfig.from_pretrained(ckpt_path, trust_remote_code=trust_remote_code)
    plan = decide_distribution_plan(
        hf_config,
        session.world_size,
        ep_size=env.ep_degree() if ep_size is None else ep_size,
        fsdp_size=env.fsdp_size() if fsdp_size is None else fsdp_size,
    )
    if session.is_main:
        print(f"Distribution: {plan!r}", flush=True)
    return plan, hf_config


def resolve_distribution_plan(
    ckpt_path: str,
    *,
    trust_remote_code: bool = False,
    ep_size: "int | str | None" = None,
    fsdp_size: "int | str | None" = None,
) -> tuple[DistributionPlan | None, PretrainedConfig | None]:
    """Decide how :func:`load_pretrained` distributes ``ckpt_path`` in this session.

    Returns ``(plan, hf_config)``: a :class:`~modelopt.torch.utils.distributed.DistributionPlan`
    and the config it was decided from, which a sharded load takes back as
    ``load_pretrained(ckpt_path, distribution=plan, hf_config=hf_config)`` rather than read it
    again. Outside a distributed :class:`~modelopt.torch.utils.distributed.ModelOptSession`:
    ``(None, None)``, the model loads whole. ``load_pretrained`` decides the same way on its own;
    deciding first is for a caller that loads an unsharded model differently.

    :func:`~modelopt.torch.utils.moe.decide_distribution_plan` decides, once on rank 0, from
    ``ep_size`` and ``fsdp_size``; ``None`` reads ``MODELOPT_EP_DEGREE`` and ``MODELOPT_FSDP_SIZE``
    (:mod:`modelopt.torch.utils.env`), which default to ``"auto"``. Collective: every rank must
    call it.
    """
    if _distributed_session() is None:
        return None, None
    _, ckpt_path = ensure_local_checkpoint(ckpt_path)
    return _decide_distribution(ckpt_path, trust_remote_code, ep_size, fsdp_size)


def _route(distribution: DistributionPlan | str) -> str:
    """Which load ``distribution`` asks for: a sharded one, ``"replicated"``, or ``"sequential"``."""
    if isinstance(distribution, str):
        if distribution in ("sequential", "fsdp2"):
            return _FSDP2 if distribution == "fsdp2" else "sequential"
        raise ValueError(
            f"distribution={distribution!r}: expected None, 'sequential', 'fsdp2' or a "
            "DistributionPlan."
        )
    plan, session = distribution, _distributed_session()
    world_size = session.world_size if session is not None else 1
    if plan.world_size != world_size:
        raise ValueError(f"{plan!r} is for {plan.world_size} ranks, but this run has {world_size}.")
    if plan.is_expert_parallel:
        raise NotImplementedError(
            f"{plan!r} shards the routed experts, which load_pretrained has no load for. Plan "
            "ep_size=1 (MODELOPT_EP_DEGREE=1) to give every rank the whole model, or shard it "
            "with FSDP2."
        )
    if plan.is_fsdp_dense:
        if plan.fsdp_size != plan.world_size:
            raise NotImplementedError(
                f"{plan!r}: without expert parallelism the FSDP2 load shards over the whole world, "
                f"so fsdp_size must be {plan.world_size}."
            )
        return _FSDP2
    return "replicated" if session is not None else "sequential"


def _load_sharded(route, ckpt_path, auto_class, model_type, kwargs) -> nn.Module:
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
    if not dist.is_initialized():
        raise RuntimeError("The FSDP2 load needs a torch.distributed process group (torchrun).")
    if hf_config is None:
        hf_config = AutoConfig.from_pretrained(ckpt_path, trust_remote_code=trust_remote_code)
    if getattr(hf_config, "quantization_config", None) is not None:
        # It reads the tensors as stored: quantized weights would load as if they were not.
        raise NotImplementedError(
            f"The FSDP2 load cannot dequantize this {model_type} checkpoint; load it unsharded."
        )
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
    distribution: DistributionPlan | str | None = None,
    **kwargs,
) -> nn.Module:
    """Load a Hugging Face model for ModelOpt to optimize and export, whole or distributed.

    ``distribution`` decides how the model is placed:

    * ``None`` (default): the active :class:`~modelopt.torch.utils.distributed.ModelOptSession`
      decides. Without one, or in a single-process one, the model loads whole; in a distributed one,
      as :func:`resolve_distribution_plan` decides.
    * ``"sequential"``: ``auto_class.from_pretrained(ckpt_path, **kwargs)`` (``auto_class``
      defaults to ``AutoModelForCausalLM``), placed as ``kwargs`` say. Each rank that calls it gets
      the whole model.
    * ``"fsdp2"``: a causal LM FSDP2-sharded over the whole process group, each rank reading its
      own layers (:func:`parallel_load_and_prepare_fsdp2`, whose ``cpu_offload``, ``mp_policy`` and
      ``broadcast_chunk_size`` ``kwargs`` may set).
    * A :class:`~modelopt.torch.utils.distributed.DistributionPlan`, as
      :func:`resolve_distribution_plan` returns it:

      - ``fsdp_size > 1``: as ``"fsdp2"``;
      - otherwise as ``"sequential"`` on every rank, ``device_map`` defaulting to the rank's GPU.

      A plan with ``ep_size > 1`` raises: no load here shards the routed experts. The plan is kept
      as ``session.plan``, the same on every rank.

    A sharded load builds the model class and places it itself. Besides its own options it takes
    ``trust_remote_code``, ``attn_implementation`` and ``hf_config`` (the config to load with), and
    refuses any other keyword rather than ignore it.

    Whatever the placement:

    * A Hub model ID is made local first (:func:`ensure_local_checkpoint`): the export reads the
      source checkpoint from disk. Under ``torch.distributed`` every rank must call this.
    * The model type's loading hook runs
      (:func:`~modelopt.torch.models.hf.prepare_model_for_loading`), e.g. to keep the MTP head a
      Nemotron-H checkpoint carries. ``model_type`` selects it, read from the checkpoint's config
      when omitted. A sharded load cannot run it, so it refuses a checkpoint the hook would build
      modules for.
    * The checkpoint tensors the model has no parameter for are recorded
      (:func:`record_unplaced_source_keys`), so the export carries them over unchanged.
    """
    # Importing modelopt.torch.models registers every model spec; only callers of this need them.
    from modelopt.torch.models.hf import prepare_model_for_loading

    _, ckpt_path = ensure_local_checkpoint(ckpt_path)
    trust_remote_code = kwargs.get("trust_remote_code", False)
    if distribution is None:
        plan, hf_config = _decide_distribution(ckpt_path, trust_remote_code, None, None)
        distribution = "sequential" if plan is None else plan
        if plan is not None and plan.is_fsdp_dense:
            kwargs.setdefault("hf_config", hf_config)
    route = _route(distribution)
    session = _distributed_session()
    if session is not None and isinstance(distribution, DistributionPlan):
        session.plan = distribution  # What the model is loaded with, the same on every rank.
    if model_type is None:
        model_type = (
            kwargs.get("hf_config")
            or AutoConfig.from_pretrained(ckpt_path, trust_remote_code=trust_remote_code)
        ).model_type
    if route in _SHARDED_LOAD_OPTIONS:
        model = _load_sharded(route, ckpt_path, auto_class, model_type, kwargs)
    else:
        if "hf_config" in kwargs:
            raise TypeError("hf_config applies to a sharded load only.")
        if route == "replicated":
            # Each rank takes the whole model onto its own GPU. Left to "auto", every rank's
            # accelerate could pick the same idle GPU.
            kwargs.setdefault("device_map", {"": str(session.device)})
        auto_class = auto_class or AutoModelForCausalLM
        with prepare_model_for_loading(
            model_type,
            ckpt_path,
            trust_remote_code,
            model_class=None if auto_class in (AutoModel, AutoModelForCausalLM) else auto_class,
        ):
            model = auto_class.from_pretrained(ckpt_path, **kwargs)
        record_unplaced_source_keys(model, ckpt_path)
    unplaced = model._modelopt_unplaced_source_keys
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
