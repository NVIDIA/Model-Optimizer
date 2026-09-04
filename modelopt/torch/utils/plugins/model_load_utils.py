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

"""HuggingFace-coupled FSDP2 model loading helpers."""

import json
import logging
import os
import re
import time
from collections.abc import Callable
from contextlib import contextmanager
from itertools import chain
from typing import Any

import torch
import torch.nn as nn
from accelerate import init_empty_weights
from huggingface_hub import snapshot_download
from safetensors import safe_open
from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict
from torch.distributed.tensor import DTensor
from transformers import AutoConfig, AutoModelForCausalLM

try:
    from transformers.conversion_mapping import get_model_conversion_mapping
    from transformers.core_model_loading import WeightConverter, dot_natural_key, rename_source_key
except ImportError:  # transformers<5 has no weight-conversion engine
    get_model_conversion_mapping = rename_source_key = WeightConverter = dot_natural_key = None

from modelopt.torch.utils.distributed import (
    barrier,
    broadcast_state_dict,
    fsdp2_wrap,
    is_initialized,
)

logger = logging.getLogger(__name__)


def read_safetensors_subset(
    ckpt_path: str,
    weight_map: dict,
    select: Callable[[str], bool],
) -> dict:
    """Read tensors whose name satisfies ``select`` from safetensors files.

    Groups param names by file to avoid re-opening. Returns CPU tensors.
    Uses ``safe_open`` so only the requested tensors' bytes are read.

    ``get_tensor`` returns a zero-copy view into the mmap'd file; the bytes are
    not actually read from disk until first touched. We ``clone()`` here to force
    the read eagerly, while this function runs (each rank reading its own layers
    in parallel). Without it the read is deferred to the later per-source
    broadcast (``.to(device)``), which is serialized across ranks and silently
    destroys the read parallelism this loader exists to provide.
    """
    by_file: dict[str, list[str]] = {}
    for name, file in weight_map.items():
        if select(name):
            by_file.setdefault(file, []).append(name)

    state: dict[str, torch.Tensor] = {}
    for file, names in by_file.items():
        with safe_open(os.path.join(ckpt_path, file), framework="pt", device="cpu") as f:
            for name in names:
                state[name] = f.get_tensor(name).clone()
    return state


def weight_map_for(ckpt_path: str) -> dict[str, str]:
    """Return the ``param_name → safetensors_file`` map for a local checkpoint directory."""
    index_path = os.path.join(ckpt_path, "model.safetensors.index.json")
    single_file = os.path.join(ckpt_path, "model.safetensors")
    if os.path.exists(index_path):
        with open(index_path) as f:
            return json.load(f)["weight_map"]
    if os.path.exists(single_file):
        with safe_open(single_file, framework="pt", device="cpu") as f:
            return dict.fromkeys(f.keys(), "model.safetensors")
    raise RuntimeError(
        f"No safetensors checkpoint at {ckpt_path} "
        "(expected model.safetensors or model.safetensors.index.json)."
    )


def _resolve_checkpoint_dir(ckpt_path: str, rank: int) -> str:
    """Local dir for ``ckpt_path``; resolves an HF Hub ID (rank 0 downloads, others wait)."""
    if os.path.isdir(ckpt_path):
        return ckpt_path
    if rank == 0:
        snapshot_download(ckpt_path)
    if is_initialized():
        barrier()
    return snapshot_download(ckpt_path)


# Per-phase load timing: name -> [call count, seconds]. Populated by :func:`_load_phase`.
_LOAD_TIMERS: dict[str, list] = {}


@contextmanager
def _load_phase(name: str):
    """Add this block's wall time to the named load phase."""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        entry = _LOAD_TIMERS.setdefault(name, [0, 0.0])
        entry[0] += 1
        entry[1] += time.perf_counter() - t0


def _report_load_timers(total_s: float) -> None:
    """Print the per-phase load breakdown; the caller decides which rank reports."""
    print(f"\n===== fsdp2 load breakdown ({total_s:.1f}s total) =====", flush=True)
    for name, (n, secs) in sorted(_LOAD_TIMERS.items(), key=lambda kv: -kv[1][1]):
        share = f"{100.0 * secs / total_s:5.1f}%" if total_s else "    -"
        print(f"  {name:<34} {n:>6} calls {secs:>8.1f}s | {share}", flush=True)
    accounted = sum(v[1] for v in _LOAD_TIMERS.values())
    print(f"  {'UNACCOUNTED':<34} {'':>6}       {total_s - accounted:>8.1f}s\n", flush=True)


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


def _conversion_plan(model: nn.Module) -> dict | None:
    """Transformers' own conversion mapping for ``model``, or ``None`` if nothing needs converting.

    ``legacy_renames`` (``_checkpoint_conversion_mapping``) covers transformers<5; on 5+ the
    ``renamings``/``converters`` from HF's engine drive renaming + MoE weight fusion directly.
    """
    legacy_renames = dict(getattr(model, "_checkpoint_conversion_mapping", None) or {})
    renamings, converters = [], []
    for entry in get_model_conversion_mapping(model) if get_model_conversion_mapping else []:
        (converters if isinstance(entry, WeightConverter) else renamings).append(entry)
    if not (legacy_renames or renamings or converters):
        return None
    return {
        "legacy_renames": legacy_renames,
        "renamings": renamings,
        "converters": converters,
        "prefix": model.base_model_prefix,
        "meta_state_dict": model.state_dict(),
    }


def _resolve_target(plan: dict, key: str) -> tuple[str, str | None]:
    """Resolve a checkpoint key to ``(target param name, matched converter source pattern)``.

    No tensors are read. ``source_pattern`` is ``None`` for a plain (non-fused) key.
    """
    for old, new in plan["legacy_renames"].items():
        key = re.sub(old, new, key)
    if rename_source_key is None:  # transformers<5: legacy renames only, no converters
        return key, None
    return rename_source_key(
        key, plan["renamings"], plan["converters"], plan["prefix"], plan["meta_state_dict"]
    )


def _convert_keys(plan: dict, state: dict) -> dict:
    """Rename 1:1 keys and fuse per-expert keys by driving transformers' own conversion ops."""
    if rename_source_key is None:  # transformers<5: legacy renames only, no fusion
        return {_resolve_target(plan, k)[0]: v for k, v in state.items()}
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
    dtype = getattr(hf_config, "torch_dtype", None) or torch.bfloat16
    with init_empty_weights(include_buffers=False):
        model = AutoModelForCausalLM.from_config(
            hf_config, torch_dtype=dtype, trust_remote_code=trust_remote_code
        )
    model.eval()
    return model


def _layers_for_rank(n_layers: int, world_size: int, r: int) -> list[int]:
    return [i for i in range(n_layers) if i % world_size == r]


def _read_and_convert(
    resolved_path: str, weight_map: dict, keyset: set[str], plan: dict | None
) -> dict:
    with _load_phase("disk read (safetensors)"):
        raw = read_safetensors_subset(resolved_path, weight_map, lambda k: k in keyset)
    if not plan:
        return raw
    with _load_phase("key conversion (HF engine)"):
        return _convert_keys(plan, raw)


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
    with _load_phase("broadcast (network)"):
        broadcasted_state_dict = broadcast_state_dict(
            group_state_dict, src=source_rank, device=device
        )
    for layer_idx in layer_indices:
        prefix = layer_prefixes[layer_idx]
        layer_state_dict = {
            k[len(prefix) :]: v for k, v in broadcasted_state_dict.items() if k.startswith(prefix)
        }
        if cpu_offload:
            with _load_phase("cpu offload copy"):
                layer_state_dict = {k: v.cpu() for k, v in layer_state_dict.items()}
        with _load_phase("reshard (set_model_state_dict)"):
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
    weight_map: dict, plan: dict | None, model_param_names: set[str], layer_prefixes: list[str]
) -> tuple[dict[int, list[str]], list[str], int]:
    """Bucket checkpoint keys by the decoder layer their converted target lives in.

    Returns ``(layer_sources, non_layer_sources, skipped)``: ``layer_sources[i]`` holds the keys
    targeting decoder layer ``i``, ``non_layer_sources`` holds root (embed/lm_head/norm) keys, and
    ``skipped`` counts keys whose target isn't in the model (aux weights, e.g. an MTP head).
    """
    layer_sources: dict[int, list[str]] = {i: [] for i in range(len(layer_prefixes))}
    non_layer_sources: list[str] = []
    skipped = 0
    for ckpt_key in weight_map:
        target = _resolve_target(plan, ckpt_key)[0] if plan else ckpt_key
        if target not in model_param_names:
            skipped += 1
            continue
        for i, prefix in enumerate(layer_prefixes):
            if target.startswith(prefix):
                layer_sources[i].append(ckpt_key)
                break
        else:
            non_layer_sources.append(ckpt_key)
    return layer_sources, non_layer_sources, skipped


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
    _load_t0 = time.perf_counter()
    with _load_phase("weight_map (index json)"):
        weight_map = weight_map_for(resolved_path)

    with _load_phase("skeleton build (meta)"):
        model = build_meta_causal_lm(
            resolved_path, trust_remote_code, attn_implementation, hf_config
        )

    # fsdp2_wrap shards each decoder layer + the root (embed/lm_head/norm sharded, not replicated).
    with _load_phase("fsdp2_wrap"):
        decoder_layers = fsdp2_wrap(model, mp_policy=mp_policy, cpu_offload=cpu_offload)
    module_to_name = {m: n for n, m in model.named_modules()}
    layer_prefixes = [module_to_name[layer] + "." for layer in decoder_layers]

    # transformers>=5 fuses/renames checkpoint keys so they no longer match param names 1:1
    # (None => the pre-5.x identity path).
    plan = _conversion_plan(model)

    # Valid targets; keys converting to anything else are aux weights (e.g. an MTP head) we skip.
    model_param_names = {n for n, _ in chain(model.named_parameters(), model.named_buffers())}

    # Bucket each checkpoint key by its target's decoder layer (root params go to non_layer_sources).
    layer_sources, non_layer_sources, skipped = _group_sources_by_layer(
        weight_map, plan, model_param_names, layer_prefixes
    )
    if skipped:
        logger.debug(
            "skipping %d checkpoint keys not present in the model (e.g. MTP head)", skipped
        )

    with _load_phase("materialize meta params"):
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
    if rank == 0:
        _report_load_timers(time.perf_counter() - _load_t0)
    return model
