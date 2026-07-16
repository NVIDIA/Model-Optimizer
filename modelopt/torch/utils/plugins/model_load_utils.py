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
from collections.abc import Callable
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
except ImportError:  # transformers<5 has no fused-MoE weight-conversion engine
    get_model_conversion_mapping = None

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
    """Return the ``param_name → safetensors_file`` map for a local checkpoint directory.

    Handles both sharded checkpoints (``model.safetensors.index.json``) and
    single-file checkpoints (``model.safetensors``). Raises if neither exists.
    """
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


def _conversion_rules(model: nn.Module) -> dict | None:
    """Rename + fuse rules for ``model``, read from transformers' own conversion table.

    Returns ``{"renames": {pattern: repl}, "fuses": [{"src_res", "target", "stack", "cat"}]}``, or
    ``None`` when nothing needs converting (transformers<5 / already-matching checkpoint). A fuse
    stacks its per-expert sources on ``stack`` and, when multi-source, concats them on ``cat``.
    The readable source globs are logged at DEBUG (the dict keeps only compiled ``src_res``).
    """
    renames = dict(getattr(model, "_checkpoint_conversion_mapping", None) or {})
    fuses = []
    readable = []
    for rule in get_model_conversion_mapping(model) if get_model_conversion_mapping else []:
        sources, targets = list(rule.source_patterns), list(rule.target_patterns)
        ops = getattr(rule, "operations", None) or []  # rename-only rules carry none
        if not ops:
            renames.update(dict.fromkeys(sources, targets[0]))
            continue
        dims = {type(op).__name__: op.dim for op in ops}
        if len(targets) != 1 or set(dims) - {"MergeModulelist", "Concatenate"}:
            raise NotImplementedError(
                f"Unsupported conversion rule {sources} -> {targets} ({set(dims)})"
            )
        fuses.append(
            {
                "src_res": [re.compile(s.replace("*", "([^.]+)")) for s in sources],
                "target": targets[0],
                "stack": dims["MergeModulelist"],
                "cat": dims.get("Concatenate"),
            }
        )
        readable.append(f"{sources} -> {targets[0]}")
    rules = {"renames": renames, "fuses": fuses} if (renames or fuses) else None
    if rules:
        logger.debug(
            "checkpoint key conversion: renames=%s fuses=[%s]", renames, "; ".join(readable)
        )
    return rules


def _rename_key(key: str, renames: dict) -> str:
    for old, new in renames.items():
        key = re.sub(old, new, key)
    return key


def _match_fuse(key: str, fuses: list):
    """``(target_name, fuse, source_index, expert_idx)`` if ``key`` is a fuse source, else ``None``."""
    for fuse in fuses:
        for i, rx in enumerate(fuse["src_res"]):
            m = rx.search(key)
            if m:
                return key[: m.start()] + fuse["target"] + key[m.end() :], fuse, i, m.group(1)
    return None


def _target_name(rules: dict, key: str) -> str:
    """Model param name that checkpoint ``key`` feeds (rename + fuse target; no tensor needed)."""
    key = _rename_key(key, rules["renames"])
    hit = _match_fuse(key, rules["fuses"])
    return hit[0] if hit else key


def _convert_keys(rules: dict, state: dict) -> dict:
    """Rename 1:1 keys and fuse per-expert keys into the model's fused params."""
    result, groups = {}, {}  # groups[target] = (fuse, {source_index: {expert_idx: tensor}})
    for key, tensor in state.items():
        key = _rename_key(key, rules["renames"])
        hit = _match_fuse(key, rules["fuses"])
        if hit is None:
            result[key] = tensor
            continue
        target, fuse, i, expert = hit
        groups.setdefault(target, (fuse, {}))[1].setdefault(i, {})[expert] = tensor
    for target, (fuse, by_src) in groups.items():
        stacks = [
            torch.stack([by_src[i][e] for e in sorted(by_src[i], key=int)], fuse["stack"])
            for i in range(len(fuse["src_res"]))
        ]
        result[target] = stacks[0] if fuse["cat"] is None else torch.cat(stacks, fuse["cat"])
    return result


def build_meta_causal_lm(
    ckpt_path: str,
    trust_remote_code: bool,
    attn_implementation: str | None,
    hf_config=None,
):
    """Build a meta-init causal LM (no real storage allocated).

    Pass ``hf_config`` to skip the ``AutoConfig.from_pretrained`` fetch.
    """
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
    """Round-robin: the decoder-layer indices owned by rank ``r``."""
    return [i for i in range(n_layers) if i % world_size == r]


def _read_and_convert(
    resolved_path: str, weight_map: dict, keyset: set[str], rules: dict | None
) -> dict:
    """Read the checkpoint keys in ``keyset`` and apply the fuse/rename rules (identity when None)."""
    raw = read_safetensors_subset(resolved_path, weight_map, lambda k: k in keyset)
    return _convert_keys(rules, raw) if rules else raw


def _read_owned_layers(
    resolved_path: str,
    weight_map: dict,
    layer_sources: dict,
    owned: list[int],
    rules: dict | None,
) -> dict:
    """Read + convert this rank's owned decoder layers from disk (ranks read in parallel)."""
    return {
        layer_idx: _read_and_convert(
            resolved_path, weight_map, set(layer_sources[layer_idx]), rules
        )
        for layer_idx in owned
    }


def _broadcast_load_group(
    group: list[int],
    src: int,
    rank: int,
    owned: dict,
    decoder_layers: list,
    layer_prefixes: list,
    device: torch.device,
    cpu_offload: bool,
) -> None:
    """Broadcast layers ``group`` from rank ``src`` to all ranks and reshard into their FSDP2 shards.

    The owner assembles the group's full tensors; every rank receives them, reshards its local slice,
    then frees the full copy (capping the transient GPU peak). The owner drops its read copy after.
    """
    big_dict: dict | None = None
    if rank == src:
        big_dict = {}
        for i in group:
            big_dict.update(owned[i])
    full = broadcast_state_dict(big_dict, src=src, device=device)
    for layer_idx in group:
        prefix = layer_prefixes[layer_idx]
        stripped = {k[len(prefix) :]: v for k, v in full.items() if k.startswith(prefix)}
        if cpu_offload:
            stripped = {k: v.cpu() for k, v in stripped.items()}
        set_model_state_dict(
            decoder_layers[layer_idx],
            stripped,
            options=StateDictOptions(full_state_dict=True, broadcast_from_rank0=False),
        )
        del stripped
    del full
    if rank == src:
        for i in group:
            del owned[i]


def _group_sources_by_layer(
    weight_map: dict, rules: dict | None, model_param_names: set[str], layer_prefixes: list[str]
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
        target = _target_name(rules, ckpt_key) if rules else ckpt_key
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
    weight_map = weight_map_for(resolved_path)

    model = build_meta_causal_lm(resolved_path, trust_remote_code, attn_implementation, hf_config)

    # shard_root=True shards the root params (embed/lm_head/norm) too, rather than replicating them.
    decoder_layers = fsdp2_wrap(
        model, shard_root=True, mp_policy=mp_policy, cpu_offload=cpu_offload
    )
    module_to_name = {m: n for n, m in model.named_modules()}
    layer_prefixes = [module_to_name[layer] + "." for layer in decoder_layers]

    # transformers>=5 fuses/renames checkpoint keys so they no longer match param names 1:1
    # (None => the pre-5.x identity path).
    rules = _conversion_rules(model)

    # Valid targets; keys converting to anything else are aux weights (e.g. an MTP head) we skip.
    model_param_names = {n for n, _ in chain(model.named_parameters(), model.named_buffers())}

    # Bucket each checkpoint key by its target's decoder layer (root params go to non_layer_sources).
    layer_sources, non_layer_sources, skipped = _group_sources_by_layer(
        weight_map, rules, model_param_names, layer_prefixes
    )
    if skipped:
        logger.debug(
            "skipping %d checkpoint keys not present in the model (e.g. MTP head)", skipped
        )

    _materialize_meta_model(model, torch.device("cpu") if cpu_offload else device)

    owned_indices = _layers_for_rank(len(decoder_layers), world_size, rank)
    owned = _read_owned_layers(resolved_path, weight_map, layer_sources, owned_indices, rules)

    # Smaller broadcast_chunk_size lowers the transient GPU peak (more, smaller collectives).
    for src in range(world_size):
        src_layers = _layers_for_rank(len(decoder_layers), world_size, src)
        if not src_layers:
            continue
        chunk = broadcast_chunk_size or len(src_layers)
        for start in range(0, len(src_layers), chunk):
            _broadcast_load_group(
                src_layers[start : start + chunk],
                src,
                rank,
                owned,
                decoder_layers,
                layer_prefixes,
                device,
                cpu_offload,
            )

    # Non-decoder params: rank 0 reads + broadcasts; resharded into the root below.
    # TODO: layerwise support.
    non_layer = None
    if rank == 0:
        non_layer = _read_and_convert(resolved_path, weight_map, set(non_layer_sources), rules)
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
    model.requires_grad_(False)
    return model
