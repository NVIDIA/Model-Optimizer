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
import os
from collections.abc import Callable
from typing import Any
from warnings import warn

import torch
import torch.nn as nn
from accelerate import init_empty_weights
from huggingface_hub import snapshot_download
from safetensors import safe_open
from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict
from torch.distributed.tensor import DTensor
from transformers import AutoConfig, AutoModelForCausalLM

from modelopt.torch.utils.distributed import (
    barrier,
    broadcast_state_dict,
    fsdp2_wrap,
    is_initialized,
)


def read_safetensors_subset(
    ckpt_path: str,
    weight_map: dict,
    select: Callable[[str], bool],
) -> dict:
    """Read tensors whose name satisfies ``select`` from safetensors files.

    Groups param names by file to avoid re-opening. Returns CPU tensors.
    Uses ``safe_open`` so only the requested tensors' bytes are read.
    """
    by_file: dict[str, list[str]] = {}
    for name, file in weight_map.items():
        if select(name):
            by_file.setdefault(file, []).append(name)

    state: dict[str, torch.Tensor] = {}
    for file, names in by_file.items():
        with safe_open(os.path.join(ckpt_path, file), framework="pt", device="cpu") as f:
            for name in names:
                state[name] = f.get_tensor(name)
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


def parallel_load_and_prepare_fsdp2(
    ckpt_path: str,
    device: torch.device,
    rank: int,
    world_size: int,
    trust_remote_code: bool = False,
    mp_policy=None,
    cpu_offload: bool = False,
    attn_implementation: str | None = None,
    freeze: bool = True,
    hf_config=None,
) -> nn.Module:
    """Load and FSDP2-shard a HuggingFace causal LM via parallel safetensors reads.

    Round-robin assigns decoder layers to ranks; each rank reads only its owned
    layers' weights from disk in parallel, then broadcasts to the others. Non-decoder
    weights (embed, lm_head, norm) are read on rank 0 and broadcast.

    Requires an initialized ``torch.distributed`` process group (FSDP2's ``fully_shard``
    and the per-layer broadcasts both need it). A 1-rank PG (e.g. ``torchrun
    --nproc_per_node=1``) is allowed; bare single-process is not.

    Set ``freeze=False`` for training callers; PTQ keeps the default ``True``.
    Pass ``hf_config`` if the caller has already fetched it (skips a redundant fetch).
    """
    # Resolve HF Hub IDs to a local cache dir (rank 0 downloads; others wait).
    if os.path.isdir(ckpt_path):
        resolved_path = ckpt_path
    else:
        if rank == 0:
            snapshot_download(ckpt_path)
        if is_initialized():
            barrier()
        resolved_path = snapshot_download(ckpt_path)
    weight_map = weight_map_for(resolved_path)

    # Meta skeleton on every rank.
    model = build_meta_causal_lm(resolved_path, trust_remote_code, attn_implementation, hf_config)

    # Shard decoder layers (root stays unwrapped); reuse the returned detection result.
    decoder_layers = fsdp2_wrap(model, mp_policy=mp_policy, cpu_offload=cpu_offload)
    module_to_name = {m: n for n, m in model.named_modules()}
    layer_prefixes = [module_to_name[layer] + "." for layer in decoder_layers]

    # Materialize meta → empty real tensors (CPU when cpu_offload, GPU otherwise).
    _materialize_meta_model(model, torch.device("cpu") if cpu_offload else device)

    # Round-robin ownership: each rank reads only its owned layers from disk in parallel.
    owned: dict[int, dict] = {}
    for layer_idx in range(len(decoder_layers)):
        if layer_idx % world_size == rank:
            prefix = layer_prefixes[layer_idx]

            def _has_prefix(n: str) -> bool:
                return n.startswith(prefix)

            owned[layer_idx] = read_safetensors_subset(resolved_path, weight_map, _has_prefix)

    # Per-layer: owner broadcasts → every rank shards the full tensor into its DTensor.
    for layer_idx in range(len(decoder_layers)):
        src = layer_idx % world_size
        full = broadcast_state_dict(owned.get(layer_idx), src=src, device=device)
        prefix = layer_prefixes[layer_idx]
        stripped = {k[len(prefix) :]: v for k, v in full.items()}
        if cpu_offload:
            stripped = {k: v.cpu() for k, v in stripped.items()}
        set_model_state_dict(
            decoder_layers[layer_idx],
            stripped,
            options=StateDictOptions(full_state_dict=True, broadcast_from_rank0=False),
        )
        if src == rank:
            del owned[layer_idx]

    # Non-decoder params (embed, lm_head, norm) — rank 0 reads + broadcasts.
    # TODO: add support for shard_root=True and layerwise.
    layer_prefix_tuple = tuple(layer_prefixes)
    non_layer = (
        read_safetensors_subset(
            resolved_path, weight_map, lambda n: not n.startswith(layer_prefix_tuple)
        )
        if rank == 0
        else None
    )
    non_layer = broadcast_state_dict(non_layer, src=0, device=device)
    if cpu_offload:
        non_layer = {k: v.cpu() for k, v in non_layer.items()}
    # strict=False: non_layer is a subset of the full model — decoder keys will
    # show up as "missing" but that's expected. We filter and warn below.
    missing, unexpected = model.load_state_dict(non_layer, strict=False, assign=False)
    real_missing = [k for k in missing if not k.startswith(layer_prefix_tuple)]
    if real_missing:
        warn(f"Missing non-layer keys on rank {rank}: {real_missing[:5]}...")
    if unexpected:
        warn(f"Unexpected keys in non-layer state dict on rank {rank}: {unexpected[:3]}...")

    if cpu_offload:
        # All tensors were materialized on CPU only to satisfy set_model_state_dict's
        # uniform-device requirement. FSDP2 only manages the wrapped decoder layers
        # (streamed CPU↔GPU per forward); the unwrapped root (embed/lm_head/norm +
        # buffers) is ours to place, and we retain it on GPU.
        _promote_non_dtensor_to_gpu(model, device)
    if hasattr(model, "tie_weights"):
        model.tie_weights()
    if freeze:
        model.requires_grad_(False)
    return model
