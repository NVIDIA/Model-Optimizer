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

"""Utility functions for using torch.distributed."""

import functools
import io
import os
import re
import sys
import time
import traceback
from collections.abc import Callable, Iterator
from contextlib import suppress
from datetime import timedelta
from typing import Any
from warnings import warn

import torch
import torch.distributed
from torch.distributed.fsdp import CPUOffloadPolicy, FSDPModule, fully_shard
from torch.distributed.tensor import DTensor

__all__ = [
    "DistributedProcessGroup",
    "ParallelState",
    "backend",
    "barrier",
    "fsdp2_wrap",
    "gather_bytes",
    "gather_state_dicts",
    "is_available",
    "is_fsdp2_model",
    "is_initialized",
    "is_master",
    "rank",
    "size",
]


def is_available() -> bool:
    """Returns whether the distributed package is available."""
    return torch.distributed.is_available()


def is_initialized() -> bool:
    """Returns whether the distributed package is initialized."""
    return is_available() and torch.distributed.is_initialized()


def backend() -> str | None:
    """Returns the distributed backend."""
    if is_initialized():
        return "torch"
    return None


def size(group=None) -> int:
    """Returns the number of processes."""
    if backend() == "torch":
        return torch.distributed.get_world_size(group=group)
    return 1


def rank(group=None) -> int:
    """Returns the rank of the current process."""
    if backend() == "torch":
        return torch.distributed.get_rank(group=group)
    return 0


def local_rank() -> int:
    """Returns the local rank of the current process."""
    if "LOCAL_RANK" in os.environ:
        return int(os.environ["LOCAL_RANK"])
    warn("LOCAL_RANK environment variable not found. Using global rank instead.")
    return rank()


def is_master(group=None) -> bool:
    """Returns whether the current process is the master process."""
    return rank(group=group) == 0


def is_last_process(group=None) -> bool:
    """Returns whether the current process is the last process."""
    return rank(group=group) == size(group=group) - 1


def _serialize(obj: Any) -> torch.Tensor:
    buffer = io.BytesIO()
    torch.save(obj, buffer)
    storage = torch.UntypedStorage.from_buffer(buffer.getvalue(), dtype=torch.uint8)
    tensor = torch.ByteTensor(storage)
    return tensor


def _deserialize(tensor: torch.Tensor, size: int | None = None) -> Any:
    buffer = tensor.numpy().tobytes()
    if size is not None:
        buffer = buffer[:size]
    # Explicit so TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD cannot re-enable pickle for peer payloads. Custom
    # classes sent through broadcast/allgather must be registered with add_safe_globals.
    return torch.load(io.BytesIO(buffer), weights_only=True)


def _broadcast(tensor: torch.Tensor, src: int = 0, group=None) -> None:
    if backend() == "torch":
        torch.distributed.broadcast(tensor, src, group)


def broadcast(obj: Any, src: int = 0, group=None) -> Any:
    """Broadcasts an object from the source to all other processes."""
    if size() == 1:
        return obj

    # serialize
    if rank() == src:
        tensor = _serialize(obj).cuda()

    # broadcast the tensor size
    tensor_size = (
        torch.LongTensor([tensor.numel()]).cuda() if rank() == src else torch.LongTensor([0]).cuda()
    )
    _broadcast(tensor_size, src=src, group=group)

    # broadcast the tensor
    if rank() != src:
        tensor = torch.ByteTensor(size=(tensor_size.item(),)).cuda()
    _broadcast(tensor, src=src, group=group)

    # deserialize
    if rank() != src:
        obj = _deserialize(tensor.cpu())
    return obj


def gather_bytes(
    data: bytes, dst: int, group=None, chunk_bytes: int = 1 << 28
) -> list[bytearray] | None:
    """Gathers byte strings to global rank ``dst`` only, without unpickling anything on the way.

    Unlike ``torch.distributed.gather_object`` it never runs pickle, and it moves ``chunk_bytes`` per
    rank at a time to bound device memory. Returns the per-rank bytes on ``dst`` and ``None`` elsewhere.
    """
    n = size(group)
    device = torch.device("cuda" if torch.distributed.get_backend(group) == "nccl" else "cpu")
    local = (
        torch.frombuffer(bytearray(data), dtype=torch.uint8)
        if data
        else torch.empty(0, dtype=torch.uint8)
    )
    sizes_t = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(n)]
    torch.distributed.all_gather(sizes_t, torch.tensor([len(data)], device=device), group=group)
    sizes = [int(t.item()) for t in sizes_t]

    is_dst = torch.distributed.get_rank() == dst
    out = [bytearray(s) for s in sizes] if is_dst else None
    for off in range(0, max(sizes), chunk_bytes):
        length = min(chunk_bytes, max(sizes) - off)
        buf = torch.zeros(length, dtype=torch.uint8, device=device)
        piece = local[off : off + length]
        buf[: piece.numel()] = piece.to(device)
        bufs = [torch.empty_like(buf) for _ in range(n)] if is_dst else None
        torch.distributed.gather(buf, bufs, dst=dst, group=group)
        if bufs is not None and out is not None:
            for i, b in enumerate(bufs):
                k = min(length, max(0, sizes[i] - off))
                out[i][off : off + k] = b[:k].cpu().numpy().tobytes()
    return out


def gather_state_dicts(state: dict, dst: int, group=None) -> Iterator[dict]:
    """Gathers ``state`` from every rank of ``group`` to global rank ``dst``, loaded with ``weights_only``.

    Yields one rank's dict at a time on ``dst`` (to bound host memory) and nothing elsewhere. The collective
    runs eagerly, so every rank must call this before iterating.
    """
    buf = io.BytesIO()
    torch.save(state, buf)
    gathered = gather_bytes(buf.getvalue(), dst, group)
    del buf

    def _load() -> Iterator[dict]:
        while gathered:
            # Explicit weights_only so TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD cannot re-enable pickle for peers.
            yield torch.load(io.BytesIO(gathered.pop(0)), map_location="cpu", weights_only=True)

    return _load()


def _allgather(tensors: list[torch.Tensor], tensor: torch.Tensor, group=None) -> None:
    if backend() == "torch":
        torch.distributed.all_gather(tensors, tensor, group)


def allgather(obj: Any, group=None) -> list[Any]:
    """Gathers an object from all processes into a list."""
    if size(group) == 1:
        return [obj]

    # serialize
    tensor = _serialize(obj).cuda()

    # gather the tensor size
    tensor_size = torch.LongTensor([tensor.numel()]).cuda()
    tensor_sizes = [torch.LongTensor([0]).cuda() for _ in range(size(group))]
    _allgather(tensor_sizes, tensor_size, group)
    tensor_sizes = [int(tensor_size.item()) for tensor_size in tensor_sizes]
    max_size = max(tensor_sizes)

    # gather the tensor
    tensors = [torch.ByteTensor(size=(max_size,)).cuda() for _ in tensor_sizes]
    if tensor_size != max_size:
        padding = torch.ByteTensor(size=(max_size - tensor_size,)).cuda()
        tensor = torch.cat((tensor, padding), dim=0)
    _allgather(tensors, tensor, group)

    # deserialize
    objs = []
    for tensor_size, tensor in zip(tensor_sizes, tensors):
        obj = _deserialize(tensor.cpu(), size=tensor_size)
        objs.append(obj)
    return objs


def allreduce(obj: Any, reduction: str = "sum", group=None) -> Any:
    """Reduces an object from all processes."""
    objs = allgather(obj, group)
    if reduction == "sum":
        return sum(objs)
    else:
        raise NotImplementedError(reduction)


def barrier(group=None) -> None:
    """Synchronizes all processes."""
    if size() == 1:
        return
    if backend() == "torch":
        torch.distributed.barrier(group=group)


def master_only(func):
    """Decorator to run a function only on the master process and broadcast the result."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return broadcast(func(*args, **kwargs) if is_master() else None)

    return wrapper


def setup(timeout: timedelta | None = None):
    """Sets up the distributed environment."""
    torch.cuda.set_device(local_rank())
    if not is_initialized():
        torch.distributed.init_process_group("cpu:gloo,cuda:nccl", timeout=timeout)


def _fsdp2_wrap_modules(model: "torch.nn.Module") -> list["torch.nn.Module"]:
    """Return the transformer-block modules to FSDP2 ``fully_shard`` individually.

    Uses HuggingFace's own ``_no_split_modules`` -- the class names of the atomic blocks that must not be
    split across devices (what ``device_map="auto"`` wraps at) -- falling back to modelopt's decoder-layer
    auto-detection.

    For a VLM, this looks at the LANGUAGE backbone's boundaries, NOT the top-level VLM wrapper's
    ``_no_split_modules`` (which for some VLMs -- e.g. Qwen2.5-VL -- ALSO lists the VISION block classes).
    FSDP-sharding the vision blocks makes the calibration weight-access all-gather on those sharded vision
    weights DEADLOCK; keeping the vision/audio towers REPLICATED (plain) -- the language-only calibration
    never runs them and they are small enough to fit -- avoids that. This also covers VLMs whose top-level
    wrapper exposes no ``_no_split_modules`` at all (e.g. Nemotron-Nano-VL).
    """
    from modelopt.torch.quantization.utils.layerwise_calib import LayerActivationCollector

    def _boundaries(root: "torch.nn.Module") -> "list[torch.nn.Module] | None":
        ns = set(getattr(root, "_no_split_modules", None) or [])
        if ns:
            mods = [m for m in root.modules() if type(m).__name__ in ns]
            if mods:
                return mods
        layers = LayerActivationCollector.get_decoder_layers(root)
        return list(layers) if layers is not None else None

    # VLM -> wrap only the language backbone's blocks (vision/audio stay replicated); else the whole model.
    from modelopt.torch.export.model_utils import get_language_model_from_vl

    lineage = get_language_model_from_vl(model)
    if lineage and lineage[-1] is not model:
        lm_mods = _boundaries(lineage[-1])
        if lm_mods:
            return lm_mods

    mods = _boundaries(model)
    if mods:
        return mods
    raise RuntimeError(
        "Could not determine FSDP2 wrap boundaries: model exposes no `_no_split_modules` and modelopt's "
        "decoder-layer auto-detection did not match. Provide a standard HF causal-LM / VLM checkpoint."
    )


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


def _meta_dcp_load(
    model: "torch.nn.Module",
    local_dir: str,
    device,
    *,
    load_names: "set | None" = None,
    skip_names: "set | None" = None,
    max_unloaded_frac: float = 0.5,
) -> None:
    """The single meta + torch-DCP loader shared by every distributed dense path.

    materialize each rank's params on ``device`` and fill them from the HF safetensors via DCP's
    ``HuggingFaceStorageReader``.

    PLACEMENT is decided entirely by the CALLER, not here: ``dcp.load`` reads a FULL tensor into a plain
    (replicated) param and only the LOCAL SLICE into a ``Shard`` ``DTensor``. So the caller picks the layout
    by what it does before calling this -- nothing (-> Replicate, full per rank),
    ``fully_shard`` the whole model (``load_and_prepare_fsdp2`` -> Shard), or ``fully_shard`` only the dense
    modules (the EP hybrid -> dense Shard, experts left as loaded). No full-model copy for the sharded cases.

    ``load_names`` (optional) restricts the checkpoint load to a subset of model param names -- the EP-dense
    case, where the experts are ALREADY materialized by ``from_pretrained`` and must not be touched. ``None``
    loads the whole model (DP / FSDP2-full).

    Materialization is SELECTIVE (module-by-module, only where params are meta) so already-real params (EP
    experts) are never re-emptied. ``to_empty`` also re-empties BUFFERS (rotary ``inv_freq`` etc., + any the
    caller already loaded), so they are snapshotted and restored -- a params-only checkpoint load would
    otherwise leave them garbage and the calibration forward would drift (weights load fine but activation
    amax is wrong). Persistent buffers the checkpoint DOES carry are (re)loaded via DCP below.
    """
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint import HuggingFaceStorageReader

    # Snapshot real buffers (to_empty re-empties them), materialize params, restore buffers. Whole-model
    # meta (DP / FSDP2-full, ``load_names is None``) uses ``model.to_empty`` -- the FSDP2-aware root
    # materialization; a per-module selective loop corrupts the ROOT-sharded FSDP state (its state_dict then
    # returns wrong keys). The PARTIAL case (EP-dense: dense meta, experts already real) MUST be selective so
    # the real experts are not re-emptied -- there no root is fully_shard'd, so per-module to_empty is safe.
    saved_buffers = {n: b.detach().clone() for n, b in model.named_buffers() if not b.is_meta}
    # A module's ``to_empty`` re-empties every direct param, so a param already real beside a meta one
    # (a router's ``e_score_correction_bias`` next to its meta ``weight``) is kept here too: nothing
    # outside ``load_names`` would refill it.
    saved_params: dict[str, torch.Tensor] = {}
    if load_names is None:
        model.to_empty(device=device)
    else:
        for name, m in model.named_modules():
            direct = dict(m.named_parameters(recurse=False))
            if any(p.is_meta for p in direct.values()):
                prefix = f"{name}." if name else ""
                saved_params.update(
                    (prefix + n, p.detach().clone()) for n, p in direct.items() if not p.is_meta
                )
                m.to_empty(device=device, recurse=False)
    for n, b in model.named_buffers():
        src = saved_buffers.get(n)
        if src is not None:
            b.copy_(src.to(device=b.device, dtype=b.dtype))
    with torch.no_grad():
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
    target_names = set(full_sd) if load_names is None else (set(load_names) & set(full_sd))
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
    # Fail LOUD if the mapping left too much of the TARGET WEIGHT unloaded -- a key scheme this path did not
    # resolve; a silent partial load produces a garbage-weight model. Weight by NUMEL, not param COUNT: a
    # fused-expert MoE (Qwen3-MoE: per-expert checkpoint -> grouped ``experts.gate_up_proj``/``down_proj``
    # model params DCP can't fuse) leaves only ~2 params/layer unloaded but they are ~95% of the weight, so a
    # count-based guard misses it. A small tail is normal (tied weights + a VLM's not-in-checkpoint embeds).
    # ``skip_names`` = params loaded by a SEPARATE path (e.g. Qwen3-MoE fused grouped experts, filled by
    # :func:`_load_fused_experts_sharded` which fuses per-expert checkpoint tensors DCP can't concat) --
    # exclude them from the coverage guard so it doesn't (correctly) flag them as unloaded here.
    skip_names = skip_names or set()
    numel = {
        n: p.numel()
        for n, p in model.named_parameters()
        if n in target_names and n not in skip_names
    }
    want_params = set(numel)
    unloaded = want_params - mapped_names
    total_nel = max(1, sum(numel.values()))
    unloaded_nel = sum(numel[n] for n in unloaded)
    if unloaded_nel > max_unloaded_frac * total_nel:
        raise RuntimeError(
            f"meta+DCP load: {len(unloaded)}/{len(want_params)} params ({100 * unloaded_nel / total_nel:.0f}% "
            f"of the target weight) matched no checkpoint key -- the checkpoint's key naming was not resolved "
            f"to this model's state_dict (exact + conversion-mapping + suffix match all missed), or it needs a "
            f"weight conversion DCP can't do (FUSED EXPERTS, split qkv, on-the-fly dequant). Unloaded e.g.: "
            f"{sorted(unloaded)[:5]}. Use the from_pretrained load (MODELOPT_FSDP2_SHARDED_LOAD=0 / "
            f"MODELOPT_EP_DENSE_DISTRIBUTED_LOAD=0)."
        )
    if is_initialized() and rank() == 0:
        loaded_bufs = mapped_names - {n for n, _ in model.named_parameters()}
        if unloaded:
            warn(
                f"meta+DCP load: {len(unloaded)} param(s) not loaded from checkpoint (tied? re-tied): "
                f"{sorted(unloaded)[:5]}"
            )
        if loaded_bufs:
            warn(
                f"meta+DCP load: loaded {len(loaded_bufs)} persistent buffer(s) from checkpoint: "
                f"{sorted(loaded_bufs)[:5]}"
            )
    dcp.load(load_sd, storage_reader=reader)
    if hasattr(model, "tie_weights"):
        model.tie_weights()  # re-establish tied weights the checkpoint omitted


def cleanup():
    """Cleans up the distributed environment.

    The barrier is skipped when unwinding from an error, since peers may be blocked in a collective
    this rank will never reach. ``SystemExit`` is treated as a clean exit (every rank reaches it).
    That is not sufficient on its own -- ``destroy_process_group`` below blocks for the same reason
    -- so error paths must call :func:`abort` before reaching this ``finally``.
    """
    if is_initialized():
        exc = sys.exc_info()[1]
        if exc is None or isinstance(exc, SystemExit):
            with suppress(Exception):
                barrier()
        torch.distributed.destroy_process_group()


def abort(exit_code: int = 1) -> None:
    """Print the active exception and exit this rank immediately.

    Call from an ``except`` block in a distributed entrypoint. Both a barrier and
    ``destroy_process_group`` stall when peers are blocked in a collective this rank will never
    reach, and the traceback only prints once the enclosing ``finally`` returns -- so the run looks
    hung rather than failed. Exiting lets the launcher (e.g. torchrun) terminate the peers.
    ``SystemExit`` is re-raised instead, since every rank reaches an intentional exit.
    """
    exc = sys.exc_info()[1]
    if isinstance(exc, SystemExit):
        raise exc
    traceback.print_exc()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)


def is_fsdp2_model(model) -> bool:
    """Return True if any submodule of ``model`` has been wrapped with FSDP2 ``fully_shard``."""
    return any(isinstance(m, FSDPModule) for m in model.modules())


def _off_dtype_params(model) -> set[torch.nn.Parameter]:
    """Params whose dtype differs from the model's dominant (by element count) param dtype.

    FSDP2 needs one dtype per shard group, but HF models routinely keep a few params in fp32 for
    stability (e.g. MoE router gates). Pass these to ``fully_shard(ignored_params=...)``.

    TODO: Drop this and shard the off-dtype params once a stable PyTorch release includes FSDP2
    mixed-precision parameter dtype support (already on nightly).
    """
    numel_by_dtype: dict[torch.dtype, int] = {}
    for param in model.parameters():
        numel_by_dtype[param.dtype] = numel_by_dtype.get(param.dtype, 0) + param.numel()
    if len(numel_by_dtype) <= 1:
        return set()

    # Lazy import: logging imports this module at top level (circular).
    from modelopt.torch.utils.logging import warn_rank_0

    dominant = max(numel_by_dtype, key=lambda d: numel_by_dtype[d])
    off_dtype = {n: p for n, p in model.named_parameters() if p.dtype != dominant}
    off_numel = sum(numel_by_dtype[d] for d in numel_by_dtype if d != dominant)
    names = sorted(off_dtype)
    warn_rank_0(
        f"Model has mixed parameter dtypes {set(numel_by_dtype)}; FSDP2 needs one dtype per shard "
        f"group, so {len(names)} non-{dominant} parameter(s) "
        f"({100 * off_numel / sum(numel_by_dtype.values()):.2f}% of elements) will stay replicated "
        f"rather than sharded: {names[:3]}{' ...' if len(names) > 3 else ''}"
    )
    return set(off_dtype.values())


def _move_to_fsdp_device(model, params: set[torch.nn.Parameter]) -> None:
    """Move ``params`` onto the device FSDP2 computes on for ``model``.

    ``fully_shard`` only moves the params it manages, so ignored ones would be stranded on
    whatever device the caller built the model on. Meta params are left alone for deferred init.

    The device comes from a sharded param's mesh rather than its local shard: under
    ``cpu_offload`` the shard rests on CPU while compute still happens on the accelerator.
    """
    # Lazy import: logging imports this module at top level (circular).
    from modelopt.torch.utils.logging import warn_rank_0

    mesh = next((p.device_mesh for p in model.parameters() if isinstance(p, DTensor)), None)
    if mesh is None:
        warn_rank_0(
            f"FSDP2 sharded no parameter of {type(model).__name__}, so the compute device for "
            f"{len(params)} unsharded off-dtype parameter(s) cannot be determined; leaving them "
            "where they are. Move them to the compute device or the forward will fail."
        )
        return

    device = (
        torch.device("cpu")
        if mesh.device_type == "cpu"
        else torch.device(mesh.device_type, getattr(torch, mesh.device_type).current_device())
    )
    for param in params:
        if not param.is_meta and param.device != device:
            param.data = param.data.to(device)


def fsdp2_wrap(model, shard_root=True, mp_policy=None, cpu_offload: bool = False):
    """Auto-detect a HF causal-LM's decoder layers and FSDP2 ``fully_shard`` each one.

    By default (``shard_root=True``) the root module is wrapped too, so embed/lm_head/norm are
    sharded instead of replicated per rank; pass ``shard_root=False`` to leave the root replicated
    (only decoder layers sharded). Returns the detected decoder layers so callers can reuse the
    detection result.

    Parameters whose dtype differs from the model's dominant one are excluded from the wrap (see
    :func:`_off_dtype_params`), since FSDP2 rejects a shard group that mixes dtypes. They stay
    replicated and are moved onto the shards' device, which ``fully_shard`` does not do for the
    params it ignores.
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


def broadcast_state_dict(
    state_dict_or_none: dict | None,
    src: int,
    device: torch.device,
    pg=None,
) -> dict:
    """Broadcast a dict of CPU tensors from rank ``src`` to all ranks.

    Two phases: (1) broadcast metadata (key list + shape/dtype) via
    ``broadcast_object_list``, (2) broadcast each tensor via ``dist.broadcast``.
    Source rank passes the populated dict; non-source ranks pass ``None``.
    Returns a dict of tensors on ``device`` on every rank.
    """
    is_src = torch.distributed.get_rank() == src
    meta: list[Any] = (
        [{name: (tuple(t.shape), t.dtype) for name, t in state_dict_or_none.items()}]
        if is_src and state_dict_or_none is not None
        else [None]
    )
    torch.distributed.broadcast_object_list(meta, src=src, group=pg)
    meta_dict = meta[0]
    assert meta_dict is not None, f"src rank {src} passed no state dict to broadcast"

    src_state_dict = state_dict_or_none or {}
    out: dict[str, torch.Tensor] = {}
    for name, (shape, dtype) in meta_dict.items():
        if is_src:
            t = src_state_dict[name].to(device)
        else:
            t = torch.empty(shape, dtype=dtype, device=device)
        torch.distributed.broadcast(t, src=src, group=pg)
        out[name] = t
    return out


class DistributedProcessGroup:
    """A convenient wrapper around torch.distributed.ProcessGroup objects."""

    def __init__(self, group: torch.distributed.ProcessGroup | int | None = None):
        """Initialize the distributed process group."""
        self.group = group

    def is_initialized(self) -> bool:
        """Check if the distributed process group is initialized."""
        return backend() == "torch" and self.group != -1

    def rank(self) -> int:
        """Get the rank of the current process group."""
        return rank(group=self.group) if self.is_initialized() else -1

    def world_size(self) -> int:
        """Get the world size of the current process group."""
        return size(group=self.group) if self.is_initialized() else -1

    def __repr__(self) -> str:
        return f"group: {self.group}, initialized: {self.is_initialized()}, world size: {self.world_size()}"

    @staticmethod
    def get_dist_syncd_obj(
        obj: Any,
        groups: "DistributedProcessGroup | list[DistributedProcessGroup]",
        op: Callable,
    ):
        """Get the distributed synchronized object across the specified distributed groups."""

        def _get_dist_syncd_obj_across_group(obj, group: DistributedProcessGroup):
            if not group.is_initialized():
                return obj
            obj_list = [None] * group.world_size()
            torch.distributed.all_gather_object(obj_list, obj, group=group.group)
            return op(obj_list)

        for group in groups if isinstance(groups, list) else [groups]:
            obj = _get_dist_syncd_obj_across_group(obj, group)

        return obj


class ParallelState:
    """A class to manage various parallel groups such as data parallel, tensor parallel etc.

    Specify the parallel groups of type :class:`torch.distributed.ProcessGroup` for the current module.
    If the parallel group is not used, it should be set to `-1`.
    if a parallel group is `None`, it will use the default PyTorch distributed process group which is the whole world.
    """

    def __init__(
        self,
        data_parallel_group: torch.distributed.ProcessGroup | int | None = None,
        tensor_parallel_group: torch.distributed.ProcessGroup | int | None = -1,
        expert_model_parallel_group: torch.distributed.ProcessGroup | int | None = -1,
    ):
        """Initialize the parallel state."""
        self.data_parallel_group = DistributedProcessGroup(data_parallel_group)
        self.tensor_parallel_group = DistributedProcessGroup(tensor_parallel_group)
        self.expert_model_parallel_group = DistributedProcessGroup(expert_model_parallel_group)

    def __repr__(self) -> str:
        parallel_groups = (
            f"data_parallel_group: {self.data_parallel_group}, "
            f"tensor_parallel_group: {self.tensor_parallel_group}, "
            f"expert_model_parallel_group: {self.expert_model_parallel_group}"
        )
        return parallel_groups


def get_group(ranks: list[int]):
    """Returns the process group if torch.distributed.is_initialized()."""
    # NCCL has an issue with calling barrier. So we just use the gloo backebnd for group barriers.
    return torch.distributed.new_group(ranks, backend="gloo") if is_initialized() else None


def is_dtensor_sharded(model):
    """Returns True if the model is using DTensor."""
    return any(isinstance(param, DTensor) for param in model.parameters()) or any(
        isinstance(param, DTensor) for param in model.buffers()
    )


class FileLock:
    """Mutex object for writing to a file atomically using the O_EXCL directive on Unix filesystems."""

    def __init__(
        self,
        lockfile_path: str,
        all_acquire: bool = False,
        poll_time: float = 0.25,
    ):
        """Constructor.

        Args:
            lockfile_path: Path to a nonexistent file to be used as the locking mechanism.
            all_acquire: Will keep retrying to acquire a lock if True.
            poll_time: Sleep interval between retries.
        """
        self.lockfile_path = lockfile_path
        self.all_acquire = all_acquire
        self.poll_time = poll_time
        self.handle = None

    def try_acquire(self):
        try:
            self.handle = os.open(self.lockfile_path, os.O_CREAT | os.O_EXCL)
            return True
        except FileExistsError:
            return False

    def wait(self):
        while os.path.exists(self.lockfile_path):
            time.sleep(self.poll_time)

    def release(self):
        if self.handle is not None:
            os.close(self.handle)
        os.remove(self.lockfile_path)

    def __enter__(self):
        while True:
            if self.try_acquire() or not self.all_acquire:
                break
            self.wait()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.release()
