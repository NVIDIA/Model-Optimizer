# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""FSDP2 wrapping of a model's decoder layers, which needs no Hugging Face package.

The loaders (:mod:`modelopt.torch.utils.plugins.model_load_utils`) wrap with :func:`fsdp2_wrap` and
export it; it lives here so that a model whose decoder layers are registered with
``LayerActivationCollector`` can be wrapped without ``transformers`` or ``huggingface_hub``.
"""

from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from typing import Any

import torch
from torch.distributed.fsdp import CPUOffloadPolicy, FSDPModule, fully_shard
from torch.distributed.tensor import DTensor

from modelopt.torch.utils.distributed import _recorded_plan
from modelopt.torch.utils.logging import warn_rank_0

__all__ = ["fsdp2_root_unsharded", "fsdp2_wrap"]


def _off_dtype_params(model, skip: Iterable[torch.nn.Parameter] = ()) -> set[torch.nn.Parameter]:
    """Params whose dtype differs from the model's dominant (by element count) param dtype.

    FSDP2 needs one dtype per shard group, but HF models routinely keep a few params in fp32 for
    stability (e.g. MoE router gates). Pass these to ``fully_shard(ignored_params=...)``.

    The params in ``skip``, which the caller leaves out of the shard groups anyway, neither count
    nor are returned: routed experts kept packed store an integer dtype that can outnumber the rest.

    TODO: Drop this and shard the off-dtype params once a stable PyTorch release includes FSDP2
    mixed-precision parameter dtype support (already on nightly).
    """
    skip_ids = {id(p) for p in skip}  # id-based: Tensor.__eq__ is elementwise
    params = {n: p for n, p in model.named_parameters() if id(p) not in skip_ids}
    numel_by_dtype: dict[torch.dtype, int] = {}
    for param in params.values():
        numel_by_dtype[param.dtype] = numel_by_dtype.get(param.dtype, 0) + param.numel()
    if len(numel_by_dtype) <= 1:
        return set()

    dominant = max(numel_by_dtype, key=lambda d: numel_by_dtype[d])
    off_dtype = {n: p for n, p in params.items() if p.dtype != dominant}
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


def _auxiliary_modules(model, decoder_layers) -> list[torch.nn.Module]:
    """Submodules beside the decoder stack that carry a weight matrix.

    E.g. a vision or audio tower, a multimodal projector, an MTP head. Left in the root's shard
    group they would be all-gathered on every forward, although text-only calibration never runs
    them; in their own group each is gathered only when it runs. Input and
    output embeddings stay with the root, as the two are often tied, and so do modules holding only
    1-D parameters (norms), which are too small to be worth a group of their own.
    """
    layer_ids = {id(layer) for layer in decoder_layers}
    # Modules on the path from the root to a decoder layer: their other children are the candidates.
    ancestor_ids = {id(model)}
    for name, module in model.named_modules():
        if id(module) in layer_ids:
            parts = name.split(".")
            for i in range(1, len(parts)):
                ancestor_ids.add(id(model.get_submodule(".".join(parts[:i]))))
    embedding_ids = set()
    for getter in ("get_input_embeddings", "get_output_embeddings"):
        try:
            embedding = getattr(model, getter)()
        except (AttributeError, NotImplementedError):
            embedding = None
        if embedding is not None:
            embedding_ids.add(id(embedding))

    aux = []
    for module in model.modules():
        if id(module) not in ancestor_ids:
            continue
        for child in module.children():
            if id(child) in ancestor_ids or id(child) in layer_ids or id(child) in embedding_ids:
                continue
            if any(id(m) in embedding_ids for m in child.modules()):
                continue
            if any(p.dim() >= 2 for p in child.parameters()):
                aux.append(child)
    return aux


@contextmanager
def fsdp2_root_unsharded(model: torch.nn.Module) -> Iterator[None]:
    """Gather the FSDP2 root's own parameters while a submodule of ``model`` is called directly.

    Calling a submodule skips the root's pre-forward, which makes ``model`` the FSDP2 root and
    unshards the parameters it holds outside its own units (e.g. the embeddings). No-op for a model
    that is not an FSDP2 root.
    """
    if not isinstance(model, FSDPModule):
        yield
        return
    fully_shard.state(model)._lazy_init()  # else the first unit called would claim the root
    model.unshard()
    try:
        yield
    finally:
        model.reshard()


def _register_output_capture(model) -> None:
    """Re-key Transformers' output-capture registry for the classes ``fully_shard`` swapped in.

    Transformers looks up which outputs a module can record by its class, and ``fully_shard``
    replaces that class with ``FSDP<cls>``. Without this, a wrapped vision tower returns
    ``hidden_states=None`` and a VLM that reads its features from them fails on the first image.
    """
    recorders = [
        m
        for m in model.modules()
        if isinstance(m, FSDPModule) and getattr(m, "_can_record_outputs", None)
    ]
    try:
        from transformers.utils.output_capturing import _CAN_RECORD_REGISTRY
    except ImportError:
        # Every supported Transformers (5.5+) defines the registry here, so recorders without it
        # mean the private API moved and re-keying would silently stop working.
        if recorders:
            warn_rank_0(
                "transformers.utils.output_capturing._CAN_RECORD_REGISTRY not found, so FSDP2 "
                f"cannot re-key it for {type(recorders[0]).__name__}; modules that record outputs "
                "(e.g. a vision tower's hidden_states) may return None."
            )
        return
    for module in recorders:
        _CAN_RECORD_REGISTRY[str(type(module))] = module._can_record_outputs


def fsdp2_wrap(model, shard_root=True, mp_policy=None, cpu_offload: bool = False):
    """Auto-detect a HF causal-LM's decoder layers and FSDP2 ``fully_shard`` each one.

    Its callers load causal LMs and text-only VLMs; the expert-parallel dense load, which also takes
    VLMs, picks its blocks through
    :func:`~modelopt.torch.utils.plugins.model_load_utils._fsdp2_wrap_modules`, keeping a vision
    tower replicated.

    By default (``shard_root=True``) the root module is wrapped too, so embed/lm_head/norm are
    sharded instead of replicated per rank; pass ``shard_root=False`` to leave the root replicated
    (only decoder layers sharded). With the root sharded, auxiliary modules beside the decoder stack
    (a vision tower, a projector, an MTP head; see :func:`_auxiliary_modules`) get groups of their
    own. Returns the detected decoder layers so callers can reuse the detection result.

    Parameters whose dtype differs from the model's dominant one are excluded from the wrap (see
    :func:`_off_dtype_params`), since FSDP2 rejects a shard group that mixes dtypes. They stay
    replicated and are moved onto the shards' device, which ``fully_shard`` does not do for the
    params it ignores.

    It records no distribution plan, and refuses a model that, or any of whose submodules, records
    one, as a model ``load_pretrained`` returned does: wrap one loaded with ``from_pretrained``.
    """
    if (found := _recorded_plan(model)) is not None:
        # A model recorded as replicated would read so while sharded, misleading the export, and one
        # recorded as sharded is wrapped already. A ValueError, as set_distribution_plan raises.
        where, recorded = found
        raise ValueError(
            f"{where} is already recorded as distributed by {recorded!r}, and a plan is recorded "
            "once. To shard a model yourself, load it with from_pretrained rather than "
            "load_pretrained, which records the plan it loaded by, then record yours with "
            "set_distribution_plan() once it is wrapped."
        )
    # Importing the quantization package registers every mode and plugin; only a wrap needs it.
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
        for module in _auxiliary_modules(model, decoder_layers):
            fully_shard(module, **fsdp_kwargs)
        fully_shard(model, **fsdp_kwargs)
    if ignored_params:
        _move_to_fsdp_device(model, ignored_params)
    if config is not None and architectures:
        config.architectures = architectures
    _register_output_capture(model)
    return decoder_layers
