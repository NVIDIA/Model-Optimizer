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

from typing import Any

import torch
from torch.distributed.fsdp import CPUOffloadPolicy, fully_shard
from torch.distributed.tensor import DTensor

from modelopt.torch.utils.distributed import _PLAN_ATTR
from modelopt.torch.utils.logging import warn_rank_0

__all__ = ["fsdp2_wrap"]


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

    It records no distribution plan: a caller wrapping a model itself records one once it is wrapped
    (:func:`~modelopt.torch.utils.distributed.set_distribution_plan`). It refuses a model whose
    recorded plan replicates it, as the plan :func:`load_pretrained` records on a model it loads
    whole does: a plan is recorded once, so wrap a model loaded with ``from_pretrained`` instead.
    """
    recorded = getattr(model, _PLAN_ATTR, None)
    if recorded is not None and not recorded.is_fsdp_dense:
        # Sharded, the model would read as replicated: the export would gather it as if it were.
        raise RuntimeError(
            f"{type(model).__name__} is recorded as distributed by {recorded!r}, which replicates "
            "it, and a plan is recorded once. To shard a model yourself, load it with "
            "from_pretrained rather than load_pretrained, which records the plan it loaded by, "
            "then record yours with set_distribution_plan() once it is wrapped."
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
        fully_shard(model, **fsdp_kwargs)
    if ignored_params:
        _move_to_fsdp_device(model, ignored_params)
    if config is not None and architectures:
        config.architectures = architectures
    return decoder_layers
