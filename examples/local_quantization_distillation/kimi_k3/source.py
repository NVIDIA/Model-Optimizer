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


"""Selective loading through Kimi's native checkpoint and model implementations."""

import json
import re
from pathlib import Path

import torch
from safetensors import safe_open
from torch import nn

from .prepare import selected_source_key

MOE_PREFIX = "language_model.model.layers.1.block_sparse_moe."


def read_selected(source, predicate):
    """Materialize selected tensors only, even when a shard contains other layers."""
    source = Path(source)
    index = json.loads((source / "model.safetensors.index.json").read_text())["weight_map"]
    selected = {}
    for file in sorted({file for key, file in index.items() if predicate(key)}):
        with safe_open(source / file, framework="pt", device="cpu") as handle:
            names = handle.keys()
            for key in names:
                if predicate(key):
                    selected[key] = handle.get_tensor(key)
    return selected


def native_options(source):
    """Resolve pinned native Kimi configuration without Transformers model loading."""
    from nemo_automodel.components.models.common import BackendConfig
    from nemo_automodel.components.models.kimi_k3.config import KimiK3Config

    config = KimiK3Config.from_pretrained(source).text_config
    backend = BackendConfig(attn="sdpa", linear="torch", experts="torch_mm", dispatcher="torch")
    return config, backend


class _BoundaryReachedError(Exception):
    pass


class _CaptureBoundary(nn.Module):
    def forward(self, hidden_states, padding_mask=None):
        raise _BoundaryReachedError

    def init_weights(self, *args, **kwargs):
        pass


def prefix_model(source):
    """Load embeddings and the short prefix, leaving the target MoE unallocated."""
    from nemo_automodel.components.models.kimi_k3.model import KimiK3ForCausalLM

    config, backend = native_options(source)
    with torch.device("meta"):
        model = KimiK3ForCausalLM(config, backend=backend, num_hidden_layers=2)
    model.model.layers["1"].mlp = _CaptureBoundary()
    model.lm_head = None
    model.model.norm = None
    model.model.output_attn_res_norm = None
    model.model.output_attn_res_proj = None
    model.to_empty(device="cuda")
    model.model.init_weights(torch.device("cuda"))
    weights = read_selected(
        source, lambda key: selected_source_key(key) and not key.startswith(MOE_PREFIX)
    )
    converted = model.state_dict_adapter.from_hf(weights)
    model.load_state_dict(converted, strict=True)
    return model.eval().requires_grad_(False)


@torch.no_grad()
def capture_inputs(model, batch):
    """Stop the native prefix exactly at the complete first-MoE input boundary."""
    captured = []
    handle = model.model.layers["1"].mlp.register_forward_pre_hook(
        lambda module, args: captured.append(args[0].detach().clone())
    )
    from modelopt.torch.distill.plugins.automodel import AutoModelLocalDistillationAdapter

    try:
        AutoModelLocalDistillationAdapter().teacher_microbatch(
            model,
            [],
            batch,
            lambda unit, record: None,
        )
    except _BoundaryReachedError:
        pass
    finally:
        handle.remove()
    if len(captured) != 1:
        raise ValueError("The native prefix did not reach exactly one target MoE boundary.")
    return captured[0]


def load_moe(source, ep_mesh):
    """Decode only rank-owned released experts through the native Kimi adapter."""
    from nemo_automodel.components.models.kimi_k3.model import KimiK3MoE, _build_moe_config
    from nemo_automodel.components.models.kimi_k3.state_dict_adapter import KimiK3StateDictAdapter

    config, backend = native_options(source)
    moe_config = _build_moe_config(config, torch.bfloat16, None)
    count, remainder = divmod(config.num_experts, ep_mesh.size() if ep_mesh is not None else 1)
    if remainder:
        raise ValueError("The expert count must divide the native EP mesh.")
    first = (ep_mesh.get_local_rank() if ep_mesh is not None else 0) * count

    def selected(key):
        if not key.startswith(MOE_PREFIX):
            return False
        match = re.search(r"\.experts\.(\d+)\.", key)
        return match is None or first <= int(match[1]) < first + count

    weights = read_selected(source, selected)
    adapter = KimiK3StateDictAdapter(config, moe_config, backend, dtype=torch.bfloat16)
    converted = adapter.from_hf(weights, device_mesh=ep_mesh)
    with torch.device("meta"):
        module = KimiK3MoE(config, moe_config, backend)
    expected = module.state_dict()
    prefix = "model.layers.1.mlp."
    local = {key.removeprefix(prefix): value for key, value in converted.items()}
    if local.keys() != expected.keys():
        raise ValueError(f"Selective MoE keys differ: {sorted(local.keys() ^ expected.keys())}")
    module.load_state_dict(
        {key: value.to(device="cuda", dtype=expected[key].dtype) for key, value in local.items()},
        strict=True,
        assign=True,
    )
    return module.eval().requires_grad_(False)
