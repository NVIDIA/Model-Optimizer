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

"""The expert-parallel style NemotronH's block-routed MoE needs.

Everything else in this package is data; this is the one piece that has to be code, because
transformers resolves parallel styles through a global registry rather than from a config.
"""

_REGISTERED = False


def _register_style() -> None:
    """Register ``NemotronHExpertsEP`` under the name the spec's plan refers to.

    Imported lazily so this package stays importable without torch or transformers, and
    idempotent because a PTQ run may prepare several models in one process.
    """
    global _REGISTERED
    if _REGISTERED:
        return

    import torch
    from transformers.integrations.tensor_parallel import (
        ALL_PARALLEL_STYLES,
        MoeTensorParalellExperts,
        all_reduce_forward,
    )

    class NemotronHExpertsEP(MoeTensorParalellExperts):
        """Expert parallelism for a gate that emits logits only.

        The usual ``ep_router`` style remaps expert indices on the ROUTER's output, which
        requires the router to return ``(scores, indices)``. NemotronH's gate returns logits
        and the group-limited top-k happens inside the MoE block, so there is nothing for
        ``ep_router`` to remap and the experts would receive global indices they cannot use.

        So the remap happens at the EXPERTS' input instead: a token routed to one of this
        rank's local experts gets a local index, and a token belonging elsewhere is pointed at
        local index 0 with weight zero, which makes the experts' ``index_add`` a no-op for it.
        ``grouped_gemm`` has already set ``num_experts`` to the local count, and the inherited
        ``all_reduce_forward`` sums each rank's partial expert output.
        """

        def _prepare_input_fn(self, mod, inputs, device_mesh):
            hidden, top_k_index, top_k_weights = inputs[0], inputs[1], inputs[2]
            local = mod.num_experts  # local count, after grouped_gemm shards dim 0
            start = device_mesh.get_local_rank() * local
            is_local = (top_k_index >= start) & (top_k_index < start + local)
            local_index = torch.where(is_local, top_k_index - start, torch.zeros_like(top_k_index))
            local_weights = torch.where(is_local, top_k_weights, torch.zeros_like(top_k_weights))
            return (hidden, local_index, local_weights)

        def _prepare_output_fn(self, mod, outputs, device_mesh):
            return all_reduce_forward(outputs, device_mesh)

    ALL_PARALLEL_STYLES.register("nemotron_experts_ep", NemotronHExpertsEP())
    _REGISTERED = True


def prepare(hf_config, **_):
    """Register the style the plan names, then hand the config back unchanged."""
    _register_style()
    return hf_config


BASE_MODEL_EP_PLAN = {
    "layers.*.mixer.experts.up_proj": "grouped_gemm",
    "layers.*.mixer.experts.down_proj": "grouped_gemm",
    "layers.*.mixer.experts": "nemotron_experts_ep",
}
"""Ungated experts, so ``up_proj``/``down_proj`` rather than a gate/up pair."""
