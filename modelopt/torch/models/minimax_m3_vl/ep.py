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

"""DP=EP all-to-all support for transformers-native ``minimax_m3_vl`` (MiniMax-M3, a VLM MoE).

M3's text MoE (``MiniMaxM3VLSparseMoeBlock``) already follows the ExpertsInterface **sparse-dispatch**
convention -- its forward runs the router inline and calls ``self.experts(hidden, top_k_index,
top_k_weights)`` -- and its experts store the FUSED grouped params ``gate_up_proj [E, 2*inter, hidden]``
/ ``down_proj [E, hidden, inter]`` (the standard ``[E, out, in]`` layout, i.e. ``is_transposed=False``,
so NVFP4 per-block quant works). The only gap is that ``MiniMaxM3VLExperts.forward`` is HARDCODED (a
per-expert Python loop) instead of dispatching through the transformers ``ExpertsInterface``, so the
generic ``torch_a2a`` transport (``modelopt.torch.utils.moe``) has no seam to take over. This plugin:

1. Decorates ``MiniMaxM3VLExperts`` with ``use_experts_implementation`` (``is_transposed=False``,
   ``has_gate=True``, ``has_bias=False``) so its forward dispatches through ExpertsInterface and honors
   ``config._experts_implementation`` at forward time. The generic ``torch_a2a`` experts forward then
   takes over -- NCCL all-to-all dispatch -> grouped-GEMM ``_grouped_linear`` -> combine -- preserving
   M3's custom SwiGLU via the module's own ``_apply_gate``. Unlike llama4, M3's MoE block already calls
   ``experts(hidden, index, weights)``, so NO block-forward rewrite is needed.

2. Sets ``base_model_ep_plan`` on ``text_config`` (M3 is a VLM; the plan bubbles up under
   ``language_model.*``): the fused expert params -> ``grouped_gemm``; the experts container ->
   ``moe_tp_experts`` (the a2a swap in ``load_and_prepare_ep`` derives ``torch_a2a_experts``). NO
   ``ep_router`` entry -- M3's ``TopKRouter`` runs inline over the REPLICATED full router weight and
   emits GLOBAL expert ids, so it stays a plain replicated module (as for deepseek_v3 / llama4). The
   ``shared_experts`` (a dense always-on MLP) are absent from the plan, so they stay dense/replicated.

Load with ``trust_remote_code=False`` so the transformers-native ``MiniMaxM3VLConfig`` + modeling are
used (consistent field set); the repo's remote ``auto_map`` config is a thinner ``PreTrainedConfig`` that
omits native defaults (``rope_parameters`` / vision ``temporal_patch_size``) and desyncs from the native
modeling.

Version floor: transformers >= 5.13 (first release carrying ``minimax_m3_vl``).
"""

from modelopt.torch.utils.moe import require_transformers

BASE_MODEL_EP_PLAN = {
    "layers.*.mlp.experts.gate_up_proj": "grouped_gemm",
    "layers.*.mlp.experts.down_proj": "grouped_gemm",
    "layers.*.mlp.experts": "moe_tp_experts",
}

_PATCHED = False


def _patch_minimax_m3_experts() -> None:
    """Route M3's experts through the ExpertsInterface so the generic transport takes over.

    Idempotently route ``MiniMaxM3VLExperts.forward`` through the ExpertsInterface so ``torch_a2a``
    can take over (its per-expert loop is otherwise hardcoded and blind to EP sharding).
    """
    global _PATCHED
    if _PATCHED:
        return
    from transformers.integrations.moe import use_experts_implementation
    from transformers.models.minimax_m3_vl import modeling_minimax_m3_vl as m

    # is_transposed=False: gate_up_proj/down_proj are [E, out, in] (like qwen3/deepseek, unlike llama4).
    # has_gate=True: gate_up is FUSED (2*inter) and split by the module's SwiGLU ``_apply_gate``.
    use_experts_implementation(
        m.MiniMaxM3VLExperts,
        is_transposed=False,
        has_bias=False,
        has_gate=True,
    )
    _PATCHED = True


def prepare(hf_config, **_):
    """Adapt M3's experts to ExpertsInterface + set the EP plan on ``text_config``. Returns the config."""
    require_transformers("5.13", "minimax_m3_vl expert-parallel PTQ")
    _patch_minimax_m3_experts()
    text_cfg = getattr(hf_config, "text_config", None) or hf_config
    if not getattr(text_cfg, "base_model_ep_plan", None):
        text_cfg.base_model_ep_plan = dict(BASE_MODEL_EP_PLAN)
    text_cfg._experts_implementation = "torch_a2a"
    return hf_config
