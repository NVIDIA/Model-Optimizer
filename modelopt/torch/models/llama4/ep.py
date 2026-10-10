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

"""DP=EP a2a support for transformers-native ``llama4`` (Llama-4 Maverick/Scout).

Unlike every other MoE we run on a2a, Llama-4's ``Llama4TextMoe`` does NOT use the transformers
``ExpertsInterface`` sparse-dispatch convention. Its reference forward:

    router_scores, router_logits = self.router(hidden)          # router returns TWO values
    routed_in = hidden.repeat(num_experts, 1) * router_scores.T # compute ALL experts, DENSELY
    routed_out = self.experts(routed_in)                        # experts() takes ONE pre-routed tensor
    out = shared_expert(hidden) + routed_out.reshape(E, -1, H).sum(0)

i.e. routing is done inline (not via a wrappable ``RouterParallel``), the experts are computed densely
(every token through every expert, masked by the sigmoid router scores), and the experts module is
called with a single already-routed tensor -- so a2a has no ``router -> experts(hidden, top_k_index,
top_k_weights)`` seam to wrap. (This is why Llama-4 was the one MoE that didn't fit the generic path.)

This plugin monkey-patches transformers' llama4 modeling to fit the ExpertsInterface contract so the
generic ``torch_a2a`` transport takes over unchanged:

1. **Decorate ``Llama4TextExperts`` with ``use_experts_implementation``.** Llama-4 stores the expert
   weights as ``gate_up_proj`` ``[E, hidden, 2*inter]`` and ``down_proj`` ``[E, inter, hidden]`` -- the
   ``(num_experts, input_dim, output_dim)`` layout, i.e. ``is_transposed=True`` for ``_grouped_linear``
   (the opposite of qwen3/deepseek's ``[E, out, in]``). The decorator makes ``forward`` dispatch through
   ``ExpertsInterface`` and exposes ``has_gate``/``has_bias``/``is_transposed``/``_apply_gate`` -- exactly
   what ``moe.torch_a2a_experts_forward`` reads. The ``grouped_gemm`` shard updates ``num_experts`` to the
   per-rank local count, so ``num_global = num_local * ep`` is correct.

2. **Replace ``Llama4TextMoe.forward`` with the sparse convention:** extract the real top-k from the
   router (``topk(router_logits)`` -> global expert ids; ``sigmoid`` of the top values as weights, matching
   Llama-4's gating), call ``self.experts(hidden, top_k_index, top_k_weights)`` (the a2a dispatch/combine,
   which applies the weights and sums over experts -> ``[T, H]``), then add the replicated shared expert.
   Doing the routing here also bypasses ``RouterParallel`` entirely -- Llama4's router returning 2 values
   (not the 3 ``RouterParallel._prepare_output_fn`` expects) is what broke the stock-EP attempt.

3. **Drop the ``ep_router`` entry from ``base_model_ep_plan``** so the router stays a plain replicated
   ``Linear`` (we route inline). The grouped_gemm expert-param entries stay; the a2a swap derives the
   ``...experts`` container as ``torch_a2a_experts`` and drops the attention / shared-FFN plain-TP styles
   (replicated, no TP dim in the DP=EP mesh).

Because llama4 stores its experts transposed, this also needs ModelOpt's ``_QuantFusedExperts`` grouped
quantization to accept the ``is_transposed`` layout -- supported for per-tensor weight quant (FP8); NVFP4
(per-block) transposed experts are not yet supported (the block axis would be wrong) and will raise.

Version floor: transformers >= 5.6 (the ExpertsInterface / EP framework; enforced also by
``moe.load_and_prepare_ep``). Validated FP8 end-to-end on a tiny llama4 (Llama4ForCausalLM, EP4/1 node,
``validation/make_tiny_llama4.py``). Not yet run on real Llama-4-Maverick (4 nodes) or for NVFP4.
"""

from modelopt.torch.utils.moe import require_transformers

_PATCHED = False


def _patch_llama4_modeling() -> None:
    """Idempotently adapt transformers' llama4 MoE to the ExpertsInterface sparse-dispatch convention."""
    global _PATCHED
    if _PATCHED:
        return
    import torch
    from transformers.integrations.moe import use_experts_implementation
    from transformers.models.llama4 import modeling_llama4

    # (1) Make Llama4TextExperts an ExpertsInterface experts class: dispatch forward through
    # ALL_EXPERTS_FUNCTIONS (so config._experts_implementation="torch_a2a" swaps in the a2a forward) and
    # expose has_gate/has_bias/is_transposed/_apply_gate. is_transposed=True: gate_up_proj/down_proj are
    # stored [E, in, out] (Llama-4 does bmm(hidden, gate_up_proj)), unlike qwen3/deepseek's [E, out, in].
    use_experts_implementation(
        modeling_llama4.Llama4TextExperts,
        is_transposed=True,
        has_bias=False,
        has_gate=True,
    )

    # (2) Sparse-convention MoE forward: route inline, dispatch the real top-k into the experts (a2a
    # applies the weights + sums), add the replicated shared expert. Bypasses RouterParallel.
    def _moe_forward(self, hidden_states):
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_out = self.router(hidden_states)
        # Llama4Router returns (router_scores, router_logits); use the raw logits for the top-k.
        router_logits = router_out[-1] if isinstance(router_out, tuple) else router_out
        top_values, top_index = torch.topk(router_logits, self.top_k, dim=-1)
        # Llama-4 gates with sigmoid of the selected logits (== the non-zero entries of router_scores).
        top_weights = torch.sigmoid(top_values.float()).to(hidden_states.dtype)
        routed_out = self.experts(
            hidden_states, top_index, top_weights
        )  # a2a dispatch/combine -> [T, H]
        out = self.shared_expert(hidden_states)
        out = out + routed_out.view(out.shape)
        return out, router_logits

    modeling_llama4.Llama4TextMoe.forward = _moe_forward
    _PATCHED = True


def prepare(hf_config, **_):
    """Patch llama4 modeling for a2a and drop the router ep_plan entry. Returns the config."""
    require_transformers("5.6", "llama4 expert-parallel PTQ (ExpertsInterface a2a adapter)")
    _patch_llama4_modeling()
    # base_model_ep_plan lives on the text config for the Llama-4 VLM (Llama4TextConfig).
    text_cfg = getattr(hf_config, "text_config", None) or hf_config
    plan = dict(getattr(text_cfg, "base_model_ep_plan", None) or {})
    # We route inline in the patched Llama4TextMoe.forward, so keep the router a plain replicated Linear
    # (no torch_a2a_router / RouterParallel wrapping). The a2a swap keeps the grouped_gemm expert params
    # and derives the experts container; the attention / shared-FFN plain-TP styles are dropped (replicated).
    plan = {k: v for k, v in plan.items() if v != "ep_router"}
    text_cfg.base_model_ep_plan = plan
    # The decorated Llama4TextExperts.forward dispatches on ``config._experts_implementation``. Because our
    # decoration is applied at runtime (not in the source file), transformers' ``_can_set_experts_
    # implementation`` source heuristic may not see it and so from_pretrained's ``experts_implementation``
    # kwarg wouldn't stick; set it on the config now so it's present pre-load (load_and_prepare_ep also
    # force-sets it post-load, but only where the attr already exists). ``torch_a2a`` is registered into
    # ExpertsInterface by ``register_torch_a2a()`` (called from load_and_prepare_ep) before any forward runs.
    text_cfg._experts_implementation = "torch_a2a"
    return hf_config
