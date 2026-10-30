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

"""DP=EP all-to-all support for Kimi-K3 (``kimi_k3``).

a remote-code VLM whose text backbone is ``kimi_linear`` (KDA/linear attention + DeepSeek-V3-style routed MoE with a
LATENT expert space).

Kimi-K3 has no transformers-native modeling (``CONFIG_MAPPING_NAMES`` has no ``kimi_linear``/``kimi_k3``),
so -- unlike ``kimi_k25``, which swaps to native ``deepseek_v3`` -- we cannot get EP-capable fused experts
by swapping in a native backbone. We must adapt the checkpoint's REMOTE code in place. Two structural
obstacles and how this plugin removes them:

1. **Per-expert ``nn.ModuleList``.** The remote ``KimiSparseMoeBlock`` holds ``self.experts`` as an
   ``nn.ModuleList`` of ``num_experts`` ``KimiBlockSparseMLP`` (each a separate ``w1``=gate / ``w3``=up /
   ``w2``=down Linear), and its ``forward`` runs a hardcoded per-expert Python loop (``moe_infer``). The
   transformers EP framework (and ModelOpt's ``torch_a2a`` transport) instead needs the FUSED 3-D grouped
   params ``gate_up_proj [E, 2*I, H]`` / ``down_proj [E, H, I]`` behind ``ExpertsInterface``. This plugin
   replaces ``self.experts`` with a fused ``KimiK3Experts`` module (decorated with
   ``use_experts_implementation``) and rewrites ``KimiSparseMoeBlock.forward`` to dispatch through it -- so
   the generic ``torch_a2a`` experts forward (NCCL all-to-all -> quantizable ``_grouped_linear``) takes over.

2. **No expert fusion for the checkpoint's names.** The checkpoint stores per-expert ``w1``/``w3``/``w2``
   (gate/up/down), and transformers registers no expert fusion for custom-code models. This plugin
   registers ``kimi_linear`` WeightConverters that fuse ``...experts.<i>.w1|w3`` into ``gate_up_proj`` and
   stack ``w2`` into ``down_proj``, which the ``grouped_gemm`` plan entries then EP-shard.

3. **Compressed (mxfp4) experts.** In this checkpoint the 896 routed experts are compressed-tensors
   ``mxfp4-pack-quantized`` (4-bit, group 32; stored as ``w1/w3/w2.weight_packed`` + ``.weight_scale``) --
   everything else (attention, shared experts, latent projections, dense layer, embeddings, lm_head,
   vision) is plain bf16. This plugin installs the format-agnostic on-the-fly decompressor
   (``_onfly_decompress``) so from_pretrained dequantizes each per-expert packed weight -> bf16 INSIDE
   the fused-expert conversion (EP-shard-aware -- only this rank's local experts), then drops the
   ``quantization_config`` so transformers does not build a compressed-tensors HF quantizer. ModelOpt PTQ
   then inserts fp8/nvfp4 quantizers on the resulting bf16 fused params. (Re-quantizing an already-4-bit
   model to fp8/nvfp4 is lossy-on-lossy; that is the meaning of PTQ-ing this checkpoint.)

Kimi-K3 specifics preserved as DENSE / replicated (NOT quantized through the EP experts path):
  * the LATENT MoE wrapper -- ``routed_expert_down_proj`` (H->H_latent) / ``routed_expert_up_proj``
    (H_latent->H) and the optional ``routed_expert_norm`` -- the experts run in the H_latent
    (``routed_expert_hidden_size``) space between them; the block forward keeps these around the fused call.
  * the always-on ``shared_experts`` (a dense ``KimiMLP``) and the inline ``gate`` (``KimiMoEGate``,
    sigmoid noaux_tc). The gate runs over the REPLICATED router weight and emits GLOBAL top-k ids, so it
    stays a plain replicated module (no ``ep_router`` plan entry -- as for deepseek_v3's global-id
    gate). The custom ``situ`` SwiGLU (``SituAndMul``) is preserved via the fused module's
    ``_apply_gate``.

VLM: the plan is set on ``text_config`` and bubbles up under ``language_model.model.*`` (the VLM builds
``self.language_model = KimiLinearForCausalLM(config.text_config)``); vision tower + mm projector stay bf16.

Also shims ``OutputRecorder`` back into ``transformers.utils.generic`` (the remote code imports it there,
but it moved to ``transformers.modeling_utils``) so the modeling imports at all.

Version floor: transformers >= 5.13, which it was developed and validated against (the spec declares it),
with flash-linear-attention (FLA) 0.5.0, which the remote code's KDA layers need at runtime.
"""

import torch
import torch.nn as nn

from .. import _onfly_decompress
from .._remote_code import stage_full_remote_closure
from ._remote_fixes import (
    _find_module_defining,
    _install_output_recorder_shim,
    _patch_kda_a_log,
    _patch_kda_beta_sigmoid,
    _patch_kimi_cache_get_mask_sizes,
    _shim_create_causal_mask,
    _shim_tie_weights_kwargs,
)

# base-relative EP plan (bubbles to `language_model.model.<key>` in the VLM). MoE lives at
# `layers.*.block_sparse_moe` (NOT `.mlp`; layer 0 is dense `.mlp`). Fused expert params shard dim-0 via
# grouped_gemm; the experts container gets the EP group via moe_tp_experts (-> torch_a2a_experts for a2a).
# No `ep_router` entry: KimiMoEGate runs inline over the replicated router weight and emits GLOBAL ids.
BASE_MODEL_EP_PLAN = {
    "layers.*.block_sparse_moe.experts.gate_up_proj": "grouped_gemm",
    "layers.*.block_sparse_moe.experts.down_proj": "grouped_gemm",
    "layers.*.block_sparse_moe.experts": "moe_tp_experts",
}


def _find_remote_module(package: str):
    """Return the loaded remote backbone module.

    the one in ``package`` that DEFINES ``KimiSparseMoeBlock`` (the Kimi MoE block this plugin patches) -- or
    None. Name-independent; see ``_find_module_defining``.
    """
    return _find_module_defining("KimiSparseMoeBlock", package)


def _build_fused_experts_class(kmod):
    """Build the fused experts class the expert-parallel path needs.

    Build the fused ``KimiK3Experts`` class (bound to the remote module's ``SituAndMul``), decorated
    with ``use_experts_implementation`` so its forward dispatches through transformers' ``ExpertsInterface``
    and honors ``config._experts_implementation`` (``torch_a2a`` at EP-PTQ time). Stores the standard fused
    grouped params ``gate_up_proj [E, 2*I, H_latent]`` / ``down_proj [E, H_latent, I]`` (``is_transposed=
    False``, ``[E, out, in]``). ``_apply_gate`` preserves Kimi's ``situ`` SwiGLU.
    """
    from transformers.integrations.moe import use_experts_implementation

    SituAndMul = kmod.SituAndMul  # noqa: N806 (a class, keeping its own name)
    _get_situ = kmod._get_situ_activation_params

    class KimiK3Experts(nn.Module):
        def __init__(self, config, hidden_dim):
            super().__init__()
            self.num_experts = config.num_experts
            # The width the MoE block runs its experts at: the latent one under latent MoE, else hidden.
            self.hidden_dim = hidden_dim
            self.intermediate_dim = config.moe_intermediate_size
            self.gate_up_proj = nn.Parameter(
                torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim)
            )
            self.down_proj = nn.Parameter(
                torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim)
            )
            # explicit attrs the torch_a2a experts forward reads (also set by use_experts_implementation).
            self.has_gate = True
            self.has_bias = False
            self.is_transposed = False
            if getattr(config, "hidden_act", None) == "situ":
                beta, linear_beta = _get_situ(config)
                self._situ = SituAndMul(beta=beta, linear_beta=linear_beta)
            else:
                self._situ = None

        def _apply_gate(self, gate_up: torch.Tensor) -> torch.Tensor:
            # SituAndMul takes the FULL concat [gate; up] and splits internally -- matches the fused
            # gate_up_proj row layout (rows [0:I]=gate from w1, [I:2I]=up from w3).
            if self._situ is not None:
                return self._situ(gate_up)
            gate, up = gate_up.chunk(2, dim=-1)
            return nn.functional.silu(gate) * up

        def forward(self, hidden_states, top_k_index, top_k_weights):
            # Fallback (non-EP / eager) dense per-expert loop; the EP-PTQ path replaces this with the
            # torch_a2a experts forward via config._experts_implementation. Mirrors the native
            # DeepseekV3Experts/MiniMaxM3VLExperts reference loop, with situ gating.
            final = torch.zeros_like(hidden_states)
            with torch.no_grad():
                # One class past the experts: the id a router gives a token routed to no local expert.
                mask = nn.functional.one_hot(top_k_index, num_classes=self.num_experts + 1).permute(
                    2, 1, 0
                )
                hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()
            for expert_idx in hit:
                expert_idx = expert_idx[0]
                if expert_idx == self.num_experts:
                    continue
                top_k_pos, token_idx = torch.where(mask[expert_idx])
                cur = self._apply_gate(
                    nn.functional.linear(hidden_states[token_idx], self.gate_up_proj[expert_idx])
                )
                cur = nn.functional.linear(cur, self.down_proj[expert_idx])
                cur = cur * top_k_weights[token_idx, top_k_pos, None]
                final.index_add_(0, token_idx, cur.to(final.dtype))
            return final

    # is_transposed=False: [E, out, in] (like deepseek/qwen). has_gate=True: gate_up is FUSED (2*I) and
    # split by _apply_gate. has_bias=False: bias-free experts.
    use_experts_implementation(KimiK3Experts, is_transposed=False, has_bias=False, has_gate=True)
    return KimiK3Experts


def _patch_kimi_moe_block(kmod) -> None:
    """Replace the per-expert ModuleList with the fused experts module.

    Swap ``KimiSparseMoeBlock.experts`` (per-expert ModuleList) for the fused experts module and rewrite
    its ``forward`` to dispatch through it (keeping the latent down/up projections, optional norm, and
    shared experts around the fused call). Idempotent.
    """
    from modelopt.torch.utils import print_rank_0

    Block = kmod.KimiSparseMoeBlock  # noqa: N806 (a class, keeping its own name)
    if getattr(Block, "_kimi_k3_ep_patched", False):
        return
    FusedExperts = _build_fused_experts_class(kmod)  # noqa: N806 (a class, keeping its own name)

    _orig_init = Block.__init__

    def _init(self, config):
        _orig_init(self, config)
        # Replace the ModuleList of per-expert MLPs with the fused grouped-param module (EP-capable), as
        # wide as the space the block runs its experts in. The block decides latent MoE itself (Kimi-K3's
        # config has no use_latent_moe field) and the forward below follows it, so read it off the block.
        del self.experts
        width = config.routed_expert_hidden_size if self.use_latent_moe else config.hidden_size
        self.experts = FusedExperts(config, width)

    def _forward(self, hidden_states):
        identity = hidden_states
        orig_shape = hidden_states.shape
        topk_idx, topk_weight = self.gate(hidden_states)
        hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
        if self.use_latent_moe:
            hidden_states = self.routed_expert_down_proj(hidden_states)
        # Fused / a2a experts: returns the top-k-combined output in the latent space.
        y = self.experts(hidden_states, topk_idx, topk_weight)
        if self.use_latent_moe:
            if self.latent_moe_use_norm:
                y = self.routed_expert_norm(y)
            y = self.routed_expert_up_proj(y)
        y = y.reshape(orig_shape)
        if self.config.num_shared_experts is not None:
            y = y + self.shared_experts(identity)
        return y

    Block.__init__ = _init
    Block.forward = _forward
    Block._kimi_k3_ep_patched = True
    print_rank_0("[kimi_k3] patched KimiSparseMoeBlock: fused experts + ExpertsInterface forward")


def _force_load_remote_modeling(hf_config, ckpt_path: str, trust_remote_code: bool) -> type:
    """Import the remote modeling now, so its classes can be patched before the model is built.

    Force the remote modeling into ``sys.modules`` (so we can monkey-patch its classes BEFORE
    ``from_pretrained`` instantiates the model) by resolving the model class the checkpoint's config
    declares in its ``auto_map`` -- no hardcoded remote file/module name. Runs the OutputRecorder + closure
    shims first. Returns that class, whose package holds this checkpoint's remote code.
    """
    _install_output_recorder_shim()
    stage_full_remote_closure()
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    auto_map = getattr(hf_config, "auto_map", None) or {}
    class_ref = auto_map.get("AutoModelForCausalLM") or auto_map.get("AutoModel")
    if not class_ref:
        raise RuntimeError(
            "kimi_k3: config.auto_map has no AutoModel/AutoModelForCausalLM entry to force-load the remote "
            "modeling"
        )
    return get_class_from_dynamic_module(class_ref, ckpt_path, trust_remote_code=trust_remote_code)


def _register_kimi_k3_conversions() -> None:
    """Register the per-expert to fused-expert weight converters.

    Register the per-expert ``w1/w3/w2 -> gate_up_proj/down_proj`` fusion WeightConverters for Kimi-K3's
    ``kimi_linear`` backbone.

    transformers 5.13's ``get_model_conversion_mapping`` SKIPS building any weight conversions for
    custom-code (remote) models unless their class/model_type is in ``USER_REGISTERED_MAPPINGS``
    (``if submodule.is_custom_code() and class_name/model_type not in USER_REGISTERED_MAPPINGS: continue``).
    Kimi-K3 is remote code, so the per-expert -> fused-expert fusion is NEVER built: the checkpoint's
    ``experts.<i>.w1/w3/w2.weight_packed`` land as UNEXPECTED and the model's fused ``gate_up_proj`` /
    ``down_proj`` as MISSING (newly-initialized garbage, and allocated at FULL 896-expert size on every
    rank -> OOM at scale). Register the fusion converters ourselves (mirrors transformers' own native
    ``mlp.experts.*.w1/w3/w2.weight`` fusion, adapted to K3's ``block_sparse_moe.experts``): w1=gate,
    w3=up (Concatenate gate-then-up on dim 1), w2=down. The on-the-fly decompressor then rewrites these
    ``.weight`` sources to the checkpoint's ``.weight_packed`` + scales (mxfp4 -> bf16). Idempotent.
    """
    # transformers is optional. Its conversion API is in every version prepare() accepts, and a
    # load without it would leave the experts MISSING, so an ImportError here must stop the load.
    from transformers.conversion_mapping import register_checkpoint_conversion_mapping
    from transformers.core_model_loading import Concatenate, MergeModulelist, WeightConverter

    from modelopt.torch.utils import print_rank_0

    def _mapping():
        return [
            WeightConverter(
                source_patterns=[
                    "block_sparse_moe.experts.*.w1.weight",
                    "block_sparse_moe.experts.*.w3.weight",
                ],
                target_patterns="block_sparse_moe.experts.gate_up_proj",
                operations=[MergeModulelist(dim=0), Concatenate(dim=1)],
            ),
            WeightConverter(
                source_patterns="block_sparse_moe.experts.*.w2.weight",
                target_patterns="block_sparse_moe.experts.down_proj",
                operations=[MergeModulelist(dim=0)],
            ),
        ]

    # Register for the backbone MODEL_TYPE "kimi_linear" ONLY (not the class names). The VLM nests
    # KimiLinearForCausalLM (`language_model`) -> KimiLinearModel (`language_model.model`), both model_type
    # kimi_linear; registering by model_type applies the fusion ONCE (at the outer `language_model.` scope)
    # and the inner submodule is de-duplicated (ancestor already claimed the model_type). Registering the
    # CLASS names too would apply it a second time at `language_model.model.` scope -> the same expert keys
    # converted twice -> conflict at load.
    register_checkpoint_conversion_mapping("kimi_linear", _mapping(), overwrite=True)
    print_rank_0(
        "[kimi_k3] registered per-expert w1/w3/w2 -> gate_up_proj/down_proj fusion for kimi_linear "
        "(custom-code conversion; tf>=5.13 skips these otherwise)"
    )


def prepare(
    hf_config,
    *,
    ckpt_path: str | None = None,
    trust_remote_code: bool = False,
    **_,
):
    """Adapt Kimi-K3's remote MoE to the EP/a2a framework and set its EP plan.

    Steps: shim OutputRecorder -> force-load + monkey-patch the remote KimiSparseMoeBlock (fused experts +
    ExpertsInterface forward) -> set base_model_ep_plan on text_config -> register the w1/w3/w2 fusion
    converters and the decompressor. Returns the same config object, mutated in place.

    The 896 ``mxfp4-pack-quantized`` routed experts are DECOMPRESSED at load: ``_onfly_decompress``
    dequantizes each local expert's packed weight -> bf16 INSIDE the fused-expert conversion, stored as
    bf16 fused params, and the compressed-tensors ``quantization_config`` is dropped. ModelOpt then
    inserts/calibrates/folds the fp8/nvfp4 quantizers on those bf16 params. That is ~4x the packed HBM
    (the memory bottleneck), but PTQ needs the bf16 weights.
    """
    # Not at module scope: the model registry imports this module, and modelopt.torch.utils.moe
    # imports the registry.
    from modelopt.torch.utils.moe import register_torch_a2a

    if ckpt_path is None:
        raise ValueError(
            "kimi_k3: expert parallelism patches the checkpoint's own modeling code, so it needs "
            "the checkpoint's path (apply_distributed_support's ckpt_path)."
        )
    if not trust_remote_code:
        # Patching below imports the checkpoint's own modeling code, and transformers'
        # get_class_from_dynamic_module runs it without checking trust_remote_code itself.
        raise ValueError(
            "kimi_k3: expert parallelism imports the checkpoint's own modeling code; "
            "load it with trust_remote_code=True."
        )

    text_cfg = getattr(hf_config, "text_config", None) or hf_config

    # This checkpoint's remote code, not whichever revision this process imported first. Every patch
    # is idempotent, so a repeat prepare for the same checkpoint changes nothing.
    package = _force_load_remote_modeling(hf_config, ckpt_path, trust_remote_code).__module__
    package = package.rpartition(".")[0]
    kmod = _find_remote_module(package)
    if kmod is None:
        raise RuntimeError(
            "kimi_k3: could not locate the loaded remote module defining KimiSparseMoeBlock to patch"
        )
    _patch_kimi_moe_block(kmod)
    _patch_kda_a_log(kmod)
    _patch_kda_beta_sigmoid(kmod)
    _shim_create_causal_mask(kmod)
    _patch_kimi_cache_get_mask_sizes(kmod)
    _shim_tie_weights_kwargs(package)

    if not getattr(text_cfg, "base_model_ep_plan", None):
        # grouped_gemm shards the bf16 fused params; the experts container goes through moe_tp_experts
        # (-> torch_a2a_experts).
        text_cfg.base_model_ep_plan = dict(BASE_MODEL_EP_PLAN)
    # transformers refuses to build a model whose experts implementation it has not registered, and
    # the plan decision builds this one on the meta device before the expert-parallel load runs.
    register_torch_a2a()
    text_cfg._experts_implementation = "torch_a2a"

    # The 896 routed experts are compressed-tensors `mxfp4-pack-quantized` (4-bit, group 32); everything
    # else (attention, shared experts, latent projections, dense layer, embeddings, lm_head, vision) is
    # plain bf16. Register the per-expert w1/w3/w2 -> gate_up_proj/down_proj fusion converters (tf 5.13
    # skips these for custom-code models) BEFORE the decompressor patches get_model_conversion_mapping, so
    # the decompressor has converters to rewrite into compressed-aware (weight_packed -> bf16) ones. Drop
    # quantization_config afterwards so transformers builds no competing HF quantizer.
    _register_kimi_k3_conversions()
    # Plain bf16 experts come without a quantization_config: nothing to decompress.
    if any(getattr(cfg, "quantization_config", None) is not None for cfg in (hf_config, text_cfg)):
        _onfly_decompress.install_onfly_decompress(hf_config)
    for cfg in (hf_config, text_cfg):
        if getattr(cfg, "quantization_config", None) is not None:
            cfg.quantization_config = None
    return hf_config
