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

Version floor: transformers >= 5.8, for the on-the-fly decompression. Developed and validated against
transformers 5.13 + flash-linear-attention (FLA) 0.5.0, which the remote code's KDA layers need at runtime.
"""

import contextlib
import sys

import torch
import torch.nn as nn

from .. import _onfly_decompress
from .._remote_code import stage_full_remote_closure

# base-relative EP plan (bubbles to `language_model.model.<key>` in the VLM). MoE lives at
# `layers.*.block_sparse_moe` (NOT `.mlp`; layer 0 is dense `.mlp`). Fused expert params shard dim-0 via
# grouped_gemm; the experts container gets the EP group via moe_tp_experts (-> torch_a2a_experts for a2a).
# No `ep_router` entry: KimiMoEGate runs inline over the replicated router weight and emits GLOBAL ids.
BASE_MODEL_EP_PLAN = {
    "layers.*.block_sparse_moe.experts.gate_up_proj": "grouped_gemm",
    "layers.*.block_sparse_moe.experts.down_proj": "grouped_gemm",
    "layers.*.block_sparse_moe.experts": "moe_tp_experts",
}

_PATCHED = False


def _install_output_recorder_shim() -> None:
    """Restore the generic-utils symbol the remote modeling imports.

    The remote ``modeling_kimi`` does ``from transformers.utils.generic import OutputRecorder`` (used
    once, as output-recording metadata), but that symbol moved to ``transformers.modeling_utils`` in
    recent transformers. Re-export it into ``utils.generic`` (or inject a benign stub) BEFORE the remote
    modeling imports so the import succeeds. Idempotent.
    """
    import transformers.utils.generic as _g

    if hasattr(_g, "OutputRecorder"):
        return
    found = None
    for _p in ("transformers.modeling_utils", "transformers.utils"):
        try:
            _m = __import__(_p, fromlist=["OutputRecorder"])
            if hasattr(_m, "OutputRecorder"):
                found = _m.OutputRecorder
                break
        except Exception:
            pass
    if found is None:

        class OutputRecorder:  # minimal stub: satisfies the import + the metadata use only
            def __init__(self, *a, **k):
                self.args, self.kwargs = a, k

        found = OutputRecorder
    _g.OutputRecorder = found


def _find_module_defining(class_name: str):
    """Return the loaded module that DEFINES a class named ``class_name``, or None.

    Identify the remote module by the CLASS it defines, NOT by its file name -- the plugin depends only on
    the class names it patches, never on the remote code's file layout, so this survives remote-file
    renames (e.g. ``modeling_kimi`` vs ``modeling_kimi_linear``). transformers
    owns parsing + importing the remote files (via ``trust_remote_code`` + ``auto_map``); we only locate
    the already-imported module. ``__module__ == module.__name__`` requires the class to be DEFINED here
    (not merely imported), which uniquely picks the defining module (e.g. the backbone that defines
    ``KimiSparseMoeBlock``, not the VLM wrapper that may import it). Guarded with try/except because
    iterating ``sys.modules`` can touch lazy/magic modules -- notably ``torch.classes``, whose
    ``__getattr__`` returns a namespace proxy for ANY name and raises on use (the proxy is not a ``type``,
    so it is rejected regardless).
    """
    for _name, mod in list(sys.modules.items()):
        if mod is None:
            continue
        try:
            obj = getattr(mod, class_name, None)
        except Exception:
            continue
        if isinstance(obj, type) and getattr(obj, "__module__", None) == getattr(
            mod, "__name__", None
        ):
            return mod
    return None


def _find_remote_module():
    """Return the loaded remote backbone module.

    the one that DEFINES ``KimiSparseMoeBlock`` (the Kimi MoE block this plugin patches) -- or None. Name-independent;
    see ``_find_module_defining``.
    """
    return _find_module_defining("KimiSparseMoeBlock")


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
        def __init__(self, config):
            super().__init__()
            self.num_experts = config.num_experts
            # experts run in the LATENT space when use_latent_moe (routed_expert_hidden_size), else hidden.
            self.hidden_dim = (
                config.routed_expert_hidden_size
                if getattr(config, "use_latent_moe", False)
                else config.hidden_size
            )
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
                mask = nn.functional.one_hot(top_k_index, num_classes=self.num_experts).permute(
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
    Block = kmod.KimiSparseMoeBlock  # noqa: N806 (a class, keeping its own name)
    if getattr(Block, "_kimi_k3_ep_patched", False):
        return
    FusedExperts = _build_fused_experts_class(kmod)  # noqa: N806 (a class, keeping its own name)

    _orig_init = Block.__init__

    def _init(self, config):
        _orig_init(self, config)
        # Replace the ModuleList of per-expert MLPs with the fused grouped-param module (EP-capable).
        del self.experts
        self.experts = FusedExperts(config)

    def _forward(self, hidden_states):
        identity = hidden_states
        orig_shape = hidden_states.shape
        topk_idx, topk_weight = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        if self.use_latent_moe:
            hidden_states = self.routed_expert_down_proj(hidden_states)
        # Fused / a2a experts: returns the top-k-combined output in the latent space.
        y = self.experts(hidden_states, topk_idx, topk_weight)
        if self.use_latent_moe:
            if self.latent_moe_use_norm:
                y = self.routed_expert_norm(y)
            y = self.routed_expert_up_proj(y)
        y = y.view(*orig_shape)
        if self.config.num_shared_experts is not None:
            y = y + self.shared_experts(identity)
        return y

    Block.__init__ = _init
    Block.forward = _forward
    Block._kimi_k3_ep_patched = True
    print(
        "[kimi_k3] patched KimiSparseMoeBlock: fused experts + ExpertsInterface forward", flush=True
    )


def _patch_kda_a_log(kmod) -> None:
    """Fix a stale-HF-port bug in the KDA (linear-attention) layers.

    ``KimiDeltaAttention.__init__`` allocates ``A_log`` as ``empty(num_heads)`` (96), but the checkpoint stores it as
    ``[head_dim]`` (128 = ``linear_attn_config['head_dim']``). ``from_pretrained`` therefore raises a size mismatch on
    ``layers.*.self_attn.A_log`` (and ``ignore_mismatched_sizes=True`` would REINIT the trained decay -> garbage). The
    forward passes ``self.A_log`` straight to FLA's ``chunk_kda`` / ``fused_recurrent_kda`` (co-designed with this
    checkpoint), so the ONLY bug is the allocation shape. Re-allocate ``A_log`` as ``[head_dim]`` right after the
    original __init__ so the trained values load into the right shape; ``torch.empty`` is fine (overwritten by the
    checkpoint on load). Idempotent.
    """
    KDA = getattr(kmod, "KimiDeltaAttention", None)  # noqa: N806 (a class, keeping its own name)
    if KDA is None or getattr(KDA, "_kimi_k3_a_log_patched", False):
        return
    _orig_init = KDA.__init__

    def _init(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)
        a_log = getattr(self, "A_log", None)
        if a_log is not None and a_log.shape[0] != self.head_dim:
            self.A_log = nn.Parameter(
                torch.empty(self.head_dim, dtype=a_log.dtype, device=a_log.device)
            )

    KDA.__init__ = _init
    KDA._kimi_k3_a_log_patched = True
    print(
        "[kimi_k3] patched KimiDeltaAttention.A_log alloc: [num_heads] -> [head_dim] (checkpoint shape)",
        flush=True,
    )


def _patch_kda_beta_sigmoid(kmod) -> None:
    """Fix the KDA calibration NaN.

    The remote ``KimiDeltaAttention.forward`` passes the RAW gate scalar ``beta = b_proj(hidden_states)`` to FLA's
    ``chunk_kda`` / ``fused_recurrent_kda`` with ``use_beta_sigmoid_in_kernel=True``, expecting the kernel to apply
    ``sigmoid`` (the delta-rule learning rate must be in ``(0,1)``). In fla-core 0.5.x that flag is a **silent no-op**
    for the main recurrence -- the kernel uses the raw beta. Real trained weights produce ``beta`` up to ~8.4, so the
    per-step delta-rule eigenvalue ``|1 - beta*||k||^2|`` exceeds 1 (``beta*||k||^2 ~ 7.7 >> 2``) and the recurrent
    state grows exponentially (~1e11 over 128 tokens -> ``inf`` -> ``o_norm`` turns it into NaN -> ``detected nan values
    in amax`` during calibration). Random-weight smokes never tripped it because their beta is small (``beta*||k||^2 <
    2``). Verified offline on the real layer-1 inputs: pre-applying ``sigmoid(beta)`` and passing
    ``use_beta_sigmoid_in_kernel=False`` bounds the output (o~0.1, state~4.6, matching the pure-torch
    ``naive_recurrent_kda`` reference), vs 8e9 unfixed.

    Fix: wrap the module-level ``chunk_kda`` / ``fused_recurrent_kda`` in the remote module; when invoked
    with ``use_beta_sigmoid_in_kernel=True``, replace ``beta`` with ``beta.sigmoid()`` and set the flag
    False so the transform happens (correctly) in torch. Idempotent.
    """
    import inspect

    import torch

    if getattr(kmod, "_kimi_k3_beta_sigmoid_patched", False):
        return
    patched_any = False
    for fname in ("chunk_kda", "fused_recurrent_kda"):
        orig = getattr(kmod, fname, None)
        if orig is None or getattr(orig, "_kimi_k3_beta_wrapped", False):
            continue

        def _squash(arguments):
            if arguments.get("use_beta_sigmoid_in_kernel"):
                beta = arguments.get("beta")
                if isinstance(beta, torch.Tensor):
                    arguments["beta"] = torch.sigmoid(beta.float()).to(beta.dtype)
                    arguments["use_beta_sigmoid_in_kernel"] = False

        def _make(orig_fn):
            # Bind to the kernel's signature so beta and the flag are found however they are passed.
            try:
                signature = inspect.signature(orig_fn)
            except (TypeError, ValueError):
                signature = None

            def _wrapped(*args, **kwargs):
                try:
                    bound = signature.bind(*args, **kwargs)  # type: ignore[union-attr]
                except (AttributeError, TypeError):  # no signature, or a call it rejects
                    _squash(kwargs)
                    return orig_fn(*args, **kwargs)
                bound.apply_defaults()
                _squash(bound.arguments)
                return orig_fn(*bound.args, **bound.kwargs)

            setattr(_wrapped, "_kimi_k3_beta_wrapped", True)
            return _wrapped

        setattr(kmod, fname, _make(orig))
        patched_any = True
    kmod._kimi_k3_beta_sigmoid_patched = True
    if patched_any:
        print(
            "[kimi_k3] patched chunk_kda/fused_recurrent_kda: apply sigmoid(beta) in torch "
            "(fla-core use_beta_sigmoid_in_kernel is a no-op -> KDA state explodes on real weights)",
            flush=True,
        )


def _shim_create_causal_mask(kmod) -> None:
    """tf-version compat for the backbone forward's ``create_causal_mask`` call.

    The remote code (written for an older transformers) calls it as ``create_causal_mask(config=, input_embeds=,
    attention_mask=, cache_position=, past_key_values=, position_ids=)``, but tf 5.13's ``create_causal_mask`` renamed
    ``input_embeds`` -> ``inputs_embeds`` and no longer accepts ``cache_position`` -> ``TypeError`` at the first
    calibration forward. The remote module binds ``create_causal_mask`` at import (``from transformers.masking_utils
    import ...``), so wrap that module-level reference: translate ``input_embeds`` -> ``inputs_embeds`` and drop any
    kwargs the installed signature does not accept (robust to further drift). Idempotent.
    """
    import functools
    import inspect

    orig = getattr(kmod, "create_causal_mask", None)
    if orig is None or getattr(orig, "_kimi_k3_ccm_shim", False):
        return
    try:
        accepted = set(inspect.signature(orig).parameters)
    except (TypeError, ValueError):
        accepted = None

    @functools.wraps(orig)
    def _wrapped(*args, **kwargs):
        if "input_embeds" in kwargs and "inputs_embeds" not in kwargs:
            kwargs["inputs_embeds"] = kwargs.pop("input_embeds")
        if accepted is not None:
            kwargs = {k: v for k, v in kwargs.items() if k in accepted}
        return orig(*args, **kwargs)

    setattr(_wrapped, "_kimi_k3_ccm_shim", True)
    kmod.create_causal_mask = _wrapped
    print(
        "[kimi_k3] shimmed create_causal_mask (input_embeds->inputs_embeds; drop unaccepted kwargs; tf>=5.13)",
        flush=True,
    )


def _patch_kimi_cache_get_mask_sizes(kmod) -> None:
    """tf-version compat for the KDA hybrid cache.

    tf 5.13's mask preprocessing calls ``cache.get_mask_sizes(q_length: int, layer_idx)`` (q_length =
    inputs_embeds.shape[1], an int), but the remote ``KimiDynamicCache.get_mask_sizes(cache_position, layer_idx)`` does
    ``cache_position.shape[0]`` -- it expects a TENSOR (older tf) -> ``AttributeError: 'int' object has no attribute
    'shape'`` in the calibration forward. Re-implement to accept either an int query_length or a tensor cache_position
    (the body is otherwise trivial: kv_length = query_length + past_seen_tokens, kv_offset = 0). Idempotent.
    """
    Cache = getattr(kmod, "KimiDynamicCache", None)  # noqa: N806 (a class, keeping its own name)
    if Cache is None or getattr(Cache, "_kimi_k3_gms_patched", False):
        return

    def _get_mask_sizes(self, query_length, layer_idx):
        ql = query_length.shape[0] if torch.is_tensor(query_length) else int(query_length)
        past_seen_tokens = self.get_seq_length(layer_idx)
        return ql + past_seen_tokens, 0

    Cache.get_mask_sizes = _get_mask_sizes
    Cache._kimi_k3_gms_patched = True
    print(
        "[kimi_k3] patched KimiDynamicCache.get_mask_sizes (accept int query_length; tf>=5.13)",
        flush=True,
    )


def _shim_tie_weights_kwargs() -> None:
    """Absorb the ``recompute_mapping`` kwarg newer transformers passes to ``tie_weights``.

    Transformers 5.13's ``from_pretrained`` calls ``model.tie_weights(recompute_mapping=...)``, but
    Kimi-K3's remote ``KimiK3ForConditionalGeneration.tie_weights(self)`` has the older no-kwarg signature
    (it just delegates to ``self.language_model.tie_weights()``) -> ``TypeError``. Wrap it to swallow the
    extra kwargs (they only concern lm_head/embeddings tying, which the delegate handles). Idempotent.
    (Same class of fix as kimi_vl.)

    Locate the VLM wrapper by the class it DEFINES (``KimiK3ForConditionalGeneration``) via
    ``_find_module_defining`` -- name-independent and safe against the ``torch.classes`` magic module
    (see that helper). The LM ``KimiLinearForCausalLM.tie_weights`` is the native ``PreTrainedModel`` one
    (accepts the kwargs) so it needs no wrap.
    """

    def _make_tie(orig):
        def _tie(self, *args, **kwargs):
            return orig(self)

        setattr(_tie, "_kimi_k3_kwargs_shim", True)
        return _tie

    mod = _find_module_defining("KimiK3ForConditionalGeneration")
    if mod is None:
        return
    cls = getattr(mod, "KimiK3ForConditionalGeneration", None)
    if cls is None:
        return
    tw = cls.__dict__.get("tie_weights")  # only if defined ON the class
    if tw is not None and not getattr(tw, "_kimi_k3_kwargs_shim", False):
        cls.tie_weights = _make_tie(tw)


def _force_load_remote_modeling(hf_config, ckpt_path: str, trust_remote_code: bool) -> None:
    """Import the remote modeling now, so its classes can be patched before the model is built.

    Force the remote modeling into ``sys.modules`` (so we can monkey-patch its classes BEFORE
    ``from_pretrained`` instantiates the model) by resolving the model class the checkpoint's config
    declares in its ``auto_map`` -- no hardcoded remote file/module name. Runs the OutputRecorder + closure
    shims first.
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
    get_class_from_dynamic_module(class_ref, ckpt_path, trust_remote_code=trust_remote_code)


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
    try:
        from transformers.conversion_mapping import register_checkpoint_conversion_mapping
        from transformers.core_model_loading import Concatenate, MergeModulelist, WeightConverter
    except Exception as e:  # pragma: no cover - version guard
        print(
            f"[kimi_k3] WARN: transformers conversion API unavailable ({type(e).__name__}: {e}); "
            "per-expert fusion NOT registered -- experts will load as MISSING",
            flush=True,
        )
        return

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
    print(
        "[kimi_k3] registered per-expert w1/w3/w2 -> gate_up_proj/down_proj fusion for kimi_linear "
        "(custom-code conversion; tf>=5.13 skips these otherwise)",
        flush=True,
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
    converters and the decompressor. Returns the (unchanged) config.

    The 896 ``mxfp4-pack-quantized`` routed experts are DECOMPRESSED at load: ``_onfly_decompress``
    dequantizes each local expert's packed weight -> bf16 INSIDE the fused-expert conversion, stored as
    bf16 fused params, and the compressed-tensors ``quantization_config`` is dropped. ModelOpt then
    inserts/calibrates/folds the fp8/nvfp4 quantizers on those bf16 params. That is ~4x the packed HBM
    (the memory bottleneck), but PTQ needs the bf16 weights.
    """
    global _PATCHED
    from modelopt.torch.utils.moe import require_transformers

    require_transformers("5.8", "kimi_k3 expert-parallel PTQ")

    text_cfg = getattr(hf_config, "text_config", None) or hf_config

    # Monkey-patch the remote modeling (needs ckpt_path to force-load it). If ckpt_path is missing we can
    # only set the config plan; the load would then fall back to the non-EP ModuleList layout -> callers
    # must thread ckpt_path through apply_distributed_support.
    if ckpt_path is not None and not _PATCHED:
        _force_load_remote_modeling(hf_config, ckpt_path, trust_remote_code)
        kmod = _find_remote_module()
        if kmod is None:
            raise RuntimeError(
                "kimi_k3: could not locate the loaded remote module defining KimiSparseMoeBlock to patch"
            )
        _patch_kimi_moe_block(kmod)
        _patch_kda_a_log(kmod)
        _patch_kda_beta_sigmoid(kmod)
        _shim_create_causal_mask(kmod)
        _patch_kimi_cache_get_mask_sizes(kmod)
        _shim_tie_weights_kwargs()
        _PATCHED = True

    if not getattr(text_cfg, "base_model_ep_plan", None):
        # grouped_gemm shards the bf16 fused params; the experts container goes through moe_tp_experts
        # (-> torch_a2a_experts).
        text_cfg.base_model_ep_plan = dict(BASE_MODEL_EP_PLAN)
    text_cfg._experts_implementation = "torch_a2a"

    # The 896 routed experts are compressed-tensors `mxfp4-pack-quantized` (4-bit, group 32); everything
    # else (attention, shared experts, latent projections, dense layer, embeddings, lm_head, vision) is
    # plain bf16. Register the per-expert w1/w3/w2 -> gate_up_proj/down_proj fusion converters (tf 5.13
    # skips these for custom-code models) BEFORE the decompressor patches get_model_conversion_mapping, so
    # the decompressor has converters to rewrite into compressed-aware (weight_packed -> bf16) ones. Drop
    # quantization_config afterwards so transformers builds no competing HF quantizer.
    _register_kimi_k3_conversions()
    # ValueError == no compressed-tensors quantization_config, i.e. plain bf16 experts and nothing to
    # decompress.
    with contextlib.suppress(ValueError):
        _onfly_decompress.install_onfly_decompress(hf_config)
    for cfg in (hf_config, text_cfg):
        if getattr(cfg, "quantization_config", None) is not None:
            cfg.quantization_config = None
    return hf_config
