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

"""DP=EP a2a support for Kimi-VL-A3B (``kimi_vl``).

a remote-code VLM whose text backbone is a Moonlight (DeepSeek-V3-lite: MLA, ``q_lora_rank=None``, 64 routed experts, 2
shared) MoE and whose vision tower is MoonViT. The bf16 sibling of ``kimi_k25`` -- SAME shape, but NO compressed-tensors
decompressor (the checkpoint ships plain bf16 per-expert weights).

Challenge (identical to kimi_k25): the bundled ``modeling_kimi_vl.py`` builds its language model from
an INLINE ``DeepseekV3ForCausalLM`` whose ``DeepseekV3MoE`` is an ``nn.ModuleList`` of per-expert
``gate_proj``/``up_proj``/``down_proj`` Linears -- incompatible with transformers' EP machinery, which
needs the FUSED 3-D ``gate_up_proj``/``down_proj`` Parameters the transformers-native ``deepseek_v3``
modeling ships.

Solution (mirrors kimi_k25):
  1. Shim the import symbols ``modeling_kimi_vl`` needs that newer transformers dropped (e.g.
     ``is_torch_fx_available``) so the module IMPORTS -- they only guard torch.fx / legacy attn-mask
     paths that our native-LM + MoonViT forward never execute.
  2. Swap ``hf_config.text_config`` to a transformers-native ``DeepseekV3Config`` (drops the remote
     ``auto_map`` so the native class -- and its ``base_model_ep_plan`` -- is used for the LM).
  3. Apply the deepseek_v3 EP plan to that native ``text_config`` (post_init prefixes it to the VLM
     root as ``language_model.model.layers.*.mlp.experts.{gate_up_proj,down_proj}``).
  4. Monkey-patch ``modeling_kimi_vl.DeepseekV3ForCausalLM`` -> the transformers-native class, so
     ``KimiVLForConditionalGeneration.__init__`` (``self.language_model = DeepseekV3ForCausalLM(
     config.text_config)``) builds the EP-capable native LM while MoonViT + the KimiVL wrapper stay
     from the bundled code. Native deepseek_v3's per-expert WeightConverter fuses the checkpoint's
     per-expert ``.weight`` tensors into the 3-D params on load (no decompressor -- weights are bf16).

Version: matches Kimi-K2.7 -- validated on transformers 5.9.0 (has the EP framework + the legacy
symbols the remote code imports). The compat shims make it tolerant of a few later drops too.
"""

import sys

from modelopt.torch.utils.moe import apply_distributed_support

from .. import _onfly_decompress
from .._remote_code import stage_full_remote_closure


def _install_compat_import_shims():
    """Add stubs for symbols ``modeling_kimi_vl`` imports that newer transformers removed, so the module imports.

    Each guards a torch.fx / legacy-attn-mask code path the native LM + MoonViT forward never hit, so a benign stub is
    safe. Idempotent (only sets missing attrs).
    """
    try:
        import transformers.utils.import_utils as iu

        if not hasattr(iu, "is_torch_fx_available"):
            iu.is_torch_fx_available = lambda: False
    except Exception as e:
        print(f"[kimi_vl] compat-shim warning (import_utils): {type(e).__name__}: {e}", flush=True)
    try:
        import transformers.pytorch_utils as pu

        if not hasattr(pu, "is_torch_greater_or_equal_than_1_13"):
            pu.is_torch_greater_or_equal_than_1_13 = True
    except Exception as e:
        print(f"[kimi_vl] compat-shim warning (pytorch_utils): {type(e).__name__}: {e}", flush=True)
    try:
        import transformers.modeling_attn_mask_utils as amu

        if not hasattr(amu, "_prepare_4d_causal_attention_mask"):

            def _prepare_4d_causal_attention_mask(*args, **kwargs):  # pragma: no cover
                raise RuntimeError(
                    "kimi_vl: remote _prepare_4d_causal_attention_mask stub called -- the native LM "
                    "builds its own mask, so this path should be unreachable."
                )

            amu._prepare_4d_causal_attention_mask = _prepare_4d_causal_attention_mask
    except Exception as e:
        print(
            f"[kimi_vl] compat-shim warning (attn_mask_utils): {type(e).__name__}: {e}", flush=True
        )


def prepare(hf_config, *, ckpt_path: str | None = None, trust_remote_code: bool = False, **_):
    """Load Kimi-VL's language model as the transformers-native deepseek_v3.

    Swap text_config to native deepseek_v3 + apply the EP plan + monkey-patch Kimi-VL's modeling to
    build the native EP-capable DeepseekV3ForCausalLM for the language sub-model. No decompressor: the
    Kimi-VL checkpoint is plain bf16.
    """
    if not hasattr(hf_config, "text_config"):
        raise ValueError("kimi_vl.apply: hf_config has no .text_config (expected a VLM config)")

    _install_compat_import_shims()
    stage_full_remote_closure()

    # Native deepseek_v3 config for the LM (drops the remote auto_map so the native class is used).
    hf_config.text_config = _onfly_decompress.to_native_backbone_config(hf_config)
    # EP plan onto the native text_config (bubbles up to the VLM root under language_model.*).
    apply_distributed_support("deepseek_v3", hf_config.text_config)

    if ckpt_path is not None:
        _patch_kimi_vl_modeling_to_native(ckpt_path, trust_remote_code)
    return hf_config


def _drop_tie_weights_kwargs(orig_tie):
    """Wrap a remote ``tie_weights()`` that predates the kwargs newer transformers pass to it."""

    def _tie_weights(self, *args, **kwargs):  # drop recompute_mapping etc.
        return orig_tie(self)

    setattr(_tie_weights, "_kimi_vl_kwargs_shim", True)
    return _tie_weights


def _patch_kimi_vl_modeling_to_native(ckpt_path: str, trust_remote_code: bool) -> None:
    """Point the remote module's inline ``DeepseekV3ForCausalLM`` at the native class.

    Load Kimi-VL's remote ``modeling_kimi_vl`` and replace its inline ``DeepseekV3ForCausalLM`` with
    the transformers-native class, so ``KimiVLForConditionalGeneration.__init__`` builds the EP-capable
    native deepseek_v3 for ``self.language_model`` (MoonViT + wrapper stay from the bundled code).
    """
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    from transformers.models.deepseek_v3 import DeepseekV3ForCausalLM as NativeDeepseekV3ForCausalLM

    get_class_from_dynamic_module(
        "modeling_kimi_vl.KimiVLForConditionalGeneration",
        ckpt_path,
        trust_remote_code=trust_remote_code,
    )

    patched = []
    for mod_name, mod in list(sys.modules.items()):
        if mod is None:
            continue
        if mod_name.endswith(".modeling_kimi_vl") and hasattr(mod, "DeepseekV3ForCausalLM"):
            setattr(mod, "DeepseekV3ForCausalLM", NativeDeepseekV3ForCausalLM)
            # transformers 5.9's init_weights calls self.tie_weights(recompute_mapping=False), but
            # Kimi-VL's remote tie_weights() has the older no-kwarg signature -> TypeError. Wrap it to
            # swallow the extra kwargs (it only ties the LM head/embeddings; the kwargs don't apply).
            kimi_cls = getattr(mod, "KimiVLForConditionalGeneration", None)
            if kimi_cls is not None:
                own_tie = vars(kimi_cls).get("tie_weights")
                if own_tie is not None and not getattr(own_tie, "_kimi_vl_kwargs_shim", False):
                    kimi_cls.tie_weights = _drop_tie_weights_kwargs(own_tie)
            patched.append(mod_name)
    if not patched:
        raise RuntimeError(
            "kimi_vl: could not find a loaded modeling_kimi_vl in sys.modules to monkey-patch"
        )
    print(
        f"[kimi_vl] monkey-patched DeepseekV3ForCausalLM -> transformers-native (+tie_weights shim) in: "
        f"{patched}",
        flush=True,
    )
