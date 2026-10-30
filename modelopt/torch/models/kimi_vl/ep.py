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

Version: no floor beyond the EP framework's 5.7; validated on transformers 5.9.0. The compat shims
stand in for the symbols the remote code imports that later releases dropped.
"""

from .. import _kimi, _onfly_decompress
from .._remote_code import stage_full_remote_closure


def prepare(hf_config, *, ckpt_path: str | None = None, trust_remote_code: bool = False, **_):
    """Load Kimi-VL's language model as the transformers-native deepseek_v3.

    Swap text_config to native deepseek_v3 + apply the EP plan + monkey-patch Kimi-VL's modeling to
    build the native EP-capable DeepseekV3ForCausalLM for the language sub-model. No decompressor: the
    Kimi-VL checkpoint is plain bf16.
    """
    # Not at module scope: the model registry imports this module, and modelopt.torch.utils.moe
    # imports the registry.
    from modelopt.torch.utils.moe import apply_distributed_support

    ckpt_path = _kimi.checkpoint_to_patch("kimi_vl", hf_config, ckpt_path, trust_remote_code)
    _kimi.refuse_quantization_only_on_text_config("kimi_vl", hf_config)

    _kimi.install_compat_import_shims()
    stage_full_remote_closure()

    # Native deepseek_v3 config for the LM (drops the remote auto_map so the native class is used).
    hf_config.text_config = _onfly_decompress.to_native_backbone_config(hf_config)
    # EP plan onto the native text_config (bubbles up to the VLM root under language_model.*).
    hf_config.text_config = apply_distributed_support("deepseek_v3", hf_config.text_config)

    _kimi.patch_language_model_to_native(
        "kimi_vl",
        "modeling_kimi_vl",
        "KimiVLForConditionalGeneration",
        ckpt_path,
        trust_remote_code,
    )
    return hf_config
