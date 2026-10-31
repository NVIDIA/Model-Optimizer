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

"""DP=EP a2a support for Kimi-K2.5 (``kimi_k25``).

a remote-code VLM whose text backbone is a DeepSeek-V3 variant with compressed-tensors-compressed experts.
Kimi-K2.6 and Kimi-K2.7-Code ship the same model type and modeling classes.

The challenge: Kimi's bundled ``modeling_deepseek.py`` defines ``DeepseekV3MoE`` as an
``nn.ModuleList`` of per-expert ``DeepseekV3MLP`` (separate ``gate_proj`` / ``up_proj`` /
``down_proj`` Linears per expert). That layout is incompatible with transformers' EP machinery,
which expects the **fused** 3-D ``gate_up_proj`` / ``down_proj`` Parameters that the
transformers-native ``deepseek_v3`` modeling ships. We need the native modeling to get EP-capable
experts, but we also want the rest of the VLM (vision tower + mm projector) to come from Kimi's
custom code.

Solution -- monkey-patch ``modeling_kimi_k25``'s ``DeepseekV3ForCausalLM`` import to point at the
transformers-native class, AND swap ``hf_config.text_config`` to a transformers-native
``DeepseekV3Config`` (the native class won't accept Kimi's custom config subclass). Then
``KimiK25ForConditionalGeneration.__init__`` (which does ``self.language_model =
DeepseekV3ForCausalLM(config.text_config)``) builds the native EP-capable model for the language
sub-model, while vision_tower + mm_projector are still built from Kimi's custom code.

On-the-fly compressed-tensors expert decompress happens during from_pretrained's
``WeightConverter`` machinery (installed by ``_onfly_decompress.install_onfly_decompress``); the
fused per-expert source ``.weight`` tensors are decompressed + merged into the 3-D
``gate_up_proj`` / ``down_proj`` parameters. Format-agnostic (compressed_tensors decides the
dequant from the checkpoint's quantization_config).

EP plan: attached to the (now native) ``text_config``. transformers' ``post_init`` aggregates
child-module plans up to the root with name prefix, so the final root ``_ep_plan`` keys end up as
``language_model.model.layers.*.mlp.experts.{gate_up_proj,down_proj,...}`` -- matching the actual
parameter paths in the loaded VLM.

Version floor: transformers >= 5.8, which the spec declares. The on-the-fly decompressor, installed
for a compressed-tensors checkpoint, needs it: earlier releases' ``Concatenate`` trips over the
sibling keys it has already consumed. The decompressor couples tightly to
transformers' still-evolving ``WeightConverter`` conversion-ops API.
"""

from .. import _kimi, _onfly_decompress

# The checkpoint formats prepare has the experts decompress as they load; the spec states them for
# the plan decision, which runs before prepare.
DECOMPRESSED_FORMATS = ("compressed-tensors",)


def make_buildable(
    hf_config, *, ckpt_path: str | None = None, trust_remote_code: bool = False, **_
) -> None:
    """Have the checkpoint's own code import and build on this transformers.

    The plan decision needs it before ``prepare``; it changes nothing the model computes or loads.
    """
    _kimi.make_buildable(
        "modeling_kimi_k25",
        "KimiK25ForConditionalGeneration",
        hf_config,
        ckpt_path,
        trust_remote_code,
    )


def prepare(hf_config, *, ckpt_path: str | None = None, trust_remote_code: bool = False, **_):
    """Load Kimi-K2.5's language model as the transformers-native deepseek_v3.

    Swap text_config to native deepseek_v3 + monkey-patch Kimi's modeling to use the native
    DeepseekV3ForCausalLM + apply the EP plan + install the on-the-fly decompressor for
    compressed-tensors experts.
    """
    # Not at module scope: the model registry imports this module, and modelopt.torch.utils.moe
    # imports the registry.
    from modelopt.torch.utils.moe import apply_distributed_support

    ckpt_path = _kimi.checkpoint_to_patch("kimi_k25", hf_config, ckpt_path, trust_remote_code)

    # 1. Install on-the-fly compressed-tensors expert decompressor. Operates on transformers-native
    #    deepseek_v3's per-expert WeightConverter (which fuses per-expert .weight tensors into the
    #    3-D gate_up_proj/down_proj Parameters). Format-agnostic; driven by checkpoint's
    #    quantization_config, which Kimi-K2.5 carries only on the text_config step 2 replaces. It
    #    goes first, so a checkpoint it refuses leaves everything as it was. Experts stored
    #    otherwise are left to the expert-parallel load's format handling, which reads their scheme
    #    from the copy of quantization_config Kimi's config class keeps on the top-level config;
    #    step 2 leaves that copy alone. A checkpoint without that copy is refused.
    quantization_config = _onfly_decompress._find_quantization_config(hf_config) or {}
    if quantization_config.get("quant_method") in DECOMPRESSED_FORMATS:
        _onfly_decompress.install_onfly_decompress(hf_config)
    else:
        _kimi.refuse_quantization_only_on_text_config("kimi_k25", hf_config)

    # Before any remote-code load: this plugin's get_class_from_dynamic_module below, and
    # from_pretrained later in load_and_prepare_ep.
    _kimi.make_importable()  # the swap below loads the code and adapts its tie_weights

    # 2. Replace text_config with a transformers-native DeepseekV3Config built from the same fields.
    #    Kimi's __init__ passes config.text_config to DeepseekV3ForCausalLM; the native class rejects
    #    Kimi's custom config subclass. to_native_backbone_config also drops auto_map +
    #    quantization_config so transformers builds the native modeling instead of re-routing back
    #    to Kimi's remote code via auto_map.
    hf_config.text_config = _onfly_decompress.to_native_backbone_config(hf_config)

    # 3. Apply the EP plan to the now-native text_config. transformers' post_init walks children
    #    and prefixes their _ep_plan onto the parent's: text_config's plan ends up at the VLM root
    #    as `language_model.model.layers.*.mlp.experts.{gate_up_proj,down_proj}`.
    hf_config.text_config = apply_distributed_support("deepseek_v3", hf_config.text_config)

    # 4. Force-load Kimi's modeling_kimi_k25 module (so we can patch its DeepseekV3ForCausalLM
    #    symbol) and swap to the native class.
    _kimi.patch_language_model_to_native(
        "kimi_k25",
        "modeling_kimi_k25",
        "KimiK25ForConditionalGeneration",
        ckpt_path,
        trust_remote_code,
    )
    return hf_config
