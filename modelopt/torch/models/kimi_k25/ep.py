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

"""DP=EP a2a support for Kimi-K2.7-Code (``kimi_k25``).

a remote-code VLM whose text backbone is a DeepSeek-V3 variant with compressed-tensors-compressed experts.

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

Version floor: transformers >= 5.8, which the on-the-fly decompressor enforces: earlier releases'
``Concatenate`` trips over the sibling keys it has already consumed. The decompressor couples tightly to
transformers' still-evolving ``WeightConverter`` conversion-ops API.
"""

import sys

from modelopt.torch.utils.moe import apply_distributed_support

from .. import _onfly_decompress
from .._remote_code import stage_full_remote_closure


def prepare(hf_config, *, ckpt_path: str | None = None, trust_remote_code: bool = False, **_):
    """Load Kimi-K2.7's language model as the transformers-native deepseek_v3.

    Swap text_config to native deepseek_v3 + monkey-patch Kimi's modeling to use the native
    DeepseekV3ForCausalLM + apply the EP plan + install the on-the-fly expert decompressor.
    """
    if type(hf_config).__module__.startswith("transformers."):
        # From 5.13, a load without trust_remote_code builds transformers' own port, which the
        # patch below never reaches.
        raise ValueError(
            "kimi_k25: expert parallelism runs on the checkpoint's own modeling code; "
            "load it with trust_remote_code=True."
        )
    if not hasattr(hf_config, "text_config"):
        raise ValueError("kimi_k25.apply: hf_config has no .text_config (expected a VLM config)")

    # transformers stages only 1-hop relative imports; Kimi's remote code is 2 hops deep. Install the
    # full-closure staging shim BEFORE any remote-code load (this plugin's get_class_from_dynamic_module
    # below, and from_pretrained later in load_and_prepare_ep).
    stage_full_remote_closure()

    # 1. Replace text_config with a transformers-native DeepseekV3Config built from the same fields.
    #    Kimi's __init__ passes config.text_config to DeepseekV3ForCausalLM; the native class rejects
    #    Kimi's custom config subclass. to_native_backbone_config also drops auto_map +
    #    quantization_config so transformers builds the native modeling instead of re-routing back
    #    to Kimi's remote code via auto_map.
    hf_config.text_config = _onfly_decompress.to_native_backbone_config(hf_config)

    # 2. Install on-the-fly compressed-tensors expert decompressor. Operates on transformers-native
    #    deepseek_v3's per-expert WeightConverter (which fuses per-expert .weight tensors into the
    #    3-D gate_up_proj/down_proj Parameters). Format-agnostic; driven by checkpoint's
    #    quantization_config.
    _onfly_decompress.install_onfly_decompress(hf_config)

    # 3. Apply the EP plan to the now-native text_config. transformers' post_init walks children
    #    and prefixes their _ep_plan onto the parent's: text_config's plan ends up at the VLM root
    #    as `language_model.model.layers.*.mlp.experts.{gate_up_proj,down_proj}`.
    apply_distributed_support("deepseek_v3", hf_config.text_config)

    # 4. Force-load Kimi's modeling_kimi_k25 module (so we can patch its DeepseekV3ForCausalLM
    #    symbol) and swap to the native class. If ckpt_path isn't available we skip the patch and
    #    the run will fall back to Kimi's per-expert nn.ModuleList layout (non-EP-capable) -- caller
    #    should always thread ckpt_path through apply_distributed_support.
    if ckpt_path is not None:
        _patch_kimi_modeling_to_native(ckpt_path, trust_remote_code)
    return hf_config


def _patch_kimi_modeling_to_native(ckpt_path: str, trust_remote_code: bool) -> None:
    """Point the remote module's ``DeepseekV3ForCausalLM`` at the transformers-native class.

    Load Kimi's remote-code ``modeling_kimi_k25`` module and replace its ``DeepseekV3ForCausalLM``
    symbol with the transformers-native class. After this, ``KimiK25ForConditionalGeneration.__init__``
    builds the EP-capable native deepseek_v3 instead of Kimi's per-expert Linears version.
    """
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    from transformers.models.deepseek_v3 import DeepseekV3ForCausalLM as NativeDeepseekV3ForCausalLM

    # Force the dynamic-module loader to stage + import modeling_kimi_k25 into sys.modules.
    get_class_from_dynamic_module(
        "modeling_kimi_k25.KimiK25ForConditionalGeneration",
        ckpt_path,
        trust_remote_code=trust_remote_code,
    )

    patched = []
    for mod_name, mod in list(sys.modules.items()):
        if mod is None:
            continue
        # The dynamic-module loader names the module like
        # ``transformers_modules.<short-repo>.<hash>.modeling_kimi_k25``.
        if mod_name.endswith(".modeling_kimi_k25") and hasattr(mod, "DeepseekV3ForCausalLM"):
            setattr(mod, "DeepseekV3ForCausalLM", NativeDeepseekV3ForCausalLM)
            patched.append(mod_name)
    if not patched:
        raise RuntimeError(
            "kimi_k25: could not find a loaded modeling_kimi_k25 in sys.modules to monkey-patch"
        )
    print(
        f"[kimi_k25] monkey-patched DeepseekV3ForCausalLM -> transformers-native in: {patched}",
        flush=True,
    )
