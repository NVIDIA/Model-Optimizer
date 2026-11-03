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

"""Expert-parallel preparation for remote-code Kimi-VL (``kimi_vl``).

``prepare`` replaces the per-expert DeepSeek-V3 language model the checkpoint's own code builds with
transformers' native DeepSeek-V3, whose experts the expert-parallel load can shard, and keeps the
vision tower and projector from that code. It needs trusted remote code and transformers 5.7.
"""

from .. import _kimi, _onfly_decompress


def make_buildable(
    hf_config, *, ckpt_path: str | None = None, trust_remote_code: bool = False, **_
) -> None:
    """Have the checkpoint's own code import and build on this transformers.

    The plan decision needs it before ``prepare``; it changes nothing the model computes or loads.
    """
    _kimi.make_buildable(
        "modeling_kimi_vl",
        "KimiVLForConditionalGeneration",
        hf_config,
        ckpt_path,
        trust_remote_code,
    )


def prepare_refusal(
    hf_config, *, ckpt_path: str | None = None, trust_remote_code: bool = False, **_
) -> str | None:
    """Why :func:`prepare` would refuse this checkpoint, or ``None``, found changing nothing."""
    why = _kimi.prepare_refusal("kimi_vl", hf_config, ckpt_path, trust_remote_code)
    if why is None:
        try:
            _kimi.refuse_quantization_only_on_text_config("kimi_vl", hf_config)
        except ValueError as e:
            why = str(e)
    return why


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

    _kimi.make_importable()  # the swap below loads the code and adapts its tie_weights

    # A native deepseek_v3 config, with deepseek_v3's plan, which reaches the VLM root under
    # language_model.*. Its type is what has the swapped name below build the native class;
    # dropping auto_map only keeps it from naming the checkpoint's classes.
    hf_config.text_config = _onfly_decompress.to_native_backbone_config(hf_config)
    hf_config.text_config = apply_distributed_support("deepseek_v3", hf_config.text_config)

    _kimi.patch_language_model_to_native(
        "kimi_vl",
        "modeling_kimi_vl",
        "KimiVLForConditionalGeneration",
        ckpt_path,
        trust_remote_code,
    )
    return hf_config
