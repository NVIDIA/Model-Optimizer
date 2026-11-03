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

"""Expert-parallel preparation for remote-code Kimi-K2.5, K2.6 and K2.7-Code (``kimi_k25``).

``prepare`` replaces the per-expert DeepSeek-V3 language model the checkpoint's own code builds with
transformers' native DeepSeek-V3, whose experts the expert-parallel load can shard, and keeps the
vision tower and projector from that code. A compressed-tensors checkpoint's experts decompress as
they load (``_onfly_decompress``). It needs trusted remote code and transformers 5.8. transformers'
own port (from 5.13) of an unquantized checkpoint builds that native language model itself, with
its plan, so ``prepare`` leaves it as it is.
"""

from .. import _kimi, _onfly_decompress

# The checkpoint formats prepare has the experts decompress as they load; the spec states them for
# the plan decision, which runs before prepare.
DECOMPRESSED_FORMATS = ("compressed-tensors",)


def _decompresses(hf_config) -> bool:
    """Whether :func:`prepare` has this checkpoint's experts decompress as they load."""
    quantization_config = _onfly_decompress._find_quantization_config(hf_config) or {}
    return quantization_config.get("quant_method") in DECOMPRESSED_FORMATS


def _port_with_its_own_plan(hf_config) -> bool:
    """Whether ``hf_config`` is transformers' own port of an unquantized checkpoint with a plan.

    Its text config carries the plan the expert-parallel load shards by, so it loads expert-parallel
    as it is.
    """
    return (
        type(hf_config).__module__.startswith("transformers.")
        and _onfly_decompress._find_quantization_config(hf_config) is None
        and bool(getattr(getattr(hf_config, "text_config", None), "base_model_ep_plan", None))
    )


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


def prepare_refusal(
    hf_config, *, ckpt_path: str | None = None, trust_remote_code: bool = False, **_
) -> str | None:
    """Why :func:`prepare` would refuse this checkpoint, or ``None``, found changing nothing.

    Its own refusals, and the decompressor's for a compressed-tensors checkpoint, without
    installing the decompressor.
    """
    if _port_with_its_own_plan(hf_config):
        return None
    why = _kimi.prepare_refusal("kimi_k25", hf_config, ckpt_path, trust_remote_code)
    if why is not None:
        return why
    if _decompresses(hf_config):
        return _onfly_decompress.decompression_refusal(hf_config)
    try:
        _kimi.refuse_quantization_only_on_text_config("kimi_k25", hf_config)
    except ValueError as e:
        return str(e)
    return None


def prepare(hf_config, *, ckpt_path: str | None = None, trust_remote_code: bool = False, **_):
    """Load Kimi-K2.5's language model as the transformers-native deepseek_v3.

    Installs the decompressor for a compressed-tensors checkpoint, gives ``text_config`` a native
    deepseek_v3 config with deepseek_v3's plan, and has the checkpoint's code build the native
    language model from it. transformers' own port of an unquantized checkpoint, which already
    builds it, is returned as it is.
    """
    if _port_with_its_own_plan(hf_config):
        return hf_config
    # Not at module scope: the model registry imports this module, and modelopt.torch.utils.moe
    # imports the registry.
    from modelopt.torch.utils.moe import apply_distributed_support

    ckpt_path = _kimi.checkpoint_to_patch("kimi_k25", hf_config, ckpt_path, trust_remote_code)

    # First, so a checkpoint the decompressor refuses leaves everything as it was. Its scheme sits
    # on the text_config the native config below replaces. Experts stored otherwise are left to the
    # expert-parallel load's format handling, which reads the copy of quantization_config Kimi's
    # config class keeps on the top-level config; a checkpoint without that copy is refused.
    if _decompresses(hf_config):
        _onfly_decompress.install_onfly_decompress(hf_config)
    else:
        _kimi.refuse_quantization_only_on_text_config("kimi_k25", hf_config)

    # Before any remote-code load: the swap's below, and from_pretrained's in load_and_prepare_ep.
    _kimi.make_importable()  # the swap below loads the code and adapts its tie_weights

    # A native deepseek_v3 config, with deepseek_v3's plan, which reaches the VLM root under
    # language_model.*. Its type is what has the swapped name below build the native class;
    # dropping auto_map only keeps it from naming the checkpoint's classes.
    hf_config.text_config = _onfly_decompress.to_native_backbone_config(hf_config)
    hf_config.text_config = apply_distributed_support("deepseek_v3", hf_config.text_config)

    # The per-config build: the module's DeepseekV3ForCausalLM builds transformers' native class for
    # the native config above, and the checkpoint's own class for any other.
    _kimi.patch_language_model_to_native(
        "kimi_k25",
        "modeling_kimi_k25",
        "KimiK25ForConditionalGeneration",
        ckpt_path,
        trust_remote_code,
    )
    return hf_config
