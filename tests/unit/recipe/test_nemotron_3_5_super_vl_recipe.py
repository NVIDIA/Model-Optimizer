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

"""Effective quantizer selection for ``models/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16/ptq/nvfp4-mse``.

The recipe's ``quant_cfg`` is ordered and last-match-wins, so what a module ends up with
depends on rule order as much as on rule content. The KV asymmetry in particular is load
bearing: the backbone casts at a fixed FP8 range and exports no ``k_scale``/``v_scale``, while
the MTP block keeps a calibrated pair. A reordering or a widened glob would silently flip that,
and nothing else in the tree-wide recipe tests would notice.

The module tree below is synthetic: the real checkpoint is 120B, and what is under test is
name matching, not numerics.
"""

import pytest
import torch.nn as nn

import modelopt.torch.quantization as mtq
from modelopt.recipe import load_recipe
from modelopt.torch.quantization.nn import QuantModule, QuantModuleRegistry, TensorQuantizer

_RECIPE = "models/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16/ptq/nvfp4-mse"
_WIDTH = 32


class _Attention(nn.Module):
    """A NemotronH attention mixer.

    ``relative`` adds the Parakeet audio encoder's extra projection: ``*[qkvo]_proj`` matches
    ``relative_k_proj`` too, on the ``k``, so narrowing the attention globs is not a way out.
    """

    def __init__(self, relative: bool = False):
        super().__init__()
        self.q_proj = nn.Linear(_WIDTH, _WIDTH, bias=False)
        self.k_proj = nn.Linear(_WIDTH, _WIDTH, bias=False)
        self.v_proj = nn.Linear(_WIDTH, _WIDTH, bias=False)
        self.o_proj = nn.Linear(_WIDTH, _WIDTH, bias=False)
        if relative:
            self.relative_k_proj = nn.Linear(_WIDTH, _WIDTH, bias=False)


class _QuantAttention(QuantModule):
    """The KV quantizers ``register_hf_attentions_on_the_fly`` installs on a real checkpoint.

    They have to arrive through the registry rather than be set in ``_Attention.__init__``:
    a model that already holds a ``TensorQuantizer`` is treated as quantized, and
    ``mtq.quantize`` would then skip the module conversion that creates the weight quantizers.
    """

    def _setup(self):
        self.k_bmm_quantizer = TensorQuantizer()
        self.v_bmm_quantizer = TensorQuantizer()


class _CoreAttention(nn.Module):
    """Megatron-Core's ``self_attention.core_attention``, the module that holds the KV quantizers."""


class _QuantCoreAttention(_QuantAttention):
    pass


@pytest.fixture(autouse=True)
def _register_kv_attention():
    mtq.register(original_cls=_Attention, quantized_cls=_QuantAttention)
    mtq.register(original_cls=_CoreAttention, quantized_cls=_QuantCoreAttention)
    yield
    for cls in (_Attention, _CoreAttention):
        if QuantModuleRegistry.get(cls) is not None:
            mtq.unregister(cls)


class _Mamba(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_proj = nn.Linear(_WIDTH, 2 * _WIDTH, bias=False)
        self.out_proj = nn.Linear(_WIDTH, _WIDTH, bias=False)


class _Expert(nn.Module):
    def __init__(self):
        super().__init__()
        self.up_proj = nn.Linear(_WIDTH, _WIDTH, bias=False)
        self.down_proj = nn.Linear(_WIDTH, _WIDTH, bias=False)


class _MoE(nn.Module):
    def __init__(self):
        super().__init__()
        self.experts = nn.ModuleList([_Expert(), _Expert()])
        self.shared_experts = _Expert()
        self.router = nn.Linear(_WIDTH, 2, bias=False)


class _Block(nn.Module):
    def __init__(self, mixer):
        super().__init__()
        self.mixer = mixer


def _nemotron_h(with_mtp: bool) -> nn.Module:
    """Hub-named NemotronH skeleton: attention, Mamba and MoE blocks, vision and audio
    encoders, MTP tail.
    """
    model = nn.Module()
    model.backbone = nn.Module()
    model.backbone.embeddings = nn.Linear(_WIDTH, _WIDTH, bias=False)
    model.backbone.layers = nn.ModuleList([_Block(_Attention()), _Block(_Mamba()), _Block(_MoE())])
    model.lm_head = nn.Linear(_WIDTH, _WIDTH, bias=False)
    # VL wrapper pieces that must stay in BF16. Both encoders carry an attention block, not
    # just projections: their attention classes end in `Attention`, so a real run registers
    # them for KV-cache quantization and `*[kv]_bmm_quantizer` reaches them. The audio branch
    # exists whenever the omni config carries a `sound_config`, and image calibration never
    # exercises it, so anything enabled there exports without an activation scale.
    model.embed_vision = nn.Linear(_WIDTH, _WIDTH, bias=False)
    model.vision_model = nn.Module()
    model.vision_model.radio_model = nn.Module()
    model.vision_model.radio_model.blocks = nn.ModuleList([_Expert(), _Block(_Attention())])
    model.sound_projector = nn.Module()
    model.sound_projector.sound_encoder = nn.Module()
    model.sound_projector.sound_encoder.layers = nn.ModuleList([_Block(_Attention(relative=True))])
    if with_mtp:
        # block 0 carries the attention (hence the KV pair), block 1 the MoE
        model.mtp = nn.Module()
        model.mtp.layers = nn.ModuleList([_Block(_Attention()), _Block(_MoE())])
    return model


def _mcore_attention() -> nn.Module:
    layer = nn.Module()
    layer.self_attention = nn.Module()
    layer.self_attention.linear_qkv = nn.Linear(_WIDTH, 3 * _WIDTH, bias=False)
    layer.self_attention.linear_proj = nn.Linear(_WIDTH, _WIDTH, bias=False)
    layer.self_attention.core_attention = _CoreAttention()
    return layer


def _mcore_moe() -> nn.Module:
    layer = nn.Module()
    layer.mlp = nn.Module()
    layer.mlp.router = nn.Linear(_WIDTH, 2, bias=False)
    for name in ("experts", "shared_experts"):
        mlp = nn.Module()
        mlp.linear_fc1 = nn.Linear(_WIDTH, _WIDTH, bias=False)
        mlp.linear_fc2 = nn.Linear(_WIDTH, _WIDTH, bias=False)
        setattr(layer.mlp, name, mlp)
    return layer


def _nemotron_h_mcore() -> nn.Module:
    """Megatron-Core-named omni VL skeleton, as built by Megatron-Bridge.

    Names follow a real PTQ checkpoint: the language model sits under ``language_model``,
    attention is ``self_attention.linear_qkv`` / ``linear_proj``, lm_head is ``output_layer``,
    and the MTP block nests its attention and MoE layers under ``mtp_model_layer.layers``. The
    projector is ``vision_projection``. Megatron-Bridge builds no audio branch, but one is
    included so the test pins the audio guards against the Megatron-Core attention rules too.
    """
    model = nn.Module()
    lm = model.language_model = nn.Module()
    lm.embedding = nn.Linear(_WIDTH, _WIDTH, bias=False)
    lm.decoder = nn.Module()
    lm.decoder.layers = nn.ModuleList([_mcore_attention(), _Block(_Mamba()), _mcore_moe()])
    lm.output_layer = nn.Linear(_WIDTH, _WIDTH, bias=False)
    mtp_block = nn.Module()
    mtp_block.eh_proj = nn.Linear(2 * _WIDTH, _WIDTH, bias=False)
    mtp_block.mtp_model_layer = nn.Module()
    mtp_block.mtp_model_layer.layers = nn.ModuleList([_mcore_attention(), _mcore_moe()])
    lm.mtp = nn.Module()
    lm.mtp.layers = nn.ModuleList([mtp_block])
    model.vision_model = nn.Module()
    model.vision_model.blocks = nn.ModuleList([_Expert(), _mcore_attention()])
    # An attention layer stands in under the projector to prove its guard.
    model.vision_projection = nn.ModuleList([_mcore_attention()])
    model.sound_projector = nn.Module()
    model.sound_projector.sound_encoder = nn.Module()
    model.sound_projector.sound_encoder.layers = nn.ModuleList([_mcore_attention()])
    return model


def _quantize(with_mtp: bool, mcore: bool = False):
    """Apply the recipe to a fresh skeleton and return it with its named modules."""
    model = _nemotron_h_mcore() if mcore else _nemotron_h(with_mtp)
    config = load_recipe(_RECIPE).quantize.model_dump()

    # Calibration settings the recipe promises, asserted before they are dropped: weight-MSE
    # with the FP8 scale sweep, and layerwise calibration off (VL decoder layers sit under
    # `model.language_model.layers`, where layerwise_calibrate cannot find them).
    assert config["algorithm"]["method"] == "mse"
    assert config["algorithm"]["fp8_scale_sweep"] is True
    assert config["algorithm"]["layerwise"]["enable"] is False

    # No forward loop here; selection and precedence are what is under test.
    config["algorithm"] = None
    mtq.quantize(model, config)
    return model, dict(model.named_modules())


def _assert_fp8(quantizer, name):
    """Assert ``quantizer`` is enabled per-tensor FP8 E4M3."""
    assert quantizer.is_enabled, f"{name} should be enabled"
    assert quantizer.num_bits == (4, 3), f"{name} should be FP8 E4M3"
    assert quantizer.block_sizes is None, f"{name} should be per-tensor"


def _assert_nvfp4(quantizer, name, *, block_type):
    """Assert ``quantizer`` is enabled NVFP4 E2M1 over 16-element blocks with FP8 scales."""
    assert quantizer.is_enabled, f"{name} should be enabled"
    assert quantizer.num_bits == (2, 1), f"{name} should be NVFP4 E2M1"
    assert quantizer.block_sizes[-1] == 16, f"{name} should use 16-element blocks"
    assert quantizer.block_sizes["type"] == block_type, f"{name} should be {block_type}"
    assert quantizer.block_sizes["scale_bits"] == (4, 3), f"{name} should carry FP8 scales"


@pytest.mark.parametrize("with_mtp", [True, False])
def test_backbone_kv_casts_at_a_fixed_range_and_exports_no_scale(with_mtp):
    """`use_constant_amax` pins amax to the FP8 E4M3 max and registers no `_amax` buffer, so the
    export carries no backbone `k_scale`/`v_scale`. The shipped checkpoint has exactly one of
    each, both from MTP; adding backbone scales here would break that match.
    """
    _, modules = _quantize(with_mtp)

    for which in ("k", "v"):
        name = f"backbone.layers.0.mixer.{which}_bmm_quantizer"
        quantizer = modules[name]
        _assert_fp8(quantizer, name)
        assert quantizer._use_constant_amax is True, f"{name} should cast at a fixed range"
        assert not hasattr(quantizer, "_amax"), f"{name} must register no exportable scale"


def test_mtp_kv_stays_calibrated():
    """The MTP rule restates `use_constant_amax: false` so it cannot inherit the backbone rule."""
    _, modules = _quantize(with_mtp=True)

    for which in ("k", "v"):
        name = f"mtp.layers.0.mixer.{which}_bmm_quantizer"
        quantizer = modules[name]
        _assert_fp8(quantizer, name)
        assert quantizer._use_constant_amax is False, f"{name} should stay calibrated"


@pytest.mark.parametrize("with_mtp", [True, False])
def test_backbone_expert_and_projection_formats(with_mtp):
    """Experts are NVFP4 (static weights, dynamic inputs); projections and lm_head are FP8."""
    _, modules = _quantize(with_mtp)

    for expert in ("experts.0", "experts.1", "shared_experts"):
        for proj in ("up_proj", "down_proj"):
            stem = f"backbone.layers.2.mixer.{expert}.{proj}"
            _assert_nvfp4(modules[f"{stem}.weight_quantizer"], stem, block_type="static")
            _assert_nvfp4(modules[f"{stem}.input_quantizer"], stem, block_type="dynamic")

    for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
        stem = f"backbone.layers.0.mixer.{proj}"
        _assert_fp8(modules[f"{stem}.weight_quantizer"], stem)
        _assert_fp8(modules[f"{stem}.input_quantizer"], stem)

    for proj in ("in_proj", "out_proj"):
        stem = f"backbone.layers.1.mixer.{proj}"
        _assert_fp8(modules[f"{stem}.weight_quantizer"], stem)
        _assert_fp8(modules[f"{stem}.input_quantizer"], stem)

    _assert_fp8(modules["lm_head.weight_quantizer"], "lm_head")
    _assert_fp8(modules["lm_head.input_quantizer"], "lm_head")


def test_mtp_block_one_experts_match_the_backbone_formats():
    """The MTP experts are NVFP4 with static weight scales and dynamic inputs, like block 2.

    `*mtp.*` disables the whole tail first, so only the rules that follow it bring anything
    back: the block-1 experts and the block-0 KV pair. The MTP attention projections stay BF16.
    """
    _, modules = _quantize(with_mtp=True)

    for expert in ("experts.0", "experts.1", "shared_experts"):
        for proj in ("up_proj", "down_proj"):
            stem = f"mtp.layers.1.mixer.{expert}.{proj}"
            _assert_nvfp4(modules[f"{stem}.weight_quantizer"], stem, block_type="static")
            _assert_nvfp4(modules[f"{stem}.input_quantizer"], stem, block_type="dynamic")

    for proj in ("q_proj", "o_proj"):
        for kind in ("weight_quantizer", "input_quantizer"):
            name = f"mtp.layers.0.mixer.{proj}.{kind}"
            assert not modules[name].is_enabled, f"{name} should stay BF16"
    assert not modules["mtp.layers.1.mixer.router.weight_quantizer"].is_enabled


@pytest.mark.parametrize("with_mtp", [True, False])
def test_vision_audio_router_and_embeddings_stay_bf16(with_mtp):
    """Everything the blanket `*` disable is meant to leave alone stays unquantized.

    The encoder attention entries are the load-bearing ones: they are named by the FP8
    attention and KV rules earlier in the file and only the trailing disables turn them back
    off. A constant-amax KV cast exports no scale either way, so no artefact check would
    catch it being left on.
    """
    _, modules = _quantize(with_mtp)

    disabled = [
        "backbone.embeddings.weight_quantizer",
        "backbone.layers.2.mixer.router.weight_quantizer",
        "embed_vision.weight_quantizer",
        "embed_vision.input_quantizer",
        "vision_model.radio_model.blocks.0.up_proj.weight_quantizer",
        "vision_model.radio_model.blocks.0.up_proj.input_quantizer",
        "vision_model.radio_model.blocks.0.down_proj.weight_quantizer",
        "vision_model.radio_model.blocks.1.mixer.q_proj.weight_quantizer",
        "vision_model.radio_model.blocks.1.mixer.k_bmm_quantizer",
        "sound_projector.sound_encoder.layers.0.mixer.q_proj.weight_quantizer",
        "sound_projector.sound_encoder.layers.0.mixer.o_proj.input_quantizer",
        "sound_projector.sound_encoder.layers.0.mixer.relative_k_proj.weight_quantizer",
        "sound_projector.sound_encoder.layers.0.mixer.v_bmm_quantizer",
    ]
    for name in disabled:
        assert not modules[name].is_enabled, f"{name} should stay BF16"


def test_mtp_rules_are_inert_without_an_mtp_tail():
    """The recipe is shipped for checkpoints with and without MTP; the tail rules must not
    change what the backbone gets.
    """
    _, with_mtp = _quantize(with_mtp=True)
    _, without_mtp = _quantize(with_mtp=False)

    def summary(modules):
        return {
            name: (module.is_enabled, module.num_bits, module.block_sizes)
            for name, module in modules.items()
            if isinstance(module, TensorQuantizer) and not name.startswith("mtp.")
        }

    assert summary(with_mtp) == summary(without_mtp)
    assert not any(name.startswith("mtp.") for name in without_mtp)


def test_no_quantizer_outside_the_intended_set_is_enabled():
    """A widened glob would show up here first: the blanket `*` disable comes first, so every
    enabled quantizer has to be named by a later rule.
    """
    _, modules = _quantize(with_mtp=True)

    enabled = {
        name
        for name, module in modules.items()
        if isinstance(module, TensorQuantizer) and module.is_enabled
    }
    expected = set()
    for which in ("k", "v"):
        expected |= {
            f"backbone.layers.0.mixer.{which}_bmm_quantizer",
            f"mtp.layers.0.mixer.{which}_bmm_quantizer",
        }
    for kind in ("weight_quantizer", "input_quantizer"):
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            expected.add(f"backbone.layers.0.mixer.{proj}.{kind}")
        for proj in ("in_proj", "out_proj"):
            expected.add(f"backbone.layers.1.mixer.{proj}.{kind}")
        for prefix in ("backbone.layers.2", "mtp.layers.1"):
            for expert in ("experts.0", "experts.1", "shared_experts"):
                for proj in ("up_proj", "down_proj"):
                    expected.add(f"{prefix}.mixer.{expert}.{proj}.{kind}")
        expected.add(f"lm_head.{kind}")

    assert enabled == expected


def test_megatron_core_names_get_the_same_formats():
    """Under Megatron-Bridge the recipe selects the same layers as on a Hub checkpoint.

    Attention, lm_head and the MTP block have Megatron-Core names of their own; without those
    spellings attention and lm_head would silently stay BF16. Everything else is shared.
    """
    _, modules = _quantize(with_mtp=True, mcore=True)
    lm, mtp = "language_model.decoder.layers", "language_model.mtp.layers.0.mtp_model_layer.layers"

    expected = set()
    for kind in ("weight_quantizer", "input_quantizer"):
        for stem in (
            f"{lm}.0.self_attention.linear_qkv",
            f"{lm}.0.self_attention.linear_proj",
            f"{lm}.1.mixer.in_proj",
            f"{lm}.1.mixer.out_proj",
            "language_model.output_layer",
        ):
            _assert_fp8(modules[f"{stem}.{kind}"], stem)
            expected.add(f"{stem}.{kind}")
        for prefix in (f"{lm}.2", f"{mtp}.1"):
            for mlp in ("experts", "shared_experts"):
                for fc in ("linear_fc1", "linear_fc2"):
                    stem = f"{prefix}.mlp.{mlp}.{fc}"
                    block_type = "static" if kind == "weight_quantizer" else "dynamic"
                    _assert_nvfp4(modules[f"{stem}.{kind}"], stem, block_type=block_type)
                    expected.add(f"{stem}.{kind}")

    for which in ("k", "v"):
        backbone = f"{lm}.0.self_attention.core_attention.{which}_bmm_quantizer"
        _assert_fp8(modules[backbone], backbone)
        assert modules[backbone]._use_constant_amax is True, f"{backbone} should cast"
        mtp_kv = f"{mtp}.0.self_attention.core_attention.{which}_bmm_quantizer"
        _assert_fp8(modules[mtp_kv], mtp_kv)
        assert modules[mtp_kv]._use_constant_amax is False, f"{mtp_kv} should stay calibrated"
        expected |= {backbone, mtp_kv}

    enabled = {
        name
        for name, module in modules.items()
        if isinstance(module, TensorQuantizer) and module.is_enabled
    }
    assert enabled == expected
