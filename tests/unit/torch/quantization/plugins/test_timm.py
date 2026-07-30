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

import io
from copy import deepcopy

import pytest
import torch

timm = pytest.importorskip("timm")

import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq
from modelopt.recipe import load_recipe
from modelopt.torch.quantization.algorithms import (
    AutoQuantizeGradientSearcher,
    QuantRecipe,
    QuantRecipeHparam,
)
from modelopt.torch.quantization.plugins.timm import is_resnet_quantization_supported


def _get_resnet(block=timm.models.resnet.BasicBlock):
    return timm.models.resnet.ResNet(
        block=block,
        layers=[2, 1, 1, 1],
        num_classes=8,
        stem_width=8,
        channels=(8, 16, 32, 64),
    ).eval()


def _get_output_conv(block):
    return block.conv3 if isinstance(block, timm.models.resnet.Bottleneck) else block.conv2


class _UnsupportedBottleneck(timm.models.resnet.Bottleneck):
    def forward(self, input):
        return super().forward(input)


def test_resnet_recipe_support_is_limited_to_standard_blocks():
    model = timm.models.resnet.ResNet(
        block=_UnsupportedBottleneck,
        layers=[1, 1, 1, 1],
        num_classes=8,
        stem_width=8,
        channels=(8, 16, 32, 64),
    ).eval()

    assert not is_resnet_quantization_supported(model)
    model = mtq.quantize(model, {**deepcopy(mtq.INT8_DEFAULT_CFG), "algorithm": None})
    assert not hasattr(_get_output_conv(model.layer1[0]), "block_input_activation_quantizer")


@pytest.mark.parametrize(
    ("recipe_name", "conv_num_bits", "conv_weight_axis", "fc_num_bits", "fc_block_size"),
    [
        ("fp8", (4, 3), None, (4, 3), None),
        ("int8", 8, 0, 8, None),
        ("mxfp8", (4, 3), None, (4, 3), 32),
        ("nvfp4", (4, 3), None, (2, 1), 16),
        ("nvfp4_awq_lite", (4, 3), None, (2, 1), 16),
    ],
)
@pytest.mark.parametrize(
    "block_type", [timm.models.resnet.BasicBlock, timm.models.resnet.Bottleneck]
)
def test_resnet_recipe_quantizer_choices(
    recipe_name, conv_num_bits, conv_weight_axis, fc_num_bits, fc_block_size, block_type
):
    recipe = load_recipe(f"timm/resnet/ptq/{recipe_name}")
    config = recipe.quantize.model_dump()
    config["algorithm"] = None
    model = mtq.quantize(_get_resnet(block_type), config)

    assert is_resnet_quantization_supported(model)
    assert not model.conv1.input_quantizer.is_enabled
    assert not model.conv1.weight_quantizer.is_enabled
    assert not model.maxpool.input_quantizer.is_enabled

    for block in (model.layer1[1], model.layer2[0]):
        output_conv = _get_output_conv(block)
        assert output_conv.block_input_activation_quantizer.is_enabled
        assert output_conv.block_input_activation_quantizer.num_bits == conv_num_bits
        assert output_conv.block_input_activation_quantizer.axis is None
        assert output_conv.block_input_activation_quantizer.block_sizes is None
        assert output_conv.input_quantizer.num_bits == conv_num_bits
        assert output_conv.weight_quantizer.num_bits == conv_num_bits
        assert output_conv.weight_quantizer.axis == conv_weight_axis
        assert not output_conv.output_quantizer.is_enabled
        assert not block.conv1.input_quantizer.is_enabled

    projection_block = model.layer2[0]
    assert not projection_block.downsample[0].input_quantizer.is_enabled
    assert _get_output_conv(projection_block).residual_quantizer.is_enabled
    assert _get_output_conv(projection_block).residual_quantizer.num_bits == conv_num_bits
    assert not hasattr(_get_output_conv(model.layer1[1]), "residual_quantizer")

    pool_input_quantizers = [
        module
        for name, module in model.named_modules()
        if name.startswith("global_pool.") and name.endswith("input_quantizer")
    ]
    assert not any(quantizer.is_enabled for quantizer in pool_input_quantizers)

    final_output_conv = _get_output_conv(model.layer4[-1])
    assert final_output_conv.model_output_activation_quantizer.is_enabled == (recipe_name == "int8")
    if recipe_name == "int8":
        assert final_output_conv.model_output_activation_quantizer.num_bits == 8

    assert model.fc.weight_quantizer.num_bits == fc_num_bits
    if fc_block_size is None:
        assert model.fc.weight_quantizer.block_sizes is None
    else:
        assert model.fc.weight_quantizer.block_sizes[-1] == fc_block_size
    if recipe_name in ("fp8", "int8"):
        assert not model.fc.weight_quantizer.is_enabled
        assert not model.fc.input_quantizer.is_enabled
    else:
        assert model.fc.weight_quantizer.is_enabled
        assert model.fc.input_quantizer.is_enabled


@pytest.mark.parametrize(
    "config_name",
    ["FP8_DEFAULT_CFG", "INT8_DEFAULT_CFG", "MXFP8_DEFAULT_CFG", "NVFP4_DEFAULT_CFG"],
)
def test_stock_configs_do_not_enable_resnet_residual_quantizers(config_name):
    config = deepcopy(getattr(mtq, config_name))
    config["algorithm"] = None
    model = mtq.quantize(_get_resnet(), config)

    for block in (model.layer1[1], model.layer2[0]):
        output_conv = _get_output_conv(block)
        assert not output_conv.block_input_activation_quantizer.is_enabled
        if block.downsample is not None:
            assert not output_conv.residual_quantizer.is_enabled
        assert block.conv1.input_quantizer.is_enabled
    assert model.layer2[0].downsample[0].input_quantizer.is_enabled
    assert not _get_output_conv(model.layer4[-1]).model_output_activation_quantizer.is_enabled


def test_parent_conv_rule_matches_resnet_conv_subclasses():
    model = mtq.quantize(
        _get_resnet(),
        {
            "quant_cfg": [
                {"quantizer_name": "*", "enable": False},
                {
                    "parent_class": "nn.Conv2d",
                    "quantizer_name": "*weight_quantizer",
                    "cfg": {"num_bits": 7},
                },
            ],
            "algorithm": None,
        },
    )

    for conv in (
        model.conv1,
        model.layer1[0].conv1,
        _get_output_conv(model.layer1[0]),
        model.layer2[0].downsample[0],
        _get_output_conv(model.layer2[0]),
    ):
        assert conv.weight_quantizer.is_enabled
        assert conv.weight_quantizer.num_bits == 7


def test_resnet_recipe_disables_only_first_deep_stem_convolution():
    model = timm.models.resnet.ResNet(
        block=timm.models.resnet.Bottleneck,
        layers=[1, 1, 1, 1],
        num_classes=8,
        stem_width=8,
        stem_type="deep",
        channels=(8, 16, 32, 64),
    ).eval()
    recipe = load_recipe("timm/resnet/ptq/int8")
    config = recipe.quantize.model_dump()
    config["algorithm"] = None
    model = mtq.quantize(model, config)

    stem_convs = [module for module in model.conv1.modules() if hasattr(module, "weight_quantizer")]
    assert len(stem_convs) == 3
    assert not stem_convs[0].input_quantizer.is_enabled
    assert not stem_convs[0].weight_quantizer.is_enabled
    assert all(conv.input_quantizer.is_enabled for conv in stem_convs[1:])
    assert all(conv.weight_quantizer.is_enabled for conv in stem_convs[1:])


def test_resnet_recipe_quantizes_replacement_stem_pool_convolution():
    model = timm.models.resnet.ResNet(
        block=timm.models.resnet.Bottleneck,
        layers=[1, 1, 1, 1],
        num_classes=8,
        stem_width=8,
        stem_type="deep",
        replace_stem_pool=True,
        channels=(8, 16, 32, 64),
    ).eval()
    recipe = load_recipe("timm/resnet/ptq/int8")
    config = recipe.quantize.model_dump()
    config["algorithm"] = None
    model = mtq.quantize(model, config)

    assert model.maxpool[0].input_quantizer.is_enabled
    assert model.maxpool[0].weight_quantizer.is_enabled


def test_resnet_recipe_handles_avg_down_and_antialias_pools():
    model = timm.models.resnet.ResNet(
        block=timm.models.resnet.Bottleneck,
        layers=[1, 1, 1, 1],
        num_classes=8,
        stem_width=8,
        avg_down=True,
        aa_layer=torch.nn.AvgPool2d,
        channels=(8, 16, 32, 64),
    ).eval()
    recipe = load_recipe("timm/resnet/ptq/mxfp8")
    config = recipe.quantize.model_dump()
    config["algorithm"] = None
    model = mtq.quantize(model, config)

    block = model.layer2[0]
    assert not block.aa.input_quantizer.is_enabled
    assert not block.downsample[0].input_quantizer.is_enabled
    assert block.downsample[1].input_quantizer.is_enabled
    assert block.downsample[1].input_quantizer.num_bits == (4, 3)


@pytest.mark.parametrize(
    "block_type", [timm.models.resnet.BasicBlock, timm.models.resnet.Bottleneck]
)
def test_resnet_recipe_calibrates_residual_quantizers_in_one_pass(block_type):
    model = _get_resnet(block_type)
    recipe = load_recipe("timm/resnet/ptq/int8")
    calibration_calls = 0

    def forward_loop(quantized_model):
        nonlocal calibration_calls
        calibration_calls += 1
        quantized_model(torch.randn(2, 3, 32, 32))

    model = mtq.quantize(model, recipe.quantize.model_dump(), forward_loop=forward_loop)

    assert calibration_calls == 1
    for block in (model.layer1[1], model.layer2[0]):
        quantizers = [_get_output_conv(block).block_input_activation_quantizer]
        if block.downsample is not None:
            quantizers.append(_get_output_conv(block).residual_quantizer)
        for quantizer in quantizers:
            assert quantizer.amax is not None
            assert torch.isfinite(quantizer.amax).all()
            assert torch.all(quantizer.amax > 0)

    model_output_quantizer = _get_output_conv(model.layer4[-1]).model_output_activation_quantizer
    assert model_output_quantizer.amax is not None
    assert torch.isfinite(model_output_quantizer.amax).all()
    assert torch.all(model_output_quantizer.amax > 0)


def test_auto_quantize_uses_resnet_conv_format_for_cost_and_replay():
    recipe = load_recipe("timm/resnet/ptq/nvfp4")
    config = recipe.quantize.model_dump()
    config["algorithm"] = None
    model = mtq.quantize(_get_resnet(), config)
    block = model.layer2[0]
    output_conv = _get_output_conv(block)
    quant_recipe = QuantRecipe(recipe.quantize.model_dump(), name="resnet_nvfp4")
    hparam = QuantRecipeHparam(
        [quant_recipe],
        quant_modules=[output_conv],
        score_modules=[block],
        quant_module_names=["layer2.0.conv2"],
    )

    assert hparam.quant_module_replay_attrs["layer2.0.conv2"] == (
        "input_quantizer",
        "weight_quantizer",
        "output_quantizer",
        "block_input_activation_quantizer",
        "residual_quantizer",
    )
    assert hparam.quant_module_parent_classes["layer2.0.conv2"] == (
        "timm.ResNetProjectionOutputConv2d"
    )
    quantizer_choice = hparam._all_quantizer_choices[quant_recipe][output_conv]
    for quantizer_name in (
        "input_quantizer",
        "weight_quantizer",
        "block_input_activation_quantizer",
        "residual_quantizer",
    ):
        assert quantizer_choice[quantizer_name].num_bits == (4, 3)
    assert hparam.get_cost(quant_recipe) == pytest.approx(output_conv.weight.numel() * 0.5)

    hparam_name = "layer2.0.conv2.quant_recipe"
    search_state = {
        "best": {"recipe": {hparam_name: quant_recipe}},
        "candidate_stats": {
            hparam_name: {
                "module_names": hparam.quant_module_names,
                "quantizer_attrs": hparam.quant_module_replay_attrs,
                "parent_classes": hparam.quant_module_parent_classes,
            }
        },
    }
    with pytest.warns(UserWarning, match="algorithm='max'"):
        replay_config = mtq.get_auto_quantize_config(search_state)
    entries = {entry["quantizer_name"]: entry for entry in replay_config["quant_cfg"]}
    for quantizer_name in hparam.quant_module_replay_attrs["layer2.0.conv2"]:
        entry = entries[f"layer2.0.conv2.{quantizer_name}"]
        if quantizer_name == "output_quantizer":
            assert not entry["enable"]
        else:
            assert entry["enable"]
            assert entry["cfg"]["num_bits"] == (4, 3)

    assert AutoQuantizeGradientSearcher._is_auto_quantize_module(output_conv)
    assert not AutoQuantizeGradientSearcher._is_auto_quantize_module(model.conv1)


def test_auto_quantize_replay_keeps_resnet_block_input_convs_disabled():
    recipe = load_recipe("timm/resnet/ptq/int8")
    config = recipe.quantize.model_dump()
    config["algorithm"] = None
    model = mtq.quantize(_get_resnet(), config)
    conv = model.layer2[0].conv1
    quant_recipe = QuantRecipe(recipe.quantize.model_dump(), name="resnet_int8")
    hparam = QuantRecipeHparam(
        [quant_recipe],
        quant_modules=[conv],
        quant_module_names=["layer2.0.conv1"],
    )
    hparam_name = "layer2.0.conv1.quant_recipe"
    search_state = {
        "best": {"recipe": {hparam_name: quant_recipe}},
        "candidate_stats": {
            hparam_name: {
                "module_names": hparam.quant_module_names,
                "quantizer_attrs": hparam.quant_module_replay_attrs,
                "parent_classes": hparam.quant_module_parent_classes,
            }
        },
    }

    with pytest.warns(UserWarning, match="algorithm='max'"):
        replay_config = mtq.get_auto_quantize_config(search_state)
    entries = {entry["quantizer_name"]: entry for entry in replay_config["quant_cfg"]}

    assert not entries["layer2.0.conv1.input_quantizer"]["enable"]


@pytest.mark.parametrize(
    "block_type", [timm.models.resnet.BasicBlock, timm.models.resnet.Bottleneck]
)
def test_auto_quantize_calibrates_and_scores_resnet_residual_quantizers(block_type):
    recipes = [load_recipe(f"timm/resnet/ptq/{name}") for name in ("int8", "fp8")]
    data = [{"image": torch.randn(1, 3, 32, 32), "label": torch.tensor([1])}]

    model, search_state = mtq.auto_quantize(
        _get_resnet(block_type),
        constraints={"effective_bits": 8.0},
        quantization_formats=[recipe.quantize.model_dump() for recipe in recipes],
        data_loader=data,
        forward_step=lambda model, batch: model(batch["image"]),
        loss_func=lambda output, batch: torch.nn.functional.cross_entropy(output, batch["label"]),
        num_calib_steps=1,
        num_score_steps=1,
    )

    block = model.layer2[0]
    output_conv = _get_output_conv(block)
    for quantizer in (
        output_conv.block_input_activation_quantizer,
        output_conv.residual_quantizer,
    ):
        assert quantizer.is_enabled
        assert quantizer.amax is not None
        assert torch.isfinite(quantizer.amax).all()
        assert torch.all(quantizer.amax > 0)

    hparam = output_conv.get_hparam("quant_recipe")
    assert hparam.score_modules == [block]
    block_convs = [block.conv1, block.conv2, block.downsample[0]]
    if isinstance(block, timm.models.resnet.Bottleneck):
        block_convs.append(block.conv3)
    assert all(conv.get_hparam("quant_recipe") is hparam for conv in block_convs)
    assert all(
        conv.weight_quantizer.num_bits == output_conv.block_input_activation_quantizer.num_bits
        for conv in block_convs
    )
    output_conv_name = (
        "layer2.0.conv3" if isinstance(block, timm.models.resnet.Bottleneck) else "layer2.0.conv2"
    )
    candidate_stat = next(
        stat
        for stat in search_state["candidate_stats"].values()
        if output_conv_name in stat["module_names"]
    )
    assert candidate_stat["quantizer_attrs"][output_conv_name][-2:] == (
        "block_input_activation_quantizer",
        "residual_quantizer",
    )
    assert candidate_stat["parent_classes"][output_conv_name] == (
        "timm.ResNetProjectionOutputConv2d"
    )

    final_block = model.layer4[-1]
    final_output_conv = _get_output_conv(final_block)
    final_hparam = final_output_conv.get_hparam("quant_recipe")
    assert final_hparam.score_modules == [model.global_pool]
    assert final_output_conv.model_output_activation_quantizer.is_enabled == (
        final_output_conv.block_input_activation_quantizer.num_bits == 8
    )
    for quantizer_choices in final_hparam._all_quantizer_choices.values():
        quantizer = quantizer_choices[final_output_conv]["model_output_activation_quantizer"]
        if quantizer.is_enabled:
            assert quantizer.amax is not None
            assert torch.isfinite(quantizer.amax).all()
            assert torch.all(quantizer.amax > 0)


def test_resnet_quantizer_hooks_survive_save_restore():
    inputs = torch.randn(1, 3, 32, 32)
    recipe = load_recipe("timm/resnet/ptq/int8")
    model = mtq.quantize(
        _get_resnet(),
        recipe.quantize.model_dump(),
        forward_loop=lambda quantized_model: quantized_model(inputs),
    )
    expected = model(inputs)

    buffer = io.BytesIO()
    mto.save(model, buffer)
    buffer.seek(0)
    restored = mto.restore(_get_resnet(), buffer).eval()

    block = restored.layer2[0]
    output_conv = _get_output_conv(block)
    counts = {"block_input": 0, "projection": 0}

    def count_call(name):
        def hook(_module, _inputs, _output):
            counts[name] += 1

        return hook

    output_conv.block_input_activation_quantizer.register_forward_hook(count_call("block_input"))
    output_conv.residual_quantizer.register_forward_hook(count_call("projection"))

    assert torch.equal(restored(inputs), expected)
    assert counts == {"block_input": 1, "projection": 1}
    assert len(block.downsample._forward_hooks) == 1
