# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import copy
import io

import pytest
import torch
from _test_utils.torch.distributed.utils import spawn_multiprocess_job
from _test_utils.torch.quantization.models import SimpleConv, SimpleConvLinear, SimpleLinear

import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.algorithms import (
    QuantRecipe,
    QuantRecipeHparam,
    estimate_quant_compression,
)
from modelopt.torch.quantization.config import _default_disabled_quantizer_cfg
from modelopt.torch.utils.distributed import DistributedProcessGroup


class _AttentionLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = torch.nn.Linear(32, 32)
        self.k_proj = torch.nn.Linear(32, 32)
        self.v_proj = torch.nn.Linear(32, 32)
        self.o_proj = torch.nn.Linear(32, 32)

    def forward(self, x):
        for layer in [self.q_proj, self.k_proj, self.v_proj, self.o_proj]:
            x = layer(x)
        return x


class TransformerBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = _AttentionLayer()
        self.mlp = torch.nn.Linear(32, 32)

    def forward(self, x):
        x = self.attn(x)
        x = self.mlp(x)
        return x

    def get_input(self):
        return torch.randn(1, 4, 32)


@pytest.mark.parametrize(
    ("quant_cfg", "other_quant_cfg", "is_less_than"),
    [
        (mtq.FP8_DEFAULT_CFG, None, True),
        (mtq.NVFP4_DEFAULT_CFG, mtq.FP8_DEFAULT_CFG, True),
        (None, mtq.INT8_DEFAULT_CFG, False),
    ],
)
def test_quant_recipe(quant_cfg, other_quant_cfg, is_less_than):
    qr_this = QuantRecipe(quant_cfg)
    qr_other = QuantRecipe(other_quant_cfg)
    assert (qr_this < qr_other) == is_less_than

    qr_this_duplicate = QuantRecipe(quant_cfg)
    assert qr_this_duplicate in {qr_this}


def test_quant_recipe_hparam():
    model_test = torch.nn.Linear(4, 16)
    model_ref = torch.nn.Linear(4, 16)
    model_ref.load_state_dict(model_test.state_dict())

    model_test = mtq.quantize(model_test, mtq.INT8_DEFAULT_CFG)
    model_ref = mtq.quantize(model_ref, mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG)

    search_recipes = [
        QuantRecipe(mtq.INT8_DEFAULT_CFG),
        QuantRecipe(mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG),
    ]
    hparam = QuantRecipeHparam(
        search_recipes,
        quant_modules=[model_test],
    )
    model_test._register_hparam("quant_recipe", hparam)
    assert model_test.quant_recipe == QuantRecipe(mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG)
    assert model_test.get_hparam("quant_recipe").choices == sorted(
        [*search_recipes, QuantRecipe(quant_cfg=None)]
    )

    model_test.quant_recipe = QuantRecipe(mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG)
    inputs = torch.randn(1, 4, 4)
    output_test = model_test(inputs)
    output_ref = model_ref(inputs)

    assert torch.allclose(output_test, output_ref)


# use this config to test custom quantization config
INT8_CUSTOM_QUANT_TEST_CFG = {
    "quant_cfg": {
        "*weight_quantizer": {"num_bits": 8, "axis": 0},
        "*input_quantizer": {"num_bits": 8, "axis": None},
        **_default_disabled_quantizer_cfg,
    },
    "algorithm": "smoothquant",
}


@pytest.mark.parametrize(
    "model_cls",
    [SimpleConv, SimpleConvLinear, SimpleLinear, TransformerBlock],
)
@pytest.mark.parametrize(
    ("search_formats", "min_bits", "search_bits"),
    [
        ([mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG, mtq.INT8_DEFAULT_CFG], 4.0, 6.0),
        ([mtq.INT4_AWQ_CFG, mtq.INT8_SMOOTHQUANT_CFG], 4.0, 6.0),
        ([mtq.INT4_AWQ_CFG, INT8_CUSTOM_QUANT_TEST_CFG], 4.0, 6.0),
        ([mtq.INT8_SMOOTHQUANT_CFG], 8.0, 11.0),
        ([None, mtq.INT8_SMOOTHQUANT_CFG], 8.0, 11.0),
    ],
)
@pytest.mark.parametrize(
    "method",
    ["gradient", "kl_div"],
)
def test_auto_quantize(model_cls, search_formats, min_bits, search_bits, method):
    model = model_cls()

    def loss_func(output):
        return output.sum()

    best_model, search_history = mtq.auto_quantize(
        model,
        constraints={"effective_bits": search_bits},
        quantization_formats=search_formats,
        data_loader=[model.get_input() for _ in range(2)],
        forward_step=lambda model, batch: model(batch),
        loss_func=lambda output, data: output.sum(),
        num_calib_steps=2,
        num_score_steps=2,
        verbose=True,
        method=method,
    )
    assert isinstance(search_history, dict)
    assert search_history["best"]["is_satisfied"]
    effective_bits_from_search = search_history["best"]["constraints"]["effective_bits"]
    assert effective_bits_from_search <= search_bits and effective_bits_from_search >= min_bits, (
        "Search failed!"
    )

    if model_cls == TransformerBlock:
        hparam = model.attn.q_proj.get_hparam("quant_recipe")
        for layer in [model.attn.k_proj, model.attn.v_proj]:
            assert layer.get_hparam("quant_recipe") == hparam
        assert ("attn.q_proj.quant_recipe" in search_history["candidate_stats"]) != (
            "attn.k_proj.quant_recipe" in search_history["candidate_stats"]
        )

    # test restore
    buffer = io.BytesIO()
    mto.save(best_model, buffer)
    buffer.seek(0)
    new_model = model_cls()
    new_model = mto.restore(new_model, buffer)

    input = model.get_input()
    output_ref = best_model(input)
    output_test = new_model(input)
    assert torch.allclose(output_ref, output_test)


def test_auto_quantize_disable_layers():
    model = TransformerBlock()

    def loss_func(output):
        return output.sum()

    best_model, search_history = mtq.auto_quantize(
        model,
        constraints={"effective_bits": 5.0},
        quantization_formats=[
            mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG,
            mtq.INT8_DEFAULT_CFG,
        ],
        data_loader=[model.get_input() for _ in range(2)],
        forward_step=lambda model, batch: model(batch),
        loss_func=lambda output, data: output.sum(),
        disabled_layers=["*mlp*"],
        num_calib_steps=2,
        num_score_steps=2,
        verbose=True,
    )

    assert not best_model.mlp.input_quantizer.is_enabled


def test_auto_quantize_disabled_layers_no_poison():
    """disabled_layers must only affect the matched layers, not all subsequent layer groups."""
    model = TransformerBlock()

    best_model, _ = mtq.auto_quantize(
        model,
        constraints={"effective_bits": 5.0},
        quantization_formats=[mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG, mtq.INT8_DEFAULT_CFG],
        data_loader=[model.get_input() for _ in range(2)],
        forward_step=lambda model, batch: model(batch),
        loss_func=lambda output, data: output.sum(),
        disabled_layers=["*mlp*"],
        num_calib_steps=2,
        num_score_steps=2,
    )

    assert not best_model.mlp.input_quantizer.is_enabled
    hparam = best_model.attn.q_proj.get_hparam("quant_recipe")
    assert QuantRecipe(mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG) in hparam.choices


INT4INT8_AWQ_CFG = {
    "quant_cfg": {
        "*weight_quantizer": [
            {"num_bits": 4, "block_sizes": {-1: 128, "type": "static"}, "enable": True},
            {"num_bits": 8, "axis": None, "enable": True},
        ],
        "*input_quantizer": {"num_bits": 8, "axis": None, "enable": True},
        "default": {"enable": False},
    },
    "algorithm": "awq_lite",
}


@pytest.mark.parametrize("config", [mtq.INT4_AWQ_CFG, mtq.INT8_SMOOTHQUANT_CFG, INT4INT8_AWQ_CFG])
def test_pqs_folding(config):
    model_ref = SimpleLinear()
    state_dict_ref = copy.deepcopy(model_ref.state_dict())
    inputs = model_ref.get_input()
    mtq.quantize(model_ref, config, lambda model: model(inputs))

    model_test = SimpleLinear()
    model_test.load_state_dict(state_dict_ref)
    QuantRecipe.disable_folding_pqs_to_weights()
    mtq.quantize(model_test, config, lambda model: model(inputs))

    assert torch.allclose(model_ref(inputs), model_test(inputs))

    QuantRecipe.fold_pqs_to_weights(model_test)
    assert torch.allclose(model_ref(inputs), model_test(inputs))


def _test_data_parallel_auto_quantize(rank, size):
    model = SimpleLinear()

    model, search_history = mtq.auto_quantize(
        model,
        constraints={"effective_bits": 11.0},
        quantization_formats=[mtq.INT8_SMOOTHQUANT_CFG],
        data_loader=[model.get_input() for _ in range(2)],
        forward_step=lambda model, batch: model(batch),
        loss_func=lambda output, data: output.sum(),
        num_calib_steps=2,
        num_score_steps=2,
        verbose=True,
    )

    search_history_rank0 = DistributedProcessGroup.get_dist_syncd_obj(
        search_history if rank == 0 else None,
        DistributedProcessGroup(None),
        lambda a: a[0],
    )

    print(f"rank {rank} search_history: {search_history}")
    if search_history != search_history_rank0:
        print(f"rank {rank} search_history_rank0: {search_history_rank0}")

    # Assert that the costs, scores and searched recipes are the same across all ranks
    assert search_history == search_history_rank0

    assert search_history["best"]["is_satisfied"]


def test_data_parallel_auto_quantize(skip_on_windows):
    spawn_multiprocess_job(4, _test_data_parallel_auto_quantize, backend="gloo")


def test_estimate_quant_compression():
    nvfp4_affine_kv_cfg = mtq.config.QuantizeConfig(**mtq.NVFP4_AFFINE_KV_CFG)
    assert estimate_quant_compression(nvfp4_affine_kv_cfg) == 0.25

    nvfp4_awq_clip_cfg = mtq.config.QuantizeConfig(**mtq.NVFP4_AWQ_CLIP_CFG)
    assert estimate_quant_compression(nvfp4_awq_clip_cfg) == 0.25

    nvfp4_awq_full_cfg = mtq.config.QuantizeConfig(**mtq.NVFP4_AWQ_FULL_CFG)
    assert estimate_quant_compression(nvfp4_awq_full_cfg) == 0.25

    nvfp4_awq_lite_cfg = mtq.config.QuantizeConfig(**mtq.NVFP4_AWQ_LITE_CFG)
    assert estimate_quant_compression(nvfp4_awq_lite_cfg) == 0.25

    nvfp4_default_cfg = mtq.config.QuantizeConfig(**mtq.NVFP4_DEFAULT_CFG)
    assert estimate_quant_compression(nvfp4_default_cfg) == 0.25

    nvfp4_kv_cfg = mtq.config.QuantizeConfig(**mtq.NVFP4_KV_CFG)
    assert estimate_quant_compression(nvfp4_kv_cfg) == 0.25

    nvfp4_kv_rotate_cfg = mtq.config.QuantizeConfig(**mtq.NVFP4_KV_ROTATE_CFG)
    assert estimate_quant_compression(nvfp4_kv_rotate_cfg) == 0.25

    nvfp4_svdquant_default_cfg = mtq.config.QuantizeConfig(**mtq.NVFP4_SVDQUANT_DEFAULT_CFG)
    assert estimate_quant_compression(nvfp4_svdquant_default_cfg) == 0.25

    int8_default_cfg = mtq.config.QuantizeConfig(**mtq.INT8_DEFAULT_CFG)
    assert estimate_quant_compression(int8_default_cfg) == 0.5

    int8_smoothquant_cfg = mtq.config.QuantizeConfig(**mtq.INT8_SMOOTHQUANT_CFG)
    assert estimate_quant_compression(int8_smoothquant_cfg) == 0.5

    fp8_default_cfg = mtq.config.QuantizeConfig(**mtq.FP8_DEFAULT_CFG)
    assert estimate_quant_compression(fp8_default_cfg) == 0.5

    fp8_per_channel_per_token_cfg = mtq.config.QuantizeConfig(**mtq.FP8_PER_CHANNEL_PER_TOKEN_CFG)
    assert estimate_quant_compression(fp8_per_channel_per_token_cfg) == 0.5

    fp8_2d_blockwise_weight_only_cfg = mtq.config.QuantizeConfig(
        **mtq.FP8_2D_BLOCKWISE_WEIGHT_ONLY_CFG
    )
    assert estimate_quant_compression(fp8_2d_blockwise_weight_only_cfg) == 0.5

    int4_blockwise_weight_only_cfg = mtq.config.QuantizeConfig(**mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG)
    assert estimate_quant_compression(int4_blockwise_weight_only_cfg) == 0.25

    int4_awq_cfg = mtq.config.QuantizeConfig(**mtq.INT4_AWQ_CFG)
    assert estimate_quant_compression(int4_awq_cfg) == 0.25

    w4a8_awq_beta_cfg = mtq.config.QuantizeConfig(**mtq.W4A8_AWQ_BETA_CFG)
    assert estimate_quant_compression(w4a8_awq_beta_cfg) == 0.25

    mxfp8_default_cfg = mtq.config.QuantizeConfig(**mtq.MXFP8_DEFAULT_CFG)
    assert estimate_quant_compression(mxfp8_default_cfg) == 0.5

    mxfp6_default_cfg = mtq.config.QuantizeConfig(**mtq.MXFP6_DEFAULT_CFG)
    assert estimate_quant_compression(mxfp6_default_cfg) == 0.375

    mxfp4_default_cfg = mtq.config.QuantizeConfig(**mtq.MXFP4_DEFAULT_CFG)
    assert estimate_quant_compression(mxfp4_default_cfg) == 0.25

    mxint8_default_cfg = mtq.config.QuantizeConfig(**mtq.MXINT8_DEFAULT_CFG)
    assert estimate_quant_compression(mxint8_default_cfg) == 0.5

    fp8_kv_cfg = mtq.config.QuantizeConfig(**mtq.FP8_KV_CFG)
    assert estimate_quant_compression(fp8_kv_cfg) == 0.5

    fp8_affine_kv_cfg = mtq.config.QuantizeConfig(**mtq.FP8_AFFINE_KV_CFG)
    assert estimate_quant_compression(fp8_affine_kv_cfg) == 0.5


@pytest.mark.parametrize("method", ["gradient", "kl_div"])
def test_auto_quantize_checkpoint_resume(method, tmp_path, capsys):
    """Test that checkpoint can be used to resume an interrupted search."""
    model = SimpleLinear()
    checkpoint_path = str(tmp_path / "autoquant_resume_checkpoint.pth")

    # First run: save checkpoint
    model_1, state_dict_1 = mtq.auto_quantize(
        model,
        constraints={"effective_bits": 6.0},
        quantization_formats=[mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG, mtq.INT8_DEFAULT_CFG],
        data_loader=[model.get_input() for _ in range(2)],
        forward_step=lambda model, batch: model(batch),
        loss_func=lambda output, data: output.sum(),
        num_calib_steps=2,
        num_score_steps=2,
        verbose=True,
        method=method,
        checkpoint=checkpoint_path,
    )

    # Clear captured output from first run
    capsys.readouterr()

    # Second run: resume with same constraint should produce same results
    model_2 = SimpleLinear()
    model_2, state_dict_2 = mtq.auto_quantize(
        model_2,
        constraints={"effective_bits": 6.0},  # Same constraint
        quantization_formats=[mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG, mtq.INT8_DEFAULT_CFG],
        data_loader=[model_2.get_input() for _ in range(2)],
        forward_step=lambda model, batch: model(batch),
        loss_func=lambda output, data: output.sum(),
        num_calib_steps=2,
        num_score_steps=2,
        verbose=True,
        method=method,
        checkpoint=checkpoint_path,
    )

    # Verify the restore message was printed on second run
    captured = capsys.readouterr()
    assert "Restored from checkpoint, skipping scoring" in captured.out, (
        "Expected restore message when resuming from checkpoint"
    )

    # Results should be identical when using same constraint
    assert state_dict_1["candidate_stats"] == state_dict_2["candidate_stats"]
    assert state_dict_1["best"]["recipe"] == state_dict_2["best"]["recipe"]
    assert (
        pytest.approx(state_dict_1["best"]["constraints"]["effective_bits"])
        == state_dict_2["best"]["constraints"]["effective_bits"]
    )
