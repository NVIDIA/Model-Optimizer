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
from unittest.mock import patch

import pytest
import torch
import torch.distributed
import torch.distributed as dist
from packaging.version import Version

import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq
import modelopt.torch.quantization.model_calib as model_calib_module  # needed for patching awq_lite
from modelopt.torch.quantization.backends.gemm_registry import enable_real_quant_gemm
from modelopt.torch.quantization.nn.modules.tensor_quantizer import SequentialQuantizer
from modelopt.torch.quantization.utils import is_quantized_linear
from modelopt.torch.utils import torch_to

INT4_AWQ_FULL_CFG = mtq.INT4_AWQ_CFG.copy()

INT4_AWQ_FULL_CFG["algorithm"] = "awq_full"

INT4_AWQ_CLIP_CFG = mtq.INT4_AWQ_CFG.copy()
INT4_AWQ_CLIP_CFG["algorithm"] = "awq_clip"

# SVDQuant test cfg
INT4_SVDQUANT_CFG = mtq.INT4_AWQ_CFG.copy()
INT4_SVDQUANT_CFG["algorithm"] = {"method": "svdquant", "lowrank": 8}

# SVDQuant test cfg
FP4_SVDQUANT_CFG = mtq.NVFP4_AWQ_LITE_CFG.copy()
FP4_SVDQUANT_CFG["algorithm"] = {"method": "svdquant", "lowrank": 8}


def get_awq_config(algorithm="awq_lite", block_size=8):
    config = copy.deepcopy(mtq.INT4_AWQ_CFG)
    config["quant_cfg"]["*weight_quantizer"]["block_sizes"] = {-1: block_size}
    config["algorithm"]["method"] = algorithm
    config["algorithm"]["debug"] = True
    if algorithm == "awq_clip":
        config["algorithm"].pop("alpha_step")
    return config


def quantize_model_and_forward(model, config, calib_data, compress=False):
    def forward_loop(model, run_backward=False):
        for batch in calib_data:
            output = model(batch)
            if run_backward:
                output.sum().backward()

    model = mtq.quantize(model, config, forward_loop=forward_loop)
    if compress:
        mtq.compress(model)

    for module in model.modules():
        assert not isinstance(module, torch.nn.Linear) or is_quantized_linear(module)

    model.train()

    # make sure we can run the forward and backward on the quantized model
    forward_loop(model, run_backward=True)


def save_restore_test(model_cls, device, quant_config, compress=False, version=None):
    # test restoring to an unquantized model
    model_quant = model_cls().to(device)
    model_ref = model_cls().to(device)

    calib_data = [model_quant.get_input().to(device) for _ in range(2)]
    quantize_model_and_forward(model_quant.to(device), quant_config, calib_data, compress)

    state_dict = mto.modelopt_state(model_quant)

    mto.restore_from_modelopt_state(model_ref, state_dict)
    model_ref.load_state_dict(model_quant.state_dict())
    assert torch.allclose(model_quant(calib_data[0]), model_ref(calib_data[0]))

    if version is not None and Version(version) < Version("0.29"):
        # Rest of the tests are not needed for version < 0.29
        return

    if not compress:
        # gpu: test restoring to a model on cpu. If the quantizer states are not initialized correctly,
        # the buffers will be created on cuda and this test will fail
        model_ref = model_cls().to("cpu")
        state_dict = torch_to(state_dict, device="cuda" if torch.cuda.is_available() else "cpu")
        mto.restore_from_modelopt_state(model_ref, state_dict)
        model_ref.load_state_dict(model_quant.state_dict())
        model_ref(
            calib_data[0].to("cpu")
        )  # make sure all the buffers are created in the right device
        model_ref.to(device)
        assert torch.allclose(model_quant(calib_data[0]), model_ref(calib_data[0]))

    # Test that smoothquant is restored correctly
    if quant_config == mtq.INT8_SMOOTHQUANT_CFG:
        for name, module in model_ref.named_modules():
            if isinstance(module, torch.nn.Linear):
                assert module.input_quantizer.axis is None

    # test restoring to a quantized model
    model_ref = model_cls().to(device)
    quantize_model_and_forward(model_ref, mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG, calib_data)
    with pytest.raises(AssertionError, match="Model already has modelopt state!"):
        mto.restore_from_modelopt_state(model_ref, state_dict)


def _distributed_attr_check(quantizer, attr: str, op=dist.ReduceOp.MAX, groups=[]):
    quantizer_attr = getattr(quantizer, attr).clone()
    for group in groups:
        if group is not None:
            dist.all_reduce(quantizer_attr, op=op, group=group)
    assert torch.allclose(quantizer_attr, getattr(quantizer, attr))


original_awq_lite = model_calib_module.awq_lite


def _debug_awq_lite(model, forward_loop, alpha_step=0.1, debug=True, **kwargs):
    """Function to mock awq_lite function to always use debug=True for testing"""
    return original_awq_lite(model, forward_loop, alpha_step, debug=True, **kwargs)


@patch("modelopt.torch.quantization.model_calib.awq_lite", side_effect=_debug_awq_lite)
def data_tensor_context_parallel_test_helper(
    model, config, mock_awq_lite, dp_group=None, tp_group=None, test_pre_quant_scale=True
):
    # Calib data should be different across each DP rank
    dp_rank = dist.get_rank(group=dp_group)
    calib_data = model.get_dummy_input(seed=dp_rank).cuda()

    if tp_group is not None:
        # The input to first layer, the column parallel should be the same across all tp ranks
        dist.all_reduce(calib_data, op=dist.ReduceOp.AVG, group=tp_group)

    def forward_loop(model):
        model(calib_data)

    model = mtq.quantize(model, config, forward_loop)

    # Input quantizer amax
    if config not in [mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG, mtq.INT4_AWQ_CFG]:
        _distributed_attr_check(
            model.fc1.input_quantizer, "amax", dist.ReduceOp.MAX, groups=[dp_group, tp_group]
        )
        _distributed_attr_check(
            model.fc2.input_quantizer, "amax", dist.ReduceOp.MAX, groups=[dp_group, tp_group]
        )

    # Per-tensor quantization (FP8/NVFP4) expects same amax across row and column parallel ranks
    # Channel-wise (INT8) only expects same amax across row parallel ranks
    # Block-wise quantization does not expect same amax across row and column parallel ranks
    if config in [mtq.FP8_DEFAULT_CFG, mtq.NVFP4_DEFAULT_CFG]:
        if isinstance(model.fc1.weight_quantizer, SequentialQuantizer):
            for quantizer in model.fc1.weight_quantizer:
                _distributed_attr_check(
                    quantizer, "amax", dist.ReduceOp.MAX, groups=[dp_group, tp_group]
                )
        else:
            _distributed_attr_check(
                model.fc1.weight_quantizer, "amax", dist.ReduceOp.MAX, groups=[dp_group, tp_group]
            )

    if config in [
        mtq.FP8_DEFAULT_CFG,
        mtq.NVFP4_DEFAULT_CFG,
        mtq.INT8_DEFAULT_CFG,
        mtq.INT8_SMOOTHQUANT_CFG,
    ]:
        if isinstance(model.fc2.weight_quantizer, SequentialQuantizer):
            for quantizer in model.fc2.weight_quantizer:
                _distributed_attr_check(
                    quantizer, "amax", dist.ReduceOp.MAX, groups=[dp_group, tp_group]
                )
        else:
            _distributed_attr_check(
                model.fc2.weight_quantizer, "amax", dist.ReduceOp.MAX, groups=[dp_group, tp_group]
            )

    # Lets check the column parallel pre_quant_scale; it should be the same across all tp ranks
    # It is different across DP/CP ranks since the input is different
    if (
        test_pre_quant_scale
        and tp_group
        and config in [mtq.INT8_SMOOTHQUANT_CFG, mtq.INT4_AWQ_CFG, mtq.W4A8_AWQ_BETA_CFG]
    ):
        input_quantizer = model.fc1.input_quantizer
        _distributed_attr_check(
            input_quantizer, "pre_quant_scale", dist.ReduceOp.MAX, groups=[dp_group, tp_group]
        )

    # Check act scale
    if config in [mtq.INT4_AWQ_CFG, mtq.W4A8_AWQ_BETA_CFG]:
        _distributed_attr_check(
            model.fc1.awq_lite, "act_scale", dist.ReduceOp.AVG, groups=[dp_group, tp_group]
        )


def auto_quantize_helper(model):
    model, search_state = mtq.auto_quantize(
        model,
        constraints={"effective_bits": 8.0},
        quantization_formats=[mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG, mtq.INT8_DEFAULT_CFG],
        data_loader=[model.get_dummy_input().cuda() for _ in range(2)],
        forward_step=lambda model, batch: model(batch),
        forward_backward_step=lambda model, batch: model(batch).sum().backward(),
        num_calib_steps=2,
        num_score_steps=2,
        verbose=True,
    )
    search_state_list = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(search_state_list, search_state)

    search_state_rank0 = search_state_list[0]
    for search_state in search_state_list[1:]:
        assert search_state == search_state_rank0


def compute_backward_grad(
    model,
    input_tensor,
    config=None,
    quantize: bool = False,
    enable_real_quant: bool = False,
    compress: bool = False,
):
    if quantize:

        def forward_loop(model, run_backward=False):
            calib_data = [model.get_input().to(torch.float16).cuda() for _ in range(8)]
            for batch in calib_data:
                output = model(batch)
                if run_backward:
                    output.sum().backward()

        mtq.quantize(model, config, forward_loop)

        if enable_real_quant:
            enable_real_quant_gemm(model)
    if compress:
        mtq.compress(model)
    output = model(input_tensor).sum()
    model.zero_grad()
    output.backward()
    weight_grads = []
    bias_grads = []
    for layer in model.net:
        if isinstance(layer, torch.nn.Linear):
            weight_grads.append(layer.weight.grad)
            bias_grads.append(layer.bias.grad if layer.bias is not None else None)
    return weight_grads, bias_grads
