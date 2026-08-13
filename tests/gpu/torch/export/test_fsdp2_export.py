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
import warnings
from functools import partial

import pytest
import torch
from _test_utils.torch.export.utils import SmallQKVModel, ToyModel
from _test_utils.torch.misc import minimum_sm
from _test_utils.torch.quantization.tied_modules import (
    make_tied_linear_pair,
    wrap_in_parent_with_tied_keys,
)
from torch.distributed._composable.fsdp import fully_shard

import modelopt.torch.quantization as mtq
from modelopt.torch.export.layer_utils import is_quantlinear
from modelopt.torch.export.model_utils import (
    TiedWeightMap,
    _build_tied_alias_map,
    build_tied_weight_map,
)
from modelopt.torch.export.unified_export_hf import (
    _export_quantized_weight,
    requantize_resmooth_fused_llm_layers,
)
from modelopt.torch.quantization.utils import fsdp2_aware_weight_update, patch_fsdp_mp_dtypes


def _update_weight_test(rank, size):
    """Test fsdp2 weight update context for weight update -> only value changed"""
    with patch_fsdp_mp_dtypes():
        # Define and shard model
        model = ToyModel(dims=[4, 4], bias=False).to("cuda")

        assert not torch.equal(
            model.linears.weight.data,
            torch.zeros(4, 4).to(model.linears.weight.device).to(model.linears.weight.dtype),
        )

        fully_shard(model.linears)
        fully_shard(model)

        torch.distributed.barrier()

        for name, module in model.named_modules():
            if "linears" in name:
                with fsdp2_aware_weight_update(model, module):
                    module.weight.data = torch.zeros_like(module.weight.data)

        torch.distributed.barrier()
        model.linears.unshard()

        # Check if weights are as expected after unshard
        for param in model.parameters():
            assert torch.allclose(
                torch.zeros(4, 4).to(param.data.device).to(param.data.dtype), param.data
            )

        # Check if forward pass is as expected
        model.linears.reshard()
        output = model(torch.randn(4, 4).to(model.linears.weight.device))
        assert torch.allclose(torch.zeros(4, 4).to(output.device).to(output.dtype), output)


def _compress_weight_test(rank, size):
    """Test fsdp2 weight update context for weight compression -> only value,shape and dtype changed"""
    with patch_fsdp_mp_dtypes():
        # Define and shard model
        model = ToyModel(dims=[6, 6], bias=False).to("cuda")

        assert not torch.equal(
            model.linears.weight.data,
            torch.zeros(6, 6).to(model.linears.weight.device).to(model.linears.weight.dtype),
        )

        fully_shard(model.linears)
        fully_shard(model)
        torch.distributed.barrier()

        for name, module in model.named_modules():
            if "linears" in name:
                with fsdp2_aware_weight_update(model, module):
                    module.weight.data = (
                        torch.zeros(2, 2).to(torch.float8_e4m3fn).to(module.weight.data.device)
                    )

        torch.distributed.barrier()
        model.linears.unshard()
        # Check if weights are as expected after unshard
        for param in model.parameters():
            assert param.data.dtype == torch.float8_e4m3fn


def _compare_parameters_and_buffers(model1, model2):
    params1 = dict(model1.named_parameters())
    params2 = dict(model2.named_parameters())
    assert len(params1) == len(params2)
    for name, param in params1.items():
        assert torch.allclose(param.to(torch.bfloat16), params2[name].to(torch.bfloat16)), (
            f"Parameters {name} are not close, {param} != {params2[name]}"
        )
    buffers1 = dict(model1.named_buffers())
    buffers2 = dict(model2.named_buffers())
    assert len(buffers1) == len(buffers2)
    for name, buffer in buffers1.items():
        assert torch.allclose(buffer.to(torch.bfloat16), buffers2[name].to(torch.bfloat16)), (
            f"Buffers {name} are not close, {buffer} != {buffers2[name]}"
        )


def _fuse_layers(rank, size, quant_config, bias):
    with patch_fsdp_mp_dtypes():
        # Initialize model
        model = SmallQKVModel(dim=32, bias=bias).to("cuda")
        non_fsdp_model = SmallQKVModel(dim=32, bias=bias).to("cuda")
        non_fsdp_model.load_state_dict(copy.deepcopy(model.state_dict()))
        model.eval()
        non_fsdp_model.eval()

        _compare_parameters_and_buffers(model, non_fsdp_model)

        # Create calibration data ONCE
        calib_data = torch.randn(1, 32, device="cuda")

        def calib_fn(x):
            return x(calib_data)

        # Shard model
        fully_shard(model)
        torch.distributed.barrier()

        # Quantize model
        mtq.quantize(model, quant_config, calib_fn)
        mtq.quantize(non_fsdp_model, quant_config, calib_fn)

        torch.distributed.barrier()

        model.apply_embed = True
        non_fsdp_model.apply_embed = True

        requantize_resmooth_fused_llm_layers(model)
        requantize_resmooth_fused_llm_layers(non_fsdp_model)

        torch.distributed.barrier()

        # Unshard model
        model.unshard()

        _compare_parameters_and_buffers(model, non_fsdp_model)


def _export_quantized_weight_test(rank, size, quant_config, bias):
    with patch_fsdp_mp_dtypes():
        # Initialize model
        model = SmallQKVModel(dim=32, bias=bias).to("cuda")
        non_fsdp_model = SmallQKVModel(dim=32, bias=bias).to("cuda")
        non_fsdp_model.load_state_dict(copy.deepcopy(model.state_dict()))
        model.eval()
        non_fsdp_model.eval()
        _compare_parameters_and_buffers(model, non_fsdp_model)

        # Create calibration data ONCE
        calib_data = torch.randn(1, 32, device="cuda")

        def calib_fn(x):
            return x(calib_data)

        # Shard model
        fully_shard(model)
        torch.distributed.barrier()

        # Quantize model
        mtq.quantize(model, quant_config, calib_fn)
        mtq.quantize(non_fsdp_model, quant_config, calib_fn)

        torch.distributed.barrier()

        model.apply_embed = True
        non_fsdp_model.apply_embed = True

        requantize_resmooth_fused_llm_layers(model)
        requantize_resmooth_fused_llm_layers(non_fsdp_model)

        torch.distributed.barrier()

        for name, sub_module in model.named_modules():
            if is_quantlinear(sub_module):
                with fsdp2_aware_weight_update(model, sub_module):
                    _export_quantized_weight(sub_module, torch.float16)

        for name, sub_module in non_fsdp_model.named_modules():
            if is_quantlinear(sub_module):
                with fsdp2_aware_weight_update(non_fsdp_model, sub_module):
                    _export_quantized_weight(sub_module, torch.float16)

        torch.distributed.barrier()
        # Unshard model
        model.unshard()

        _compare_parameters_and_buffers(model, non_fsdp_model)


def _tied_alias_map_fsdp2_test(rank, size):
    """A declared tie forms a shared-parameter group BEFORE fully_shard but not after.

    ``_build_tied_alias_map`` groups names by ``id(parameter)``. ``fully_shard``
    splits the one shared ``nn.Parameter`` into two distinct per-module sharded
    params, so the id-group -- and therefore the alias map -- is only recoverable
    if the map is captured *before* sharding. This is the motivation for building
    the tie-map at load time rather than at export entry: names are stable across
    sharding, object identity is not.
    """
    with patch_fsdp_mp_dtypes():
        enc, dec = make_tied_linear_pair(in_features=32, out_features=32)
        model = wrap_in_parent_with_tied_keys(enc, dec, decoder_canonical=True).to("cuda")

        # Pre-shard: the tie is one shared Parameter -> alias map is produced.
        amap_pre = _build_tied_alias_map(model)
        assert amap_pre == {"encoder.weight": "decoder.weight"}, (
            f"pre-shard map should capture the declared tie, got {amap_pre}"
        )
        # Option B: capture the map now (resident), as a caller would before sharding.
        captured = build_tied_weight_map(model)

        fully_shard(model.encoder)
        fully_shard(model.decoder)
        fully_shard(model)
        torch.distributed.barrier()

        # Post-shard: fully_shard split the shared param -> id-group is gone, the
        # map is empty, and the FSDP2 guard warns rather than silently skipping.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            amap_post = _build_tied_alias_map(model)
        assert amap_post == {}, f"post-shard map should be empty (tie split), got {amap_post}"
        assert any("tied-weight alias" in str(w.message) for w in caught), (
            "expected the FSDP2 unrealized-tie warning after sharding"
        )

        # The pre-shard captured map still resolves the tie (names survive sharding).
        assert isinstance(captured, TiedWeightMap)
        assert captured.group_key("encoder.weight") == "decoder.weight"
        assert captured.group_key("decoder.weight") == "decoder.weight"


def test_fsdp2_tied_alias_map_needs_preshard_capture(dist_workers):
    if torch.cuda.device_count() < 2:
        pytest.skip("needs >=2 GPUs to shard a tied weight into distinct params")
    dist_workers.run(_tied_alias_map_fsdp2_test)


@minimum_sm(90)
def test_fsdp2_weight_compress_context_for_export(dist_workers):
    dist_workers.run(_compress_weight_test)


def test_fsdp2_weight_update_context_for_export(dist_workers):
    dist_workers.run(_update_weight_test)


@pytest.mark.parametrize(
    "quant_config",
    [
        mtq.INT8_DEFAULT_CFG,
        mtq.INT4_AWQ_CFG,
        mtq.FP8_DEFAULT_CFG,
        mtq.NVFP4_DEFAULT_CFG,
        mtq.NVFP4_AWQ_LITE_CFG,
        # mtq.W4A8_AWQ_BETA_CFG, #TODO: Fix unit test for this case
        # mtq.FP8_2D_BLOCKWISE_WEIGHT_ONLY_CFG, #TODO: Fix unit test for this case
        mtq.W4A8_MXFP4_FP8_CFG,
        mtq.NVFP4_MLP_ONLY_CFG,
        mtq.NVFP4_OMLP_ONLY_CFG,
    ],
)
@pytest.mark.parametrize("bias", [True, False])
def test_fsdp2_weight_update_context_for_fuse_layers(dist_workers, quant_config, bias):
    dist_workers.run(partial(_fuse_layers, quant_config=quant_config, bias=bias))


@pytest.mark.parametrize(
    "quant_config",
    [
        mtq.INT8_DEFAULT_CFG,
        mtq.INT4_AWQ_CFG,
        mtq.FP8_DEFAULT_CFG,
        mtq.NVFP4_DEFAULT_CFG,
        mtq.NVFP4_AWQ_LITE_CFG,
        # mtq.W4A8_AWQ_BETA_CFG, #TODO: Fix unit test for this case
        # mtq.FP8_2D_BLOCKWISE_WEIGHT_ONLY_CFG, #TODO: Fix unit test for this case
        mtq.W4A8_MXFP4_FP8_CFG,
        mtq.NVFP4_MLP_ONLY_CFG,
        mtq.NVFP4_OMLP_ONLY_CFG,
    ],
)
@pytest.mark.parametrize("bias", [True, False])
def test_fsdp2_weight_update_context_for_export_quantized_weight(dist_workers, quant_config, bias):
    dist_workers.run(partial(_export_quantized_weight_test, quant_config=quant_config, bias=bias))
