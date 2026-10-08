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

"""Distributed coverage for rank-local fakequant checkpoint reload."""

import importlib.util
import warnings
from functools import partial
from pathlib import Path

import pytest
import torch
from _test_utils.torch.distributed.utils import DistributedWorkerPool
from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
from vllm.distributed.parallel_state import (
    destroy_model_parallel,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.model_executor.layers.linear import ColumnParallelLinear, RowParallelLinear

import modelopt.torch.quantization as mtq

_EXAMPLES_DIR = Path(__file__).resolve().parents[4] / "examples/vllm_serve"
_SPEC = importlib.util.spec_from_file_location(
    "vllm_reload_utils_test", _EXAMPLES_DIR / "vllm_reload_utils.py"
)
reload_utils = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(reload_utils)


def _test_rank_local_quantizer_reload(tmp_path, rank, size):
    assert size == 2
    config = VllmConfig(
        parallel_config=ParallelConfig(
            tensor_parallel_size=size,
            distributed_executor_backend="external_launcher",
            disable_custom_all_reduce=True,
        )
    )
    with set_current_vllm_config(config):
        init_distributed_environment(
            world_size=size, rank=rank, local_rank=torch.cuda.current_device(), backend="nccl"
        )
        initialize_model_parallel(tensor_model_parallel_size=size, backend="nccl")
        try:
            model = torch.nn.Module()
            model.folded = RowParallelLinear(8, 4, bias=False, params_dtype=torch.float32)
            model.active = ColumnParallelLinear(4, 8, bias=False, params_dtype=torch.float32)
            model = mtq.quantize(
                model,
                {
                    "quant_cfg": [
                        {"quantizer_name": "*", "enable": False},
                        {
                            "quantizer_name": "*.input_quantizer",
                            "cfg": {"num_bits": (4, 3)},
                        },
                        {
                            "quantizer_name": "*.weight_quantizer",
                            "cfg": {"num_bits": (4, 3), "axis": 0},
                        },
                    ],
                    "algorithm": None,
                },
            ).cuda()
            values = torch.tensor([-1.0625, -0.5625, 0.5625, 1.0625], device="cuda")
            for layer in (model.folded, model.active):
                with torch.no_grad():
                    layer.weight.copy_(values.repeat(4, 1) * (rank + 1))
                layer.input_quantizer.amax = torch.tensor(8.0, device="cuda")
                layer.weight_quantizer.amax = torch.full((4, 1), 448.0, device="cuda")
            model.folded.input_quantizer.pre_quant_scale = torch.ones(4, device="cuda")
            if rank == 1:
                with torch.no_grad():
                    model.folded.weight.copy_(model.folded.weight_quantizer(model.folded.weight))
            original_folded_weight = model.folded.weight.detach().clone()
            original_active_weight = model.active.weight.detach().clone()
            global_amax = torch.tensor(
                [56, 56, 112, 112, 224, 224, 448, 448], dtype=torch.float32
            ).reshape(8, 1)
            global_pre_scale = torch.arange(1, 9, dtype=torch.float32) / 8
            checkpoint = {
                "active.weight_quantizer._amax": global_amax,
                "folded.input_quantizer._pre_quant_scale": global_pre_scale,
            }
            if rank == 0:
                checkpoint["folded.weight_quantizer._amax"] = torch.full((4, 1), 3.5)
                checkpoint["folded.input_quantizer._amax"] = torch.tensor(32.0)
            else:
                checkpoint["active.input_quantizer._amax"] = torch.tensor(64.0)
            path = tmp_path / f"quantizer_state_rank{rank}.pth"
            torch.save(checkpoint, path)

            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                model.load_state_dict(reload_utils.load_state_dict_from_path(str(path), model))
            assert not any("missing from every rank" in str(w.message) for w in captured)
            assert not any("Could not all_gather" in str(w.message) for w in captured)
            assert model.folded.weight_quantizer.is_enabled == (rank == 0)
            assert model.active.weight_quantizer.is_enabled
            assert model.folded.input_quantizer.is_enabled
            assert model.active.input_quantizer.is_enabled
            expected_amax = global_amax[4 * rank : 4 * (rank + 1)].cuda()
            expected_pre_scale = global_pre_scale[4 * rank : 4 * (rank + 1)].cuda()
            torch.testing.assert_close(model.active.weight_quantizer.amax, expected_amax)
            torch.testing.assert_close(
                model.folded.input_quantizer.pre_quant_scale, expected_pre_scale
            )
            assert model.folded.input_quantizer.amax == (32 if rank == 0 else 8)
            assert model.active.input_quantizer.amax == (8 if rank == 0 else 64)
            reload_utils.shard_pre_quant_scale_for_tp(model)
            torch.testing.assert_close(
                model.folded.input_quantizer.pre_quant_scale, expected_pre_scale
            )

            active_scale = expected_amax / 448
            expected_active_weight = (original_active_weight / active_scale).clamp(-448, 448).to(
                torch.float8_e4m3fn
            ).float() * active_scale
            if rank == 0:
                expected_folded_weight = (original_folded_weight * 128).clamp(-448, 448).to(
                    torch.float8_e4m3fn
                ).float() / 128
            else:
                expected_folded_weight = original_folded_weight
            mtq.fold_weight(model)
            torch.testing.assert_close(model.folded.weight, expected_folded_weight, rtol=0, atol=0)
            torch.testing.assert_close(model.active.weight, expected_active_weight, rtol=0, atol=0)
            assert not model.folded.weight_quantizer.is_enabled
            assert not model.active.weight_quantizer.is_enabled
            mtq.fold_weight(model)
            torch.testing.assert_close(model.folded.weight, expected_folded_weight, rtol=0, atol=0)
            torch.testing.assert_close(model.active.weight, expected_active_weight, rtol=0, atol=0)
        finally:
            destroy_model_parallel()


@pytest.fixture(scope="module")
def reload_workers():
    if torch.cuda.device_count() < 2:
        pytest.skip("Need at least 2 GPUs")
    workers = DistributedWorkerPool(world_size=2, backend="nccl", teardown_fn=None)
    try:
        yield workers
    finally:
        workers.shutdown()


def test_rank_local_quantizer_reload(reload_workers, tmp_path):
    """Load rank-local state and fold each TP weight without quantizing it twice."""
    reload_workers.run(partial(_test_rank_local_quantizer_reload, tmp_path))
