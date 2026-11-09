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


"""Topology-native HF DCP preserves physical fields and pipeline key unions."""

import copy
import json
import os
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor, Shard, distribute_tensor

import modelopt.torch.distill as mtd
import modelopt.torch.quantization as mtq
from modelopt.torch.distill.plugins.transformers import TransformersLocalDistillationAdapter
from modelopt.torch.distill.plugins.transformers.checkpoint import publish, restore
from modelopt.torch.quantization.nn import TensorQuantizer
from modelopt.torch.quantization.repair_partition import prepare_packed_weights_for_tp


@pytest.fixture
def checkpoint_context():
    if int(os.environ.get("WORLD_SIZE", 1)) != 4 or not torch.cuda.is_available():
        pytest.skip("Launch on four GPU ranks.")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    if not dist.is_initialized():
        dist.init_process_group("nccl", timeout=timedelta(seconds=120))


def _local(value):
    return value.to_local() if isinstance(value, DTensor) else value


def _snapshot(model):
    return {
        name: _local(value).detach().clone()
        for name, value in model.state_dict().items()
        if isinstance(value, torch.Tensor)
    }


def _assert_snapshot(model, expected):
    for name, value in model.state_dict().items():
        if name in expected:
            torch.testing.assert_close(_local(value), expected[name], rtol=0, atol=0, msg=name)


def _directory(tmp_path, suffix):
    value = [str(tmp_path / suffix) if dist.get_rank() == 0 else None]
    dist.broadcast_object_list(value, src=0)
    return Path(value[0])


def _quantize(model, kind):
    spec = (
        {"num_bits": (4, 3), "axis": None}
        if kind == "fp8"
        else {"num_bits": (2, 1), "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)}}
    )
    return mtq.quantize(
        model,
        {
            "quant_cfg": [
                {"quantizer_name": "*", "enable": False},
                {"quantizer_name": "*weight_quantizer*", "cfg": spec},
            ],
            "algorithm": "max",
        },
    )


@pytest.mark.parametrize("kind", ["fp8", "nvfp4"])
@pytest.mark.timeout(180)
def test_hf_dcp_tp_fsdp_preserves_rank_fields(checkpoint_context, tmp_path, kind):
    torch.manual_seed(83)
    mesh = init_device_mesh("cuda", (2, 2), mesh_dim_names=("dp", "tp"))
    model = _quantize(
        nn.Sequential(nn.Linear(64, 64, bias=False), nn.Linear(64, 64, bias=False))
        .cuda()
        .bfloat16(),
        kind,
    )
    mtq.compress(model)
    prepare_packed_weights_for_tp(model)
    # This fixture checkpoints physical column/row TP ownership without running a
    # forward. Torch's recursive Linear styles also visit quantizer children.
    for layer, dimension in zip(model, (0, 1), strict=True):
        weight = layer._parameters["weight"]
        distributed = nn.Parameter(
            distribute_tensor(weight.detach(), mesh["tp"], [Shard(dimension)]),
            requires_grad=weight.requires_grad,
        )
        dict.__setitem__(layer._parameters, "weight", distributed)
    adapter = TransformersLocalDistillationAdapter(mesh)
    adapter.prepare(model, model, [], mtd.LocalDistillationConfig())
    for layer, expected_shape in zip(model, ((32, 64), (64, 32)), strict=True):
        assert isinstance(layer._parameters["weight"], DTensor)
        assert tuple(layer._parameters["weight"].to_local().shape) == expected_shape
        assert tuple(layer.weight_quantizer._native_weight_meta["source_shape"]) == expected_shape
    # Disabled activation fields can share calibration storage without changing weight math.
    first = model[0].input_quantizer
    first._set_buffer("_amax", torch.tensor(3.0 + dist.get_rank(), device="cuda"))
    model[1].input_quantizer = first
    fully_shard(model, mesh=mesh["dp"])
    expected = _snapshot(model)
    root = _directory(tmp_path, "tp-" + kind)
    context = adapter.context
    config = mtd.LocalDistillationConfig(output_dir=str(root))
    result = publish(model, context, config, {"test": kind})
    assert result["checkpoint"] == str(root)
    assert (root / "COMMITTED.json").is_file()
    # The public runner adds publication timing to this report after committing.
    Path(result["metrics"]).write_text(
        json.dumps({"test": kind, "runtime": {"publish_seconds": 0.25}}, indent=2) + "\n"
    )
    unexpected = Path(result["metrics"]).parent / "unlisted.json"
    if dist.get_rank() == 0:
        unexpected.write_text("{}\n")
    dist.barrier()
    with pytest.raises(ValueError, match="missing or extra"):
        restore(model, context, root)
    _assert_snapshot(model, expected)
    if dist.get_rank() == 0:
        unexpected.unlink()
    dist.barrier()
    # Breaking a saved module tie must fail collectively before native weight loading.
    model[1].input_quantizer = copy.deepcopy(first)
    with pytest.raises(ValueError, match="tied-quantizer-module ownership"):
        restore(model, context, root)
    _assert_snapshot(model, expected)
    model[1].input_quantizer = first
    native_properties = {
        name: copy.deepcopy(module._native_weight_meta)
        for name, module in model.named_modules()
        if isinstance(module, TensorQuantizer) and hasattr(module, "_native_weight_meta")
    }
    with torch.no_grad():
        for parameter in model.parameters():
            _local(parameter).zero_()
        for module in model.modules():
            if isinstance(module, TensorQuantizer) and "_amax" in module._buffers:
                module._buffers["_amax"].zero_()
            if isinstance(module, TensorQuantizer):
                for name, buffer in module._buffers.items():
                    if name.startswith("_native_weight_"):
                        buffer.zero_()
                if hasattr(module, "_native_weight_meta"):
                    module._native_weight_meta = None
    assert restore(model, context, root) is model
    _assert_snapshot(model, expected)
    for name, metadata in native_properties.items():
        assert model.get_submodule(name)._native_weight_meta == metadata
    assert model[0].input_quantizer._amax is model[1].input_quantizer._amax
    assert model[0].input_quantizer is model[1].input_quantizer
    with pytest.raises(ValueError, match="new or empty"):
        publish(model, context, config, {})
    # A corrupt committed shard must be rejected collectively before any tensor changes.
    before_failure = _snapshot(model)
    if dist.get_rank() == 0:
        manifest = json.loads((root / "COMMITTED.json").read_text())
        relative = next(name for name in manifest["files"] if name.endswith(".distcp"))
        with (root / relative).open("ab") as stream:
            stream.write(b"corrupt")
    dist.barrier()
    with pytest.raises(ValueError, match="integrity"):
        restore(model, context, root)
    _assert_snapshot(model, before_failure)


@pytest.mark.timeout(180)
def test_hf_dcp_pipeline_stages_save_union_of_distinct_keys(checkpoint_context, tmp_path):
    torch.manual_seed(89)
    mesh = init_device_mesh("cuda", (2, 2), mesh_dim_names=("pp", "dp"))
    stage = mesh["pp"].get_local_rank()
    width = 32 if stage == 0 else 48
    model = nn.Sequential()
    model.add_module("layers", nn.Sequential())
    model.layers.add_module(str(stage), nn.Linear(width, width, bias=False))
    model = _quantize(model.cuda().bfloat16(), "nvfp4")
    fully_shard(model, mesh=mesh["dp"])
    mtq.compress(model)
    expected = _snapshot(model)
    root = _directory(tmp_path, "pipeline")
    context = SimpleNamespace(device_mesh=mesh, process_group=None, moe_mesh=None)
    publish(model, context, mtd.LocalDistillationConfig(output_dir=str(root)), {"stage": stage})
    relocated = root.with_name("relocated-pipeline")
    if dist.get_rank() == 0:
        os.replace(root, relocated)
    dist.barrier()
    root = relocated
    metadata = FileSystemReader(root / "weights").read_metadata()
    names = metadata.state_dict_metadata
    assert any("layers.0.weight" in name for name in names)
    assert any("layers.1.weight" in name for name in names)
    with torch.no_grad():
        for parameter in model.parameters():
            _local(parameter).zero_()
    restore(model, context, root)
    _assert_snapshot(model, expected)
    failed = root.with_name("failed-publication")
    with pytest.raises(ValueError, match="metadata failed"):
        publish(
            model,
            context,
            mtd.LocalDistillationConfig(output_dir=str(failed)),
            {"invalid": object()},
        )
    assert not failed.exists()
    with pytest.raises(ValueError, match=r"incomplete|unpublished"):
        restore(model, context, failed)
