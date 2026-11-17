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

"""Native IQ payload placement, mesh adoption and real four-GPU projection checks."""

import copy
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
from torch.nn import functional as F

import modelopt.torch.distill as mtd
import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.config import QuantizerAttributeConfig
from modelopt.torch.quantization.ggml import IQ_FORMAT_REGISTRY
from modelopt.torch.quantization.ggml.common import pin_packed_weight
from modelopt.torch.quantization.nn import TensorQuantizer


@pytest.mark.parametrize("name", sorted(IQ_FORMAT_REGISTRY))
@pytest.mark.parametrize("dim", [0, 1])
def test_native_iq_partition_preserves_pinned_projection(name, dim):
    from modelopt.torch.quantization.repair_partition import (
        NativeWeightPartition,
        native_weight_partition,
    )

    torch.manual_seed(59)
    fmt = IQ_FORMAT_REGISTRY[name]
    weight = torch.randn(4, 512)
    importance = 1 + torch.rand(512) if fmt.weighted else None
    packed, shape = fmt.quantize(
        weight, **({"importance": importance} if importance is not None else {})
    )
    quantizer = TensorQuantizer(
        QuantizerAttributeConfig(num_bits=name, backend="ggml", block_sizes={-1: fmt.block_size})
    )
    search = fmt.payload_scales(packed.reshape(-1, fmt.block_bytes)) if fmt.payload_scales else None
    pin_packed_weight(quantizer, name, packed, importance, source=weight, search_scales=search)
    expected = fmt.dequantize(packed, shape, dtype=torch.float32)
    view = NativeWeightPartition((4, 512), dim, 1, 2, split_factor=2 if dim == 0 else 1)
    local = view.select(weight)
    fields, _, _ = native_weight_partition(quantizer, view, source=local)
    for key, value in fields.items():
        quantizer._set_buffer(key, value) if isinstance(value, torch.Tensor) else setattr(
            quantizer, key, value
        )
    x = torch.randn(3, local.shape[-1], requires_grad=True)
    actual = F.linear(x, quantizer(local))
    reference = F.linear(x, view.select(expected))
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    actual.square().sum().backward()
    actual_grad = x.grad.clone()
    x.grad = None
    reference.square().sum().backward()
    torch.testing.assert_close(actual_grad, x.grad, rtol=0, atol=0)
    before = {n: v.clone() for n, v in quantizer.state_dict().items()}
    invalid = NativeWeightPartition((4, 512), 1, 0, 2 * (512 // fmt.block_size))
    with pytest.raises(ValueError):
        native_weight_partition(quantizer, invalid)
    assert all(torch.equal(v, before[n]) for n, v in quantizer.state_dict().items())


def test_packed_frozen_projection_staging_and_restore(tmp_path):
    import modelopt.torch.opt as mto
    from modelopt.torch.quantization.qtensor import FP8QTensor, QTensorWrapper
    from modelopt.torch.quantization.repair_partition import prepare_packed_weights_for_tp

    torch.manual_seed(60)
    model = mtq.quantize(
        torch.nn.Sequential(torch.nn.Linear(256, 16, bias=False)),
        {"quant_cfg": [{"quantizer_name": "*", "enable": False}], "algorithm": None},
    )
    layer = model[0]
    native, scale = FP8QTensor.quantize(layer.weight.detach(), axis=0)
    layer.weight_quantizer.set_from_attribute_config(
        QuantizerAttributeConfig(num_bits=(4, 3), axis=0, fake_quant=False)
    )
    layer.weight_quantizer._set_buffer("_scale", scale)
    layer.weight = QTensorWrapper(native)
    decoded = native.dequantize(scale=scale)
    prepare_packed_weights_for_tp(model)
    x = torch.randn(3, 256, requires_grad=True)
    expected = F.linear(x, decoded)
    actual = model(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.square().sum().backward()
    grad = x.grad.clone()
    x.grad = None
    expected.square().sum().backward()
    torch.testing.assert_close(grad, x.grad, rtol=0, atol=0)
    assert not layer.weight.requires_grad
    path = tmp_path / "staged.pt"
    mto.save(model, path)
    restored = mto.restore(torch.nn.Sequential(torch.nn.Linear(256, 16, bias=False)), path)
    torch.testing.assert_close(restored(x), expected, rtol=0, atol=0)
    assert torch.equal(
        restored[0].weight_quantizer._native_weight_codes, native._quantized_data.view(torch.uint8)
    )


@pytest.mark.timeout(60)
def test_native_mesh_adoption_preserves_forward_and_hooks(tmp_path):
    from torch.distributed.device_mesh import init_device_mesh
    from transformers import LlamaConfig, LlamaForCausalLM
    from transformers.distributed.utils import TransformersDeviceMesh

    from modelopt.torch.distill.plugins.transformers import TransformersBypassAdapter

    if dist.is_initialized():
        pytest.skip("This single-process adoption case owns its isolated Gloo group")
    dist.init_process_group(
        "gloo",
        init_method=(tmp_path / "rendezvous").as_uri(),
        rank=0,
        world_size=1,
        timeout=timedelta(seconds=30),
    )
    try:
        manager = TransformersDeviceMesh(
            init_device_mesh(
                "cpu", (1, 1, 1, 1), mesh_dim_names=("pp", "dp_replicate", "fsdp", "tp")
            ),
            init_device_mesh(
                "cpu", (1, 1, 1, 1), mesh_dim_names=("pp", "dp_replicate", "efsdp", "ep")
            ),
        )
        config = LlamaConfig(
            vocab_size=32,
            hidden_size=256,
            intermediate_size=256,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            use_cache=False,
        )
        config._attn_implementation = "eager"
        teacher = LlamaForCausalLM(config).eval()
        batch = {"input_ids": torch.randint(3, 32, (1, 4))}
        adapter = TransformersBypassAdapter(manager)
        student = mtq.quantize(
            copy.deepcopy(teacher),
            {
                "quant_cfg": [
                    {"quantizer_name": "*", "enable": False},
                    *[
                        {"quantizer_name": pattern, "cfg": {"num_bits": "iq2_s", "backend": "ggml"}}
                        for pattern in adapter.weight_patterns(teacher)
                    ],
                ],
                "algorithm": "max",
            },
            lambda m: m(**batch),
        )
        with torch.no_grad():
            expected = student(**batch).logits
        assert adapter.parallelize(student) is student
        hooks = [(m, len(m._forward_hooks), len(m._forward_pre_hooks)) for m in student.modules()]
        assert adapter.parallelize(student) is student
        with torch.no_grad():
            torch.testing.assert_close(student(**batch).logits, expected, rtol=0, atol=0)
        assert (
            mtd.bypass(student, teacher, [batch], {"num_steps": 0}, device_mesh=manager) is student
        )
        assert all(
            len(m._forward_hooks) == after and len(m._forward_pre_hooks) == before
            for m, after, before in hooks
        )
    finally:
        dist.destroy_process_group()


@pytest.fixture
def native_workers(_dist_pools):
    if torch.cuda.device_count() < 4:
        pytest.skip("Native overlapping TP/FSDP/EP projection checks require four GPUs")
    from _test_utils.torch.distributed.utils import (
        DistributedWorkerPool,
        default_worker_teardown,
        get_live_pool,
    )

    return get_live_pool(
        _dist_pools,
        4,
        lambda: DistributedWorkerPool(
            world_size=4, backend="nccl", teardown_fn=default_worker_teardown
        ),
    )


def _native_projection_worker(rank, size, frozen_format):
    from _test_utils.transformers_families import build_model, cases
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import DTensor
    from transformers.distributed.utils import TransformersDeviceMesh

    from modelopt.torch.distill.plugins.transformers import TransformersBypassAdapter

    assert size == 4
    torch.cuda.set_device(rank)
    torch.manual_seed(61)
    record = cases(["qwen3_moe"])[0]
    record["minimum_pattern_prefix"] = 2
    record["config"]["mlp_only_layers"] = [0]
    model = build_model(record, torch.device("cuda", rank))
    batch = {"input_ids": torch.randint(3, 100, (2, 8), device=torch.device("cuda", rank))}
    frozen_cfg = (
        {"num_bits": (4, 3), "axis": None}
        if frozen_format == "fp8"
        else {"num_bits": (2, 1), "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)}}
    )
    mtq.quantize(
        model,
        {
            "quant_cfg": [
                {"quantizer_name": "*", "enable": False},
                {
                    "quantizer_name": "*mlp*weight_quantizer*",
                    "cfg": {"num_bits": "iq1_m", "backend": "ggml"},
                },
                {"quantizer_name": "*mlp.gate.weight_quantizer*", "enable": False},
                {"quantizer_name": "*mlp.experts.down_proj_weight_quantizers*", "cfg": frozen_cfg},
            ],
            "algorithm": "max",
        },
        lambda m: m(**batch),
    )
    with torch.no_grad():
        expected = model(**batch).logits.detach().cpu()
    manager = TransformersDeviceMesh(
        init_device_mesh("cuda", (1, 1, 2, 2), mesh_dim_names=("pp", "dp_replicate", "fsdp", "tp")),
        init_device_mesh(
            "cuda", (1, 1, 1, 4), mesh_dim_names=("pp", "dp_replicate", "efsdp", "ep")
        ),
    )
    adapter = TransformersBypassAdapter(manager)
    adapter.parallelize(model)
    with torch.no_grad():
        actual = model(**batch).logits
        actual = actual.full_tensor() if isinstance(actual, DTensor) else actual
    torch.testing.assert_close(actual.cpu(), expected, rtol=0.015, atol=0.02)


@pytest.mark.parametrize("frozen_format", ["fp8", "nvfp4"])
@pytest.mark.timeout(300)
def test_native_iq_placement_preserves_projection(native_workers, frozen_format):
    native_workers.run(_native_projection_worker, frozen_format)
