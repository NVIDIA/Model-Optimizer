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

"""Native IQ field gradients, frozen downstream projections, and ordinary persistence."""

import pytest
import torch
from _test_utils.torch.distributed.utils import DistributedWorkerPool
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard, distribute_tensor
from torch.distributed.tensor.debug import CommDebugMode
from torch.nn import functional as F

import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.ggml import IQ_FORMAT_REGISTRY
from modelopt.torch.quantization.repair import IQRepairPolicy
from modelopt.torch.quantization.repair_formats import _ReplayScales


def _config(entries):
    return {"quant_cfg": [{"quantizer_name": "*", "enable": False}, *entries], "algorithm": "max"}


def _iq(path, name):
    return {"quantizer_name": path, "cfg": {"num_bits": name, "backend": "ggml"}}


class _Expert(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Linear(256, 256, bias=False)
        self.up = nn.Linear(256, 256, bias=False)
        self.down = nn.Linear(256, 256, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class _MixedMoE(nn.Module):
    def __init__(self):
        super().__init__()
        self.router = nn.Linear(256, 2, bias=False)
        self.experts = nn.ModuleList([_Expert(), _Expert()])
        self.shared = _Expert()

    def forward(self, x):
        scores = self.router(x).softmax(-1)
        return self.shared(x) + sum(
            expert(x) * scores[:, i : i + 1] for i, expert in enumerate(self.experts)
        )


@pytest.mark.parametrize("frozen_format", ["fp8", "nvfp4", "nvfp4_compressed"])
def test_iq_scales_backpropagate_through_frozen_projection(frozen_format):
    if frozen_format.startswith("nvfp4") and not torch.cuda.is_available():
        pytest.skip("Real NVFP4 forward/backward requires a CUDA allocation")
    device = "cuda" if frozen_format.startswith("nvfp4") else "cpu"
    torch.manual_seed(23)
    model = _MixedMoE().to(device)
    x = torch.randn(2, 256, device=device)
    frozen_cfg = (
        {"num_bits": (4, 3), "axis": None}
        if frozen_format == "fp8"
        else {"num_bits": (2, 1), "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)}}
    )
    mtq.quantize(
        model,
        _config(
            [
                _iq("shared.*weight_quantizer", "iq2_s"),
                _iq("experts.*.gate.weight_quantizer", "iq1_m"),
                _iq("experts.*.up.weight_quantizer", "iq1_m"),
                {"quantizer_name": "experts.*.down.weight_quantizer", "cfg": frozen_cfg},
            ]
        ),
        lambda m: m(x),
    )
    if frozen_format == "nvfp4_compressed":
        mtq.compress(
            model, mtq.CompressConfig(compress={"default": False, "experts.*.down*": True})
        )
    frozen = {
        n: v.detach().clone()
        for n, v in model.state_dict().items()
        if "router" in n or ("experts" in n and ".down." in n)
    }
    policy = IQRepairPolicy()
    try:
        fields = policy.prepare(model)
        expected = model.shared(x)
        scores = model.router(x).detach().softmax(-1)
        for i, expert in enumerate(model.experts):
            # The oracle uses the actual quantized frozen down weight and ordinary differentiable linear.
            weight = expert.down.weight_quantizer(expert.down.weight).detach()
            hidden = F.silu(expert.gate(x)) * expert.up(x)
            expected = expected + F.linear(hidden, weight) * scores[:, i : i + 1]
        expected.square().sum().backward()
        gradients = [p.grad.detach().clone() for p in fields]
        for p in fields:
            p.grad = None
        model(x).square().sum().backward()
        for parameter, reference in zip(fields, gradients):
            torch.testing.assert_close(parameter.grad, reference, rtol=2e-5, atol=2e-7)
        assert any(p.grad.count_nonzero() for p in fields)
        assert all(p.grad is None for expert in model.experts for p in expert.down.parameters())
        optimizer = torch.optim.Adam(fields, lr=0.001)
        optimizer.step()
        assert set(optimizer.state) == set(fields)
        policy.finalize_()
    finally:
        policy.close()
    after = model.state_dict()
    for name, value in frozen.items():
        torch.testing.assert_close(after[name], value, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for peak memory")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_native_replay_peak_memory_and_vjp(dtype, record_property):
    """Measure fallback replay for a 4096-by-14336 projection, excluding packing/decoding."""
    torch.manual_seed(105)
    fmt = IQ_FORMAT_REGISTRY["iq1_s"]
    elements = 4096 * 14336
    shape = (
        elements // fmt.block_size,
        fmt.block_size // fmt.int8_group_size,
        fmt.int8_group_size,
    )
    steps = torch.randn(shape[:-1], device="cuda", requires_grad=True)
    _ReplayScales.apply(
        torch.ones(1, 1, shape[-1], device="cuda", dtype=torch.int8),
        torch.ones(1, 1, device="cuda"),
        dtype,
    )
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    levels = torch.randint(-16, 16, shape, device="cuda", dtype=torch.int8)
    start.record()
    output = _ReplayScales.apply(levels, steps, dtype)
    end.record()
    torch.cuda.synchronize()
    peak_bytes = torch.cuda.max_memory_allocated() - baseline
    budget_bytes = elements * (1 + output.element_size()) + 8 * 262144
    record_property("replay_dtype", str(dtype))
    record_property("replay_elements", elements)
    record_property("replay_forward_ms", start.elapsed_time(end))
    record_property("replay_forward_peak_bytes", peak_bytes)
    record_property("replay_forward_budget_bytes", budget_bytes)
    assert peak_bytes <= budget_bytes

    reference_steps = steps.detach().clone().requires_grad_()
    expected = (levels.float() * reference_steps.unsqueeze(-1)).to(dtype)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    gradient = torch.randn_like(output)
    (actual_vjp,) = torch.autograd.grad(output, steps, gradient)
    (expected_vjp,) = torch.autograd.grad(expected, reference_steps, gradient)
    torch.testing.assert_close(actual_vjp, expected_vjp, rtol=0, atol=0)


def _repair_expert_sharded_dtensor_job(rank, size):
    # Transformers is optional; only the Gloo workers construct its fused expert bank.
    from _test_utils.torch.transformers_models import get_tiny_mixtral
    from transformers.models.mixtral.modeling_mixtral import MixtralExperts

    from modelopt.torch.quantization.plugins.huggingface import _QuantFusedExperts

    mesh = init_device_mesh("cpu", (size,))
    for local_experts in (True, False):
        model = get_tiny_mixtral(
            hidden_size=256,
            intermediate_size=16,
            num_hidden_layers=1,
            num_local_experts=4,
            num_experts_per_tok=2,
            vocab_size=32,
            dtype=torch.float32,
        )
        bank = next(module for module in model.modules() if isinstance(module, MixtralExperts))
        mtq.quantize(
            bank,
            {
                **_config([_iq("*gate_up_proj_weight_quantizers*", "iq1_s")]),
                "algorithm": None,
            },
        )
        assert isinstance(bank, _QuantFusedExperts)
        for name in ("gate_up_proj", "down_proj"):
            weight = getattr(bank, name)
            setattr(bank, name, nn.Parameter(distribute_tensor(weight.detach(), mesh, [Shard(0)])))
        if local_experts:
            bank.num_experts = 4 // size
            start = rank * bank.num_experts
            for name in ("gate_up_proj_weight_quantizers", "down_proj_weight_quantizers"):
                quantizers = getattr(bank, name)
                setattr(bank, name, nn.ModuleList(quantizers[start : start + bank.num_experts]))
        first = bank.gate_up_proj
        assert isinstance(first, DTensor) and first.placements == (Shard(0),)
        assert first.shape[0] == 4 and first.to_local().shape[0] == 4 // size
        assert len(bank.gate_up_proj_weight_quantizers) == bank.num_experts
        assert len(bank.down_proj_weight_quantizers) == bank.num_experts
        before = {
            name: getattr(bank, name).to_local().detach().clone()
            for name in ("gate_up_proj", "down_proj")
        }
        flags = {parameter: parameter.requires_grad for parameter in bank.parameters()}
        quantizers = list(bank.iter_weight_quantizers_for_calibration())
        quantizer_flags = [(q._if_calib, q._enable_pre_quant_scale, q._rotate) for q in quantizers]
        policy = IQRepairPolicy()
        try:
            refused = False
            with CommDebugMode() as comms:
                try:
                    fields = policy.prepare(bank)
                except ValueError:
                    refused = True
            assert refused == (not local_experts), (
                local_experts,
                refused,
                dict(comms.get_comm_counts()),
            )
            assert comms.get_total_counts() == 0, dict(comms.get_comm_counts())
            if refused:
                assert all(parameter.requires_grad == flag for parameter, flag in flags.items())
                assert [
                    (q._if_calib, q._enable_pre_quant_scale, q._rotate) for q in quantizers
                ] == quantizer_flags
                assert all(
                    not hasattr(q, "_local_repair_state") and "_fake_quantize" not in q.__dict__
                    for q in quantizers
                )
                for name, expected in before.items():
                    torch.testing.assert_close(
                        getattr(bank, name).to_local(), expected, rtol=0, atol=0
                    )
                continue

            inputs = torch.randn(2, 256)
            projected = [
                quantizer(weight)
                for weight, quantizer in bank.iter_weights_for_calibration()
                if quantizer.is_enabled
            ]
            before_replay = torch.stack([weight.detach() for weight in projected])
            loss = sum(F.linear(inputs, weight).square().mean() for weight in projected)
            loss.backward()
            assert fields and any(
                field.grad is not None and field.grad.count_nonzero() for field in fields
            )
            assert all(parameter.grad is None for parameter in bank.parameters())
            optimizer = torch.optim.SGD(fields, lr=10)
            optimizer.step()
            policy.project_()
            expected = torch.stack(
                [
                    quantizer(bank.get_expert_weight("gate_up_proj", index)).detach()
                    for index, quantizer in enumerate(bank.gate_up_proj_weight_quantizers)
                ]
            )
            assert not torch.equal(expected, before_replay)
            policy.finalize_()
            assert bank.gate_up_proj is first
            assert not torch.equal(first.to_local(), before["gate_up_proj"])
            torch.testing.assert_close(first.to_local(), expected, rtol=0, atol=0)
            torch.testing.assert_close(
                bank.down_proj.to_local(), before["down_proj"], rtol=0, atol=0
            )
        finally:
            policy.close()


@pytest.mark.timeout(300)
def test_iq_repair_requires_local_expert_dtensor_ownership():
    pytest.importorskip("transformers")
    pool = DistributedWorkerPool(world_size=2, backend="gloo")
    try:
        pool.job_timeout = 180
        pool.run(_repair_expert_sharded_dtensor_job)
    finally:
        pool.shutdown()
