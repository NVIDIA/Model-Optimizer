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

import json
import time
from contextlib import contextmanager
from datetime import timedelta

import pytest
import torch
from _test_utils.torch.distributed.utils import DistributedWorkerPool
from torch import nn
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor, Shard, distribute_tensor
from torch.distributed.tensor.debug import CommDebugMode
from torch.nn import functional as F

import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.ggml import IQ_FORMAT_REGISTRY
from modelopt.torch.quantization.ggml.common import pin_packed_weight, pinned_packed_weight
from modelopt.torch.quantization.repair import IQRepairPolicy, plan_iq_repair
from modelopt.torch.quantization.repair_formats import _ReplayScales
from modelopt.torch.quantization.utils import enable_weight_access_and_writeback


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Native padded IQ replay requires CUDA")
@pytest.mark.parametrize("name", sorted(IQ_FORMAT_REGISTRY))
def test_iq_padded_native_repair_lifecycle(name):
    torch.manual_seed(107)
    model = nn.Linear(300, 4, bias=False, device="cuda", dtype=torch.bfloat16)
    x = torch.randn(8, 300, device="cuda", dtype=torch.bfloat16)
    quant_cfg = _iq("weight_quantizer", name)
    quant_cfg["cfg"]["block_sizes"] = {-1: 256}
    mtq.quantize(model, _config([quant_cfg]), lambda m: m(x))
    baseline = model(x).detach()
    policy = IQRepairPolicy()
    try:
        fields = policy.prepare(model)
        torch.testing.assert_close(model(x), baseline, rtol=0, atol=0)
        optimizer = torch.optim.SGD(fields, lr=10)
        model(x).float().square().mean().backward()
        assert fields and any(
            field.grad is not None and field.grad.count_nonzero() for field in fields
        )
        optimizer.step()
        policy.project_()
        expected = model(x).detach()
        assert not torch.equal(expected, baseline)
        state = model.weight_quantizer._local_repair_state
        decoded = IQ_FORMAT_REGISTRY[name].dequantize(
            state.native_payload(), torch.tensor(state.codec_shape), dtype=model.weight.dtype
        )
        torch.testing.assert_close(expected, F.linear(x, decoded[:, :300]), rtol=0, atol=0)
        policy.finalize_()
    finally:
        policy.close()
    torch.testing.assert_close(model(x), expected, rtol=0, atol=0)


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


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required for finalization memory"
)
@torch.no_grad()
def test_native_finalization_peak_memory_and_outputs(record_property, capsys):
    """Measure the real three-projection transaction, including decode, pinning and CPU backups."""
    torch.manual_seed(106)
    rows, columns, projections = 4096, 14336, 3
    dtype, device = torch.float16, "cuda"
    fmt = IQ_FORMAT_REGISTRY["iq2_s"]
    model = nn.ModuleList(
        [
            nn.Linear(columns, rows, bias=False, device=device, dtype=dtype)
            for _ in range(projections)
        ]
    )
    mtq.quantize(model, {**_config([_iq("*weight_quantizer", "iq2_s")]), "algorithm": None})
    blocks = rows * columns // fmt.block_size
    for module in model:
        native = torch.randint(
            0,
            256,
            (rows, columns // fmt.block_size, fmt.block_bytes),
            device=device,
            dtype=torch.uint8,
        )
        native = fmt.replace_payload_scales(
            native.reshape(-1, fmt.block_bytes),
            torch.full((blocks,), 2**-8, device=device, dtype=dtype),
        ).reshape_as(native)
        module.weight.copy_(fmt.dequantize(native, torch.tensor(module.weight.shape), dtype=dtype))
        pin_packed_weight(module.weight_quantizer, fmt.name, native, source=module.weight)

    packed_bytes = projections * blocks * fmt.block_bytes
    source_key_bytes = projections * blocks * 24
    decoded_bytes = rows * columns * torch.empty((), dtype=dtype).element_size()
    finite_check_budget_bytes = rows * columns
    scratch_bytes = 64 * 1024**2
    budget_bytes = (
        packed_bytes + source_key_bytes + decoded_bytes + finite_check_budget_bytes + scratch_bytes
    )
    raw_rollback_cpu_bytes = projections * decoded_bytes
    inputs = torch.randn(2, columns, device=device, dtype=dtype)
    policy = IQRepairPolicy()
    try:
        fields = policy.prepare(model)
        for parameter in fields:
            parameter.add_(0.01)
        policy.project_()
        expected = [module(inputs).detach() for module in model]
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        wall_start = time.perf_counter()
        start.record()
        policy.finalize_()
        end.record()
        torch.cuda.synchronize()
        wall_ms = (time.perf_counter() - wall_start) * 1000
        peak_bytes = torch.cuda.max_memory_allocated() - baseline
        metrics = {
            "finalize_device": torch.cuda.get_device_name(),
            "finalize_dtype": str(dtype),
            "finalize_projection_shape": [rows, columns],
            "finalize_projections": projections,
            "finalize_cuda_ms": start.elapsed_time(end),
            "finalize_wall_ms": wall_ms,
            "finalize_baseline_allocated_bytes": baseline,
            "finalize_extra_peak_bytes": peak_bytes,
            "finalize_extra_budget_bytes": budget_bytes,
            "finalize_packed_publication_bytes": packed_bytes,
            "finalize_source_key_publication_bytes": source_key_bytes,
            "finalize_largest_decoded_bytes": decoded_bytes,
            "finalize_finite_check_budget_bytes": finite_check_budget_bytes,
            "finalize_scratch_budget_bytes": scratch_bytes,
            "finalize_raw_rollback_cpu_bytes_computed": raw_rollback_cpu_bytes,
        }
        for name, value in metrics.items():
            record_property(name, value)
        with capsys.disabled():
            print("Bypass finalize metrics: " + json.dumps(metrics, sort_keys=True), flush=True)
        assert peak_bytes <= budget_bytes
        for module, reference in zip(model, expected, strict=True):
            torch.testing.assert_close(module(inputs), reference, rtol=0, atol=0)
    finally:
        policy.close()


def _repair_expert_sharded_dtensor_job(rank, size):
    # Transformers is optional; only the Gloo workers construct its fused expert bank.
    from _test_utils.torch.transformers_models import get_tiny_mixtral
    from transformers.models.mixtral.modeling_mixtral import MixtralExperts

    from modelopt.torch.quantization.plugins.huggingface import _QuantFusedExperts

    mesh = init_device_mesh("cpu", (size,))
    for total_experts, local_experts, local_counts in (
        (4, True, (2, 2)),
        (4, False, (2, 2)),
        (1, False, (1, 0)),
    ):
        model = get_tiny_mixtral(
            hidden_size=256,
            intermediate_size=16,
            num_hidden_layers=1,
            num_local_experts=total_experts,
            num_experts_per_tok=min(2, total_experts),
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
            bank.num_experts = total_experts // size
            start = rank * bank.num_experts
            for name in ("gate_up_proj_weight_quantizers", "down_proj_weight_quantizers"):
                quantizers = getattr(bank, name)
                setattr(bank, name, nn.ModuleList(quantizers[start : start + bank.num_experts]))
        first = bank.gate_up_proj
        assert isinstance(first, DTensor) and first.placements == (Shard(0),)
        assert first.shape[0] == total_experts and first.to_local().shape[0] == local_counts[rank]
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
                total_experts,
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


def _iq_fsdp2_finalization_rollback_job(rank, size):
    group = torch.distributed.new_group(backend="nccl", timeout=timedelta(seconds=10))
    mesh = DeviceMesh.from_group(group, "cuda")
    device = torch.device(f"cuda:{rank}")

    def synchronize(error):
        records: list[str | None] = [None] * size
        torch.distributed.all_gather_object(records, error, group=group)
        errors = sorted({record for record in records if record})
        if errors:
            raise ValueError("; ".join(errors))

    cases = (
        (name, stage)
        for name in ("iq1_m", "iq2_s")
        for stage in ("all_writeback", "rank0_writeback", "rank0_native")
    )
    try:
        for name, stage in cases:
            torch.manual_seed(109)
            model = nn.Linear(256, 4, bias=False, device=device)
            mtq.quantize(model, {**_config([_iq("weight_quantizer", name)]), "algorithm": None})
            quantizer = model.weight_quantizer
            quantizer.pre_quant_scale = torch.linspace(0.5, 1.5, 256, device=device)
            model = fully_shard(model, mesh=mesh, reshard_after_forward=True)
            inputs = torch.randn(3, 256, device=device)
            baseline = model(inputs).detach()
            model.reshard()
            weight = model._parameters["weight"]
            assert isinstance(weight, DTensor) and weight.placements == (Shard(0),)
            original_shard = weight.to_local().detach().clone()
            assert original_shard.shape == (2, 256)
            policy = IQRepairPolicy()
            plan = plan_iq_repair([model])
            try:
                with enable_weight_access_and_writeback(model, model, writeback=False):
                    fields = policy.prepare_plan(plan, 0)
                with torch.no_grad():
                    for field in fields:
                        field.add_(0.25)
                with enable_weight_access_and_writeback(model, model, writeback=False):
                    expected = model(inputs).detach()
                assert not torch.equal(expected, baseline)
                state = quantizer._local_repair_state
                native = state.native_payload().detach().clone()
                masters = [field.detach().clone() for field in fields]
                buffers = {
                    key: value.detach().clone() for key, value in quantizer.state_dict().items()
                }
                flags = quantizer._enable_pre_quant_scale, quantizer._rotate
                failure = RuntimeError("FSDP2 publication failure: " + stage)
                armed = True
                if stage == "rank0_native":
                    native_payload = state.native_payload

                    def fail_native_once():
                        nonlocal armed
                        payload = native_payload()
                        if rank == 0 and armed:
                            armed = False
                            raise failure
                        return payload

                    state.native_payload = fail_native_once

                @contextmanager
                def access(*, writeback):
                    nonlocal armed
                    with enable_weight_access_and_writeback(model, model, writeback=writeback):
                        yield
                    if writeback and armed and stage != "rank0_native":
                        armed = False
                        assert not torch.equal(
                            model._parameters["weight"].to_local(), original_shard
                        )
                        if stage == "all_writeback" or rank == 0:
                            raise failure

                with pytest.raises((RuntimeError, ValueError)) as error:
                    policy.finalize_(
                        weight_access=access,
                        synchronize_error=synchronize,
                    )
                if stage == "all_writeback" or rank == 0:
                    assert error.value is failure
                else:
                    assert isinstance(error.value, ValueError)
                restored = model._parameters["weight"]
                assert isinstance(restored, DTensor) and restored.placements == (Shard(0),)
                torch.testing.assert_close(restored.to_local(), original_shard, rtol=0, atol=0)
                assert quantizer._local_repair_state is state
                assert (quantizer._enable_pre_quant_scale, quantizer._rotate) == flags
                after = quantizer.state_dict()
                assert after.keys() == buffers.keys()
                for key, value in buffers.items():
                    torch.testing.assert_close(after[key], value, rtol=0, atol=0)
                for field, reference in zip(fields, masters, strict=True):
                    torch.testing.assert_close(field, reference, rtol=0, atol=0)

                policy.finalize_(
                    weight_access=access,
                    synchronize_error=synchronize,
                )
                torch.testing.assert_close(model(inputs), expected, rtol=0, atol=0)
                torch.testing.assert_close(
                    pinned_packed_weight(quantizer, name), native, rtol=0, atol=0
                )
                model.reshard()
                decoded = IQ_FORMAT_REGISTRY[name].dequantize(
                    native, torch.tensor((4, 256)), dtype=original_shard.dtype
                )
                torch.testing.assert_close(
                    model._parameters["weight"].to_local(),
                    decoded[2 * rank : 2 * rank + 2],
                    rtol=0,
                    atol=0,
                )
                assert not hasattr(quantizer, "_local_repair_state")
                assert model._parameters["weight"].requires_grad
            finally:
                policy.close()
    finally:
        torch.distributed.destroy_process_group(group)


@pytest.mark.timeout(300)
@pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="FSDP2 IQ rollback requires two CUDA devices"
)
def test_iq_fsdp2_finalization_rolls_back_shards_and_can_retry():
    pool = DistributedWorkerPool(world_size=2, backend="nccl")
    try:
        pool.job_timeout = 180
        pool.failure_grace = 5
        pool.run(_iq_fsdp2_finalization_rollback_job)
    finally:
        pool.shutdown()
