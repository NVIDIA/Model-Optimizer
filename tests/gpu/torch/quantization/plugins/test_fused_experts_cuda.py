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

"""One-kernel NVFP4 fake quant of Hugging Face fused MoE experts."""

import pytest
import torch
from _test_utils.torch.transformers_models import get_tiny_qwen3_moe

import modelopt.torch.quantization as mtq
import modelopt.torch.quantization.plugins.huggingface as hf_plugin
import modelopt.torch.quantization.repair_formats as repair_formats
from modelopt.torch.quantization import tensor_quant
from modelopt.torch.quantization.repair import IQRepairPolicy

# As the plugin and the unit tests treat it: grouped_mm is intercepted where transformers has it.
try:
    from transformers.integrations import moe as moe_integration

    HAS_GROUPED_MM = hasattr(moe_integration, "_grouped_linear")
except ImportError:
    HAS_GROUPED_MM = False

pytestmark = pytest.mark.skipif(
    not hasattr(tensor_quant.triton_kernel, "fp4_fake_quant_block"),
    reason="fp4_fake_quant_block requires compute >= 8.9",
)


@pytest.mark.parametrize(
    "impl",
    [
        "eager",
        pytest.param(
            "grouped_mm",
            marks=pytest.mark.skipif(
                not HAS_GROUPED_MM, reason="transformers grouped_mm backend unavailable"
            ),
        ),
    ],
)
@pytest.mark.parametrize("pass_through_bwd", [True, False])
def test_stacked_nvfp4_experts_match_per_expert_quantizers(monkeypatch, pass_through_bwd, impl):
    model = get_tiny_qwen3_moe().cuda().train()
    input_ids = torch.randint(0, model.config.vocab_size, (2, 16), device="cuda")
    mtq.quantize(model, mtq.NVFP4_DEFAULT_CFG, lambda m: m(input_ids))
    # Train on the backend under test; both fake-quantize each projection's experts in one launch.
    model.config._experts_implementation = impl
    experts = [m for m in model.modules() if isinstance(m, hf_plugin._QuantFusedExperts)]
    assert experts
    for module in experts:
        quantizers = [*module.gate_up_proj_weight_quantizers, *module.down_proj_weight_quantizers]
        for idx, quantizer in enumerate(quantizers):
            quantizer._pass_through_bwd = pass_through_bwd
            if idx % 3 == 0:  # weights above amax: saturated block scales, clipped gradients
                quantizer.amax = quantizer.amax * 0.5

    def loss_and_grads(checkpointing):
        if checkpointing:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        else:
            model.gradient_checkpointing_disable()
        model.zero_grad(set_to_none=True)
        loss = model(input_ids, labels=input_ids).loss
        loss.backward()
        return loss.detach(), {n: p.grad for n, p in model.named_parameters() if p.grad is not None}

    calls = []
    stacked_fake_quant = tensor_quant.stacked_nvfp4_fake_quant
    monkeypatch.setattr(
        tensor_quant,
        "stacked_nvfp4_fake_quant",
        lambda *args: calls.append(args) or stacked_fake_quant(*args),
    )
    grouped_calls = []
    if HAS_GROUPED_MM:
        grouped_linear = moe_integration._grouped_linear
        monkeypatch.setattr(
            moe_integration,
            "_grouped_linear",
            lambda *args, **kwargs: grouped_calls.append(1) or grouped_linear(*args, **kwargs),
        )
    results = []
    for checkpointing in (False, True):
        calls.clear()
        results.append(loss_and_grads(checkpointing))
        # Checkpointing reruns the forward in backward.
        assert len(calls) == 2 * len(experts) * (1 + checkpointing)
    assert bool(grouped_calls) == (impl == "grouped_mm"), "the backend under test did not run"
    # Without autograd only the grouped forward, which quantizes every expert, takes the kernel.
    calls.clear()
    with torch.no_grad():
        logits = model(input_ids).logits
    assert len(calls) == (2 * len(experts) if impl == "grouped_mm" else 0)

    monkeypatch.setattr(hf_plugin, "_try_stacked_fake_quant", lambda *args: None)
    with torch.no_grad():
        assert torch.equal(model(input_ids).logits, logits)
    ref_loss, ref_grads = loss_and_grads(checkpointing=False)
    for loss, grads in results:
        assert torch.equal(loss, ref_loss)
        assert grads.keys() == ref_grads.keys()
        for name, grad in grads.items():
            assert torch.equal(grad, ref_grads[name]), name


def test_quantizer_backward_hooks_keep_per_expert_path():
    """A backward hook on a weight quantizer only fires when that quantizer runs as a module."""
    model = get_tiny_qwen3_moe().cuda().train()
    input_ids = torch.randint(0, model.config.vocab_size, (2, 16), device="cuda")
    mtq.quantize(model, mtq.NVFP4_DEFAULT_CFG, lambda m: m(input_ids))
    hook_calls = []
    for module in model.modules():
        if isinstance(module, hf_plugin._QuantFusedExperts):
            for quantizer in module.gate_up_proj_weight_quantizers:
                quantizer.register_full_backward_hook(lambda *args: hook_calls.append(args))

    model(input_ids, labels=input_ids).loss.backward()
    assert hook_calls


@pytest.mark.parametrize("name", ["iq1_m", "iq2_xxs"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_grouped_iq_repair_batches_native_scale_graphs(name, dtype):
    model = (
        get_tiny_qwen3_moe(
            hidden_size=256,
            moe_intermediate_size=256,
            num_hidden_layers=1,
            num_experts=8,
            num_attention_heads=4,
            dtype=dtype,
        )
        .cuda()
        .eval()
    )
    mtq.quantize(
        model,
        {
            "quant_cfg": [
                {"quantizer_name": "*", "enable": False},
                {
                    "quantizer_name": "*experts.*weight_quantizers.*",
                    "cfg": {"num_bits": name, "backend": "ggml"},
                },
            ],
            "algorithm": "max",
        },
    )
    experts = next(m for m in model.modules() if isinstance(m, hf_plugin._QuantFusedExperts))
    policy = IQRepairPolicy()
    handle = None
    try:
        policy.prepare(experts)
        quantizers = experts.gate_up_proj_weight_quantizers
        states = [q._local_repair_state for q in quantizers]
        with torch.no_grad():
            for state in states:
                state.global_delta.uniform_(-0.2, 0.2)
                state.local_delta.uniform_(-0.2, 0.2)
                state.global_base[0] = 0
        fields = tuple(p for state in states for p in state.parameters())
        weight = experts.gate_up_proj
        reference = torch.stack([q(w) for q, w in zip(quantizers, weight)])
        gradient = torch.randn_like(reference)
        reference_grads = torch.autograd.grad(reference, fields, gradient)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as trace:
            actual = experts._quantize_grouped_weight(weight, quantizers)
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        actual_grads = torch.autograd.grad(actual, fields, gradient)
        for actual_grad, reference_grad in zip(actual_grads, reference_grads):
            torch.testing.assert_close(actual_grad, reference_grad, rtol=0, atol=0)
        # Project a bank together: per-expert graphs make this path launch-bound at scale.
        exponential_calls = sum(e.count for e in trace.key_averages() if e.key == "aten::exp")
        assert exponential_calls <= 2, exponential_calls
        handle = quantizers[0].register_forward_hook(lambda _m, _a, output: output * 2)
        hooked = experts._quantize_grouped_weight(weight, quantizers)
        expected = reference.detach().clone()
        expected[0].mul_(2)
        torch.testing.assert_close(hooked, expected, rtol=0, atol=0)
    finally:
        if handle is not None:
            handle.remove()
        policy.close()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
def test_grouped_iq_repair_uses_bank_device(monkeypatch):
    current = torch.cuda.current_device()
    target = (current + 1) % torch.cuda.device_count()
    policy = IQRepairPolicy()
    try:
        with torch.cuda.device(target):
            model = (
                get_tiny_qwen3_moe(
                    hidden_size=256,
                    moe_intermediate_size=256,
                    num_hidden_layers=1,
                    num_experts=8,
                    num_attention_heads=4,
                )
                .cuda()
                .eval()
            )
            mtq.quantize(
                model,
                {
                    "quant_cfg": [
                        {"quantizer_name": "*", "enable": False},
                        {
                            "quantizer_name": "*experts.*weight_quantizers.*",
                            "cfg": {"num_bits": "iq1_m", "backend": "ggml"},
                        },
                    ],
                    "algorithm": "max",
                },
            )
            experts = next(
                m for m in model.modules() if isinstance(m, hf_plugin._QuantFusedExperts)
            )
            policy.prepare(experts)
            quantizers = experts.gate_up_proj_weight_quantizers
            weight = experts.gate_up_proj
            fields = tuple(p for q in quantizers for p in q._local_repair_state.parameters())
            reference = torch.stack([q(w) for q, w in zip(quantizers, weight)])
            gradient = torch.randn_like(reference)
            expected_grads = torch.autograd.grad(reference, fields, gradient)
        kernels = repair_formats._native_scale_kernels(weight)
        decode = kernels.decode_iq_levels
        calls = []

        def checked_decode(packed, *args):
            # Refuse the unsafe launch before it can poison the process's CUDA context.
            assert torch.cuda.current_device() == packed.device.index
            calls.append(torch.cuda.current_device())
            return decode(packed, *args)

        monkeypatch.setattr(kernels, "decode_iq_levels", checked_decode)
        actual = experts._quantize_grouped_weight(weight, quantizers)
        actual_grads = torch.autograd.grad(actual, fields, gradient)
        assert torch.cuda.current_device() == current
        assert calls == [target]
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        for actual_grad, expected_grad in zip(actual_grads, expected_grads):
            torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)
    finally:
        policy.close()
