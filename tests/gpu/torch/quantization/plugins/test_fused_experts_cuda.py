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
from modelopt.torch.quantization import tensor_quant

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
