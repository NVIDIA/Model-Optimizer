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

import pytest
import torch
from _test_utils.torch.megatron.models import get_mcore_gpt_model
from _test_utils.torch.megatron.utils import (
    get_forward,
    initialize_for_megatron,
    sharded_state_dict_test_helper,
)

import modelopt.torch.quantization as mtq

pytest.importorskip("fla")  # Megatron-Core GatedDeltaNet and the state QDQ kernel need fla
GatedDeltaNet = pytest.importorskip("megatron.core.ssm.gated_delta_net").GatedDeltaNet

from modelopt.torch.export.plugins.vllm_fakequant_megatron import VllmFqGPTModelExporter
from modelopt.torch.quantization.plugins.gated_delta_net import _state_qdq_chunk_gated_delta_rule
from modelopt.torch.quantization.plugins.megatron import _QuantGatedDeltaNet

try:
    _state_qdq_chunk_gated_delta_rule()
except RuntimeError as e:
    pytest.skip(str(e), allow_module_level=True)

SEED = 1234


def _make_model(tp_size, num_heads=1):
    model = (
        get_mcore_gpt_model(
            tensor_model_parallel_size=tp_size,
            num_layers=1,
            hidden_size=64,
            num_attention_heads=4,
            vocab_size=32,
            max_sequence_length=128,
            experimental_attention_variant="gated_delta_net",
            linear_num_key_heads=num_heads,
            linear_num_value_heads=num_heads,
            linear_key_head_dim=32,
            linear_value_head_dim=32,
        )
        .cuda()
        .eval()
    )
    # Retain enough history for chunk-boundary state rounding to affect the next chunk.
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, GatedDeltaNet):
                module.A_log.fill_(-4)
                module.dt_bias.zero_()
    return model


def _gdn_config(sites):
    return {
        "quant_cfg": [{"quantizer_name": "*", "enable": False}]
        + [
            {
                "quantizer_name": f"*gdn_{site}_quantizer",
                "cfg": {
                    "num_bits": (4, 3),
                    "axis": (0, 1) if site == "state" else (0, 1, 2),
                    "type": "dynamic",
                },
            }
            for site in sites
        ],
        "algorithm": None,
    }


def _test_gdn_qat_helper(rank, size, checkpoint_path, sites):
    initialize_for_megatron(
        tensor_model_parallel_size=size, pipeline_model_parallel_size=1, seed=SEED
    )
    model = _make_model(size)
    forward = get_forward(model)
    outputs = []
    handles = [
        module.register_forward_hook(
            lambda module, args, result: outputs.append(result[0].detach().clone())
        )
        for module in model.modules()
        if isinstance(module, GatedDeltaNet)
    ]
    with torch.no_grad():
        loss_ref = forward(model)
    gdn_ref = outputs.copy()
    outputs.clear()

    mtq.quantize(model, _gdn_config(sites))
    gdn_modules = [m for m in model.modules() if isinstance(m, _QuantGatedDeltaNet)]
    assert gdn_modules, "no GatedDeltaNet layer was wrapped"
    for module in gdn_modules:
        assert module.gdn_state_quantizer.is_enabled == ("state" in sites)
        assert module.gdn_w_quantizer.is_enabled == ("w" in sites)

    with torch.no_grad():
        loss_quant = forward(model)
    assert torch.isfinite(loss_quant).all()
    # BF16 residual additions can hide a small GDN change at the final loss.
    assert len(outputs) == len(gdn_ref) == len(gdn_modules)
    assert any(not torch.equal(actual, ref) for actual, ref in zip(outputs, gdn_ref)), (
        "enabled QDQ must change the GDN branch output"
    )
    for handle in handles:
        handle.remove()

    for site in sites:
        mtq.disable_quantizer(model, f"*gdn_{site}_quantizer")
    with torch.no_grad():
        torch.testing.assert_close(forward(model), loss_ref, rtol=1e-4, atol=1e-4)
    for site in sites:
        mtq.enable_quantizer(model, f"*gdn_{site}_quantizer")

    restored = _make_model(size)
    sharded_state_dict_test_helper(checkpoint_path, model, restored, forward)
    restored_gdn = [m for m in restored.modules() if isinstance(m, _QuantGatedDeltaNet)]
    assert len(restored_gdn) == len(gdn_modules)
    for module in restored_gdn:
        assert module.gdn_state_quantizer.is_enabled == ("state" in sites)
        assert module.gdn_w_quantizer.is_enabled == ("w" in sites)
        assert module.in_proj.weight.grad is not None
        assert torch.isfinite(module.in_proj.weight.grad).all()

    # The checkpoint helper already ran backward through the restored model.
    optimizer = torch.optim.SGD(restored.parameters(), lr=1e-3)
    before = restored_gdn[0].in_proj.weight.detach().clone()
    optimizer.step()
    assert not torch.equal(restored_gdn[0].in_proj.weight, before)


def _compile_gdn_qat_kernels(rank, size, sites):
    initialize_for_megatron(
        tensor_model_parallel_size=size, pipeline_model_parallel_size=1, seed=SEED
    )
    model = _make_model(size)
    forward = get_forward(model)
    with torch.no_grad():
        forward(model)
    mtq.quantize(model, _gdn_config(sites))
    with torch.no_grad():
        forward(model)
    model.train()
    forward(model).sum().backward()
    torch.cuda.synchronize()


@pytest.fixture
def compiled_gdn_workers(dist_workers_size_1):
    """Warm one small QAT model in the same worker, outside the test-call budget."""
    # Ampere exercises W QDQ; native FP8 devices also exercise state QDQ.
    sites = ("state", "w") if torch.cuda.get_device_capability() >= (8, 9) else ("w",)
    dist_workers_size_1.run(_compile_gdn_qat_kernels, sites)
    return dist_workers_size_1, sites


def test_gdn_qat_and_sharded_restore(compiled_gdn_workers, tmp_path):
    """Train through QDQ after a Megatron distributed-checkpoint round trip."""
    workers, sites = compiled_gdn_workers
    workers.run(_test_gdn_qat_helper, tmp_path, sites)


def _gate_forward_backward(in_proj, x, gates, grad):
    """Gate rows of in_proj(x), and the input / gate-weight gradients of a loss on them alone."""
    x = x.detach().requires_grad_()
    in_proj.weight.grad = None
    output = in_proj(x)[0]
    (output[..., gates] * grad).sum().backward()
    return output.detach(), x.grad, in_proj.weight.grad[gates].clone()


def _test_in_proj_gates_unquantized_helper(rank, size):
    initialize_for_megatron(
        tensor_model_parallel_size=size, pipeline_model_parallel_size=1, seed=SEED
    )
    model = _make_model(size, num_heads=2)
    gdn = next(m for m in model.modules() if isinstance(m, GatedDeltaNet))
    in_proj = gdn.in_proj
    num_rows = in_proj.weight.shape[0]
    gates = slice(num_rows - 2 * 2 // size, num_rows)  # beta + alpha: one row per value head each
    weight = in_proj.weight.detach().clone()
    x = torch.randn(8, 2, 64, device="cuda", dtype=weight.dtype)
    grad = torch.randn(8, 2, gates.stop - gates.start, device="cuda", dtype=weight.dtype)
    reference = _gate_forward_backward(in_proj, x, gates, grad)

    fp8 = {"num_bits": (4, 3), "axis": None}
    config = {
        "quant_cfg": [
            {"quantizer_name": "*", "enable": False},
            {"quantizer_name": "*in_proj.weight_quantizer", "cfg": fp8},
            {"quantizer_name": "*in_proj.input_quantizer", "cfg": fp8},
        ],
        "algorithm": "max",
    }
    mtq.quantize(model, config, get_forward(model))

    # The gate rows, and the gradients through them alone, are the BF16 ones; the others are not.
    output, x_grad, gate_weight_grad = _gate_forward_backward(in_proj, x, gates, grad)
    torch.testing.assert_close(output[..., gates], reference[0][..., gates])
    assert not torch.allclose(output[..., : gates.start], reference[0][..., : gates.start])
    torch.testing.assert_close(x_grad, reference[1])
    torch.testing.assert_close(gate_weight_grad, reference[2])

    exporter = object.__new__(VllmFqGPTModelExporter)
    exporter._quantizer_recipe_markers, exporter.exclude_modules = [], []
    exporter._grouped_fq_modules, exporter._packing_experts = set(), False
    exported = exporter._get_quantized_state(in_proj, weight.dtype)[0]["weight"]
    assert torch.equal(exported[gates], weight[gates].cpu())
    assert not torch.equal(exported[: gates.start], weight[: gates.start].cpu())

    # An AWQ / SmoothQuant input scale, which the stored weight compensates for, still applies.
    # Powers of two, so that scaling the input and dividing the weight round nothing.
    scale = 2.0 ** torch.randint(-1, 2, (64,), device="cuda").to(weight.dtype)
    in_proj.input_quantizer.pre_quant_scale = scale
    with torch.no_grad():
        in_proj.weight.div_(scale)
    scaled_weight = in_proj.weight.detach().clone()
    with torch.no_grad():
        output = in_proj(x)[0]
    torch.testing.assert_close(output[..., gates], reference[0][..., gates])

    mtq.fold_weight(model)
    assert torch.equal(in_proj.weight[gates], scaled_weight[gates])
    assert not torch.equal(in_proj.weight[: gates.start], scaled_weight[: gates.start])


def test_in_proj_keeps_gates_unquantized(request):
    """The beta / alpha rows of the fused in_proj stay BF16 in fake quant, as in every export."""
    # TP stays <= 2, the number of value heads.
    workers = request.getfixturevalue(f"dist_workers_size_{min(torch.cuda.device_count(), 2)}")
    workers.run(_test_in_proj_gates_unquantized_helper)
