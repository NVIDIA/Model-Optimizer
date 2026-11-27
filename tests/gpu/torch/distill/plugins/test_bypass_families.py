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

"""Real reduced release models: packed text, IQ updates, frozen teacher and native reload."""

import copy
import json

import pytest
import torch

pytest.importorskip("transformers", minversion="5.23")

from _test_utils.transformers_families import build_model, hf_torch_kernels, release_cases

import modelopt.torch.distill as mtd
import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq
from modelopt.torch.distill.plugins.transformers import TransformersBypassAdapter
from modelopt.torch.models.kimi_k31.modeling_kimi_k31 import KimiK31ShortConvolution
from modelopt.torch.models.step5 import Step5Config
from modelopt.torch.models.step5 import modeling_step5 as step5_math
from modelopt.torch.quantization.ggml import IQ_FORMAT_REGISTRY
from modelopt.torch.quantization.ggml.common import pinned_packed_weight
from modelopt.torch.quantization.ggml.scale_fields import decode_scale_fields, replace_scale_fields
from modelopt.torch.quantization.nn import QuantModule

_PORTED = {"deepseek_v41", "kimi_k31", "step5"}


@pytest.mark.parametrize(
    "record", [r for r in release_cases() if r["family"] in _PORTED], ids=lambda r: r["model_id"]
)
@pytest.mark.timeout(300)
def test_release_text_bypass_and_native_reload(record, tmp_path):
    torch.manual_seed(149)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher = build_model(record, device)
    teacher_before = {n: p.detach().clone() for n, p in teacher.state_dict().items()}
    ids = torch.randint(3, 100, (2, 8), device=device)
    batch = {
        "input_ids": ids,
        "attention_mask": torch.ones_like(ids),
        "position_ids": torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]], device=device).expand(2, -1),
        "_packed_seq_ids": torch.tensor([[1, 1, 1, 1, 2, 2, 2, 2]], device=device).expand(2, -1),
    }
    forward = {k: v for k, v in batch.items() if k != "_packed_seq_ids"}
    adapter = TransformersBypassAdapter()
    with hf_torch_kernels(teacher):
        student = mtq.quantize(
            copy.deepcopy(teacher),
            {
                "quant_cfg": [
                    {"quantizer_name": "*", "enable": False},
                    *[
                        {
                            "quantizer_name": pattern,
                            "cfg": {"num_bits": "iq2_xxs", "backend": "ggml"},
                        }
                        for pattern in adapter.weight_patterns(teacher)
                    ],
                ],
                "algorithm": "max",
            },
            lambda model: model(**forward),
        )
        if record["family"] == "step5":
            shared = [
                q
                for n, q in student.named_modules()
                if ("share_expert" in n or "shared_experts" in n) and n.endswith("weight_quantizer")
            ]
            assert shared and all(q.is_enabled and q.num_bits == "iq2_xxs" for q in shared)
        original = {}
        fmt = IQ_FORMAT_REGISTRY["iq2_xxs"]
        for module in student.modules():
            if isinstance(module, QuantModule):
                for weight, quantizer in module.iter_weights_for_calibration():
                    if quantizer.is_enabled and quantizer.backend == "ggml":
                        original[quantizer] = fmt.pack(weight, quantizer).clone()
        frozen = {
            n: p.detach().clone()
            for n, p in student.named_parameters()
            if ".attn." in n or ".self_attn." in n or ".gate." in n or "norm" in n
        }
        assert original
        assert (
            mtd.bypass(
                student,
                teacher,
                [batch, batch],
                {"num_steps": 2, "global_batch_size": 2, "sequence_length": 8},
            )
            is student
        )
        changed = False
        for quantizer, before in original.items():
            after = pinned_packed_weight(quantizer, "iq2_xxs")
            global_scale, local_scale, _ = decode_scale_fields(before, fmt)
            assert torch.equal(replace_scale_fields(after, fmt, global_scale, local_scale), before)
            changed |= not torch.equal(before, after)
        assert changed
        assert all(torch.equal(student.get_parameter(n), p) for n, p in frozen.items())
        assert all(torch.equal(teacher.state_dict()[n], p) for n, p in teacher_before.items())
        assert not any("repair" in n for n, _ in student.named_modules())
        with torch.no_grad():
            expected = student(**forward).logits
        mto.save(student, tmp_path / "student.pt")
        restored = mto.restore(build_model(record, device), tmp_path / "student.pt")
        with torch.no_grad(), hf_torch_kernels(restored):
            torch.testing.assert_close(restored(**forward).logits, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FLA short convolution requires CUDA")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_k31_fla_short_convolution_preserves_fp32_weight_boundary(dtype):
    # The optional FLA path is admitted by the same import as the model's kernel selector.
    causal_conv1d = pytest.importorskip("fla.modules.conv.causal_conv1d").causal_conv1d

    torch.manual_seed(17)
    conv = KimiK31ShortConvolution(8, 3).cuda()
    with torch.no_grad():
        conv.weight.normal_(std=0.1)
    x = torch.randn(2, 16, 8, device="cuda", dtype=dtype, requires_grad=True)
    reference = x.detach().clone().requires_grad_()
    actual = conv(x, kernel=causal_conv1d)
    expected = conv(reference)
    assert conv.weight.dtype == torch.float32 and actual.dtype == dtype
    torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.003)
    actual.float().square().sum().backward()
    expected.float().square().sum().backward()
    torch.testing.assert_close(x.grad, reference.grad, rtol=0.05, atol=0.01)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Measure sparse attention peak memory on CUDA"
)
@pytest.mark.timeout(300)
def test_step5_release_sparse_geometry_bounds_saved_gather_memory(monkeypatch, record_property):
    # Release config 364b3b03:64Q/4KV heads,192head_dim,512regions x8tokens.
    # Reduce only the projection input width; the selected K/V banks retain release geometry.
    config = Step5Config(
        hidden_size=128,
        num_hidden_layers=1,
        num_attention_heads=64,
        num_attention_groups=4,
        head_dim=192,
        layer_types=["full_attention"],
        partial_rotary_factors=[1 / 3],
        moe_layer_list=[],
        moe_num_experts=1,
        moe_top_k=1,
        sparse_config={
            "proxy_dim": 256,
            "sparse_indexer_num_heads": 16,
            "num_provider_groups": 4,
            "topk": 512,
            "region_block_size": 8,
        },
    )
    attention = step5_math.Step5Attention(config, 0).to(device="cuda", dtype=torch.bfloat16)
    prefix, length = 4096, 64
    positions = torch.arange(prefix + length, device="cuda").unsqueeze(0)
    valid = torch.ones_like(positions, dtype=torch.bool)
    layout = step5_math._region_layout(positions, valid, 8)
    layout.cached_regions = 512
    hidden = torch.randn(1, length, 128, device="cuda", dtype=torch.bfloat16)
    keys = torch.randn(1, 4, prefix, 192, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    values = torch.randn_like(keys, requires_grad=True)
    with torch.no_grad():
        prefix_hidden = torch.randn(1, prefix, 128, device="cuda", dtype=torch.bfloat16)
        prefix_layout = step5_math._region_layout(positions[:, :prefix], valid[:, :prefix], 8)
        _, compressed = attention.indexer.prepare(
            prefix_hidden, positions[:, :prefix], prefix_layout
        )

    def run(recompute):
        attention.zero_grad(set_to_none=True)
        keys.grad = values.grad = None
        inputs = hidden.detach().clone().requires_grad_()
        cache = step5_math.Step5Cache()
        cache.update(keys, values, 0)
        cache.indexer_states[0] = compressed
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        with monkeypatch.context() as patch:
            if not recompute:
                patch.setattr(step5_math, "checkpoint", lambda fn, *args, **kwargs: fn(*args))
            output = attention(inputs, positions[:, prefix:], layout, cache)
            output.float().square().sum().backward()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() - baseline
        return peak, output.detach().clone(), inputs.grad.detach().clone()

    run(True)
    old_peak, expected, expected_gradient = run(False)
    new_peak, actual, actual_gradient = run(True)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=0.03, atol=0.003)
    report = {
        "baseline_peak_delta_bytes": old_peak,
        "recompute_peak_delta_bytes": new_peak,
        "query_tokens": length,
        "prefix_tokens": prefix,
        "heads": 64,
        "kv_groups": 4,
        "head_dim": 192,
        "topk_regions": 512,
        "region_block_size": 8,
    }
    record_property("step5_sparse_memory", json.dumps(report))
    print(json.dumps(report))
    assert new_peak < old_peak * 0.9, report
