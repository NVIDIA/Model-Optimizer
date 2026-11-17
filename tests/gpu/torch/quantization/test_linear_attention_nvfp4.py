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

import torch

from modelopt.torch.quantization.config import QuantizerAttributeConfig
from modelopt.torch.quantization.linear_attention.state_clip import NVFP4StateClip
from modelopt.torch.quantization.linear_attention.utils import _state_qdq, state_quantizer_config
from modelopt.torch.quantization.nn import TensorQuantizer


def test_nvfp4_state_qdq_and_identity_gradient():
    torch.manual_seed(191)
    value = torch.randn(2, 2, 32, 32, device="cuda", requires_grad=True)
    quantizer = TensorQuantizer(
        QuantizerAttributeConfig(
            num_bits=(2, 1),
            type="dynamic",
            pass_through_bwd=True,
            block_sizes={-1: 16, "type": "dynamic", "scale_bits": (4, 3)},
        )
    ).cuda()
    assert state_quantizer_config(quantizer) == ("nvfp4", 16)
    result = _state_qdq(value, quantizer)
    expected = torch.stack(
        [quantizer(matrix) for matrix in value.reshape(-1, value.shape[-2], value.shape[-1])]
    ).reshape_as(value)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    assert torch.isfinite(result).all() and not torch.equal(result, value)

    # A much larger co-scheduled request or head must not change the first matrix's scale.
    small = value.detach()[:1, :1]
    for dim in (0, 1):
        mixed = torch.cat((small, small * 3000), dim=dim)
        first = _state_qdq(mixed, quantizer)[:1, :1]
        torch.testing.assert_close(first, _state_qdq(small, quantizer), rtol=0, atol=0)
    probe = torch.randn_like(value)
    (gradient,) = torch.autograd.grad((result * probe).sum(), value)
    torch.testing.assert_close(gradient, probe, rtol=0, atol=0)


def test_learned_state_clip_format_and_isolation():
    torch.manual_seed(41)
    value = torch.randn(2, 4, 32, 64, device="cuda")
    quantizer = TensorQuantizer(
        QuantizerAttributeConfig(
            num_bits=(2, 1),
            type="dynamic",
            pass_through_bwd=True,
            block_sizes={-1: 16, "type": "dynamic", "scale_bits": (4, 3)},
        )
    ).cuda()
    clip = NVFP4StateClip(torch.full((4,), 10000.0, device="cuda"))
    torch.testing.assert_close(clip(value), _state_qdq(value, quantizer), atol=2e-5, rtol=2e-5)
    clip.initial_cap.fill_(1.0)
    for dim in (0,):
        mixed = torch.cat((value[:1], value[:1] * 3000), dim=dim)
        torch.testing.assert_close(clip(mixed)[:1], clip(value[:1]), atol=0, rtol=0)
    torch.testing.assert_close(
        clip(torch.zeros_like(value)), torch.zeros_like(value), atol=0, rtol=0
    )
    object.__setattr__(quantizer, "_state_clip", clip)
    quantizer.disable()
    assert _state_qdq(value, quantizer) is value


def test_learned_state_clip_gradient_checkpoint_and_graph():
    torch.manual_seed(43)
    value = torch.randn(1, 4, 32, 64, device="cuda", requires_grad=True)
    clip = NVFP4StateClip(torch.ones(4, device="cuda"))
    output = clip(value)
    value_grad, cap_grad = torch.autograd.grad(output.square().mean(), (value, clip.phi))
    assert torch.isfinite(value_grad).all() and torch.isfinite(cap_grad).all()
    assert cap_grad.abs().sum() > 0
    with torch.no_grad():
        clip.phi.add_(-1e-5 * cap_grad.sign())
    restored = NVFP4StateClip(torch.ones(4, device="cuda"))
    restored.load_state_dict(clip.state_dict())
    torch.testing.assert_close(restored(value), clip(value), atol=0, rtol=0)
    # FP32 learned caps must survive the model's BF16 wrapper.
    original_caps = restored.caps().detach().clone()
    restored.bfloat16()
    assert restored.phi.dtype == restored.initial_cap.dtype == torch.float32
    torch.testing.assert_close(restored.caps(), original_caps, atol=0, rtol=0)
    static = value.detach().clone()
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream), torch.no_grad():
        for _ in range(3):
            restored(static)
    torch.cuda.current_stream().wait_stream(stream)
    with torch.cuda.graph(graph), torch.no_grad():
        captured = restored(static)
    with torch.no_grad():
        restored.phi.add_(1e-4)
        static.mul_(1.2)
    graph.replay()
    torch.testing.assert_close(captured, restored(static), atol=0, rtol=0)
