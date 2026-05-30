# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

pytest.importorskip("vllm")

from modelopt.torch.quantization.plugins import vllm as vllm_plugin


class _FakeQuantizer:
    def __init__(self, enabled=True):
        self.is_enabled = enabled


class _FakeQuantMethod:
    def __init__(self, backend_name, with_kernel=True):
        self.unquantized_backend = backend_name
        self.moe_kernel = object() if with_kernel else None


class _FakeFusedMoE:
    def __init__(self, backend_name, quantizers_enabled=True, with_kernel=True):
        quantizers_enabled = (
            (quantizers_enabled,) * 4
            if isinstance(quantizers_enabled, bool)
            else quantizers_enabled
        )
        self.w13_input_quantizer = _FakeQuantizer(quantizers_enabled[0])
        self.w2_input_quantizer = _FakeQuantizer(quantizers_enabled[1])
        self.w13_weight_quantizer = _FakeQuantizer(quantizers_enabled[2])
        self.w2_weight_quantizer = _FakeQuantizer(quantizers_enabled[3])
        self.quant_method = _FakeQuantMethod(backend_name, with_kernel=with_kernel)


@pytest.mark.parametrize("backend_name", ["TRITON", "BATCHED_TRITON"])
def test_decomposed_moe_backends_are_supported(backend_name):
    module = _FakeFusedMoE(backend_name)
    original_kernel = module.quant_method.moe_kernel

    vllm_plugin._ensure_supported_moe_fakequant_backend(module)

    assert module.quant_method.moe_kernel is original_kernel


def test_disabled_expert_quantizers_do_not_require_decomposed_backend():
    module = _FakeFusedMoE("FLASHINFER_CUTLASS", quantizers_enabled=False)
    original_kernel = module.quant_method.moe_kernel

    vllm_plugin._ensure_supported_moe_fakequant_backend(module)

    assert module.quant_method.moe_kernel is original_kernel


@pytest.mark.parametrize("backend_name", ["FLASHINFER_CUTLASS", "FLASHINFER_TRTLLM", "AITER"])
def test_expert_fakequant_rejects_fused_or_unsupported_moe_backend(backend_name):
    module = _FakeFusedMoE(backend_name)

    with pytest.raises(RuntimeError, match="moe_backend='triton'"):
        vllm_plugin._ensure_supported_moe_fakequant_backend(module)


def test_any_enabled_expert_quantizer_requires_decomposed_backend():
    module = _FakeFusedMoE("FLASHINFER_CUTLASS", quantizers_enabled=(False, True, False, False))

    with pytest.raises(RuntimeError, match="--moe-backend triton"):
        vllm_plugin._ensure_supported_moe_fakequant_backend(module)


def test_missing_vllm_moe_kernel_is_left_unchanged():
    module = _FakeFusedMoE("FLASHINFER_CUTLASS", with_kernel=False)

    vllm_plugin._ensure_supported_moe_fakequant_backend(module)

    assert module.quant_method.moe_kernel is None
