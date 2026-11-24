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

"""CUDA memory and freshness of staged native payload reads in inference mode."""

import json
import statistics
import time

import pytest
import torch

import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.config import QuantizerAttributeConfig
from modelopt.torch.quantization.qtensor import FP8QTensor, QTensorWrapper
from modelopt.torch.quantization.repair_partition import prepare_packed_weights_for_tp
from modelopt.torch.quantization.utils.native_weight import get_pinned_native_weight


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Native read memory qualification needs CUDA"
)
def test_staged_native_reads_bound_scratch_and_detect_inference_mutation():
    model = mtq.quantize(
        torch.nn.Sequential(torch.nn.Linear(4096, 4096, bias=False, device="cuda")),
        {"quant_cfg": [{"quantizer_name": "*", "enable": False}], "algorithm": None},
    )
    layer, quantizer = model[0], model[0].weight_quantizer
    native, scale = FP8QTensor.quantize(layer.weight.detach(), axis=0)
    quantizer.set_from_attribute_config(
        QuantizerAttributeConfig(num_bits=(4, 3), axis=0, fake_quant=False)
    )
    quantizer._set_buffer("_scale", scale)
    layer.weight = QTensorWrapper(native)
    with torch.inference_mode():
        prepare_packed_weights_for_tp(model)
        elapsed, scratch = [], []
        for _ in range(4):
            torch.cuda.synchronize()
            allocated = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            assert get_pinned_native_weight(quantizer, layer.weight) is not None
            torch.cuda.synchronize()
            elapsed.append(time.perf_counter() - start)
            scratch.append(torch.cuda.max_memory_allocated() - allocated)
        assert scratch[0] <= 16 * 1024**2
        assert max(scratch[1:]) <= 1024**2
        # Inference mode permits in-place edits; identity alone cannot validate freshness.
        layer.weight.add_(1)
        assert get_pinned_native_weight(quantizer, layer.weight) is None
    print(
        "Native pinned read CUDA metrics: "
        + json.dumps(
            {
                "device": torch.cuda.get_device_name(),
                "shape": [4096, 4096],
                "first_read_seconds": elapsed[0],
                "warm_median_seconds": statistics.median(elapsed[1:]),
                "extra_peak_allocated_bytes": scratch,
            }
        )
    )
