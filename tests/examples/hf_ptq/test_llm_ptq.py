# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from _test_utils.examples.hf_ptq_utils import PTQCommand
from _test_utils.examples.models import MIXTRAL_PATH, T5_PATH, TINY_LLAMA_PATH


# Encoder-decoder representative; BART shares the same export path.
@pytest.mark.parametrize("command", [PTQCommand(quant="fp8", min_sm=89)], ids=PTQCommand.param_str)
def test_ptq_t5(command):
    command.run(T5_PATH)


@pytest.mark.parametrize(
    "command",
    [
        PTQCommand(quant="fp8", min_sm=90),
    ],
    ids=PTQCommand.param_str,
)
def test_ptq_mixtral(command):
    command.run(MIXTRAL_PATH)


# Per-format export on tiny models (int8_weight_only, nvfp4_awq_lite, ...) is covered by
# tests/gpu/torch/export/test_unified_hf_export_and_check_safetensors.py.
@pytest.mark.parametrize(
    "command",
    [
        PTQCommand(quant="int8_smoothquant", kv_cache_quant="none"),
        PTQCommand(quant="int4_awq", kv_cache_quant="none"),
        PTQCommand(quant="w4a8_awq_beta", kv_cache_quant="none"),
        # GGML IQ weight-only, recipe-driven: the five formats between 1.56 and 2.56 bits per
        # weight, on the MLP layers with layerwise GPTQ. These encoders require every weight's
        # input dimension to be a multiple of 256; TinyLlama's 2048 and 5632 both are.
        PTQCommand(recipe="general/ptq/iq1_s", kv_cache_quant="none"),
        PTQCommand(recipe="general/ptq/iq1_m", kv_cache_quant="none"),
        PTQCommand(recipe="general/ptq/iq2_xxs", kv_cache_quant="none"),
        PTQCommand(recipe="general/ptq/iq2_xs", kv_cache_quant="none"),
        PTQCommand(recipe="general/ptq/iq2_s", kv_cache_quant="none"),
        # Q8_0 has 32-value blocks and needs no calibration forward pass.
        PTQCommand(recipe="general/ptq/q8_0", kv_cache_quant="none"),
        # fp8 and nvfp4 checkpoints are also deployed with TRT-LLM
        PTQCommand(quant="fp8", min_sm=89),
        PTQCommand(quant="fp8", kv_cache_quant="none", min_sm=89),
        PTQCommand(quant="nvfp4"),
        PTQCommand(quant="mxfp8", min_sm=100),
        # Calibrated KV cache
        PTQCommand(quant="nvfp4_awq_lite", kv_cache_quant="nvfp4"),
        # AutoQuantize recipe; KV via --kv_cache_quant fallback
        PTQCommand(
            recipe="general/auto_quantize/nvfp4_fp8_at_5p4bits",
            kv_cache_quant="nvfp4",
            calib_batch_size=4,
        ),
        # multi_gpu
        PTQCommand(quant="fp8", min_gpu=2, min_sm=89),
        PTQCommand(quant="nvfp4", min_gpu=2, min_sm=100),
    ],
    ids=PTQCommand.param_str,
)
def test_ptq_llama(command):
    command.run(TINY_LLAMA_PATH)
