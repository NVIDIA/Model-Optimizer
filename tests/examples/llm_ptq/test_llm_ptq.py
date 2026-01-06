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
from _test_utils.examples.llm_ptq_utils import PTQCommand, WithRequirements
from _test_utils.examples.models import (
    BART_PATH,
    MIXTRAL_PATH,
    T5_PATH,
    TINY_LLAMA_PATH,
    WHISPER_PATH,
)


@pytest.mark.parametrize(
    "command",
    [
        PTQCommand(quant="fp8", min_sm=89),
    ],
    ids=PTQCommand.param_str,
)
def test_ptq_bart(command):
    command.run(BART_PATH)


class TestT5(WithRequirements):
    requirements = [("transformers", "4.48.0")]

    @pytest.mark.parametrize(
        "command",
        [
            PTQCommand(quant="fp8", min_sm=89),
        ],
        ids=PTQCommand.param_str,
    )
    def test_ptq_t5(self, command):
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


class TestWhisper(WithRequirements):
    requirements = [
        ("librosa", None),
        ("soundfile", None),
    ]

    @pytest.mark.parametrize(
        "command",
        [
            # Auto-batch-size computation seems to take >10mins for Whisper hence using a fixed batch size
            PTQCommand(quant="fp8", calib_batch_size=16, min_sm=89),
        ],
        ids=PTQCommand.param_str,
    )
    def test_ptq_whisper(self, command):
        command.run(WHISPER_PATH)


@pytest.mark.parametrize(
    "command",
    [
        PTQCommand(quant="int8_sq", kv_cache_quant="none"),
        PTQCommand(quant="int8_sq", kv_cache_quant="none", tp=2, pp=2),
        PTQCommand(quant="int8_wo", kv_cache_quant="none"),
        PTQCommand(quant="int4_awq", kv_cache_quant="none"),
        PTQCommand(quant="w4a8_awq", kv_cache_quant="none"),
        PTQCommand(quant="nvfp4"),
        PTQCommand(quant="nvfp4_awq"),
        # autoquant
        PTQCommand(
            quant="int4_awq,nvfp4,fp8,w4a8_awq",
            calib_batch_size=4,
            auto_quantize_bits=6.4,
            kv_cache_quant="none",
        ),
        # kv_cache
        PTQCommand(quant="nvfp4_awq", kv_cache_quant="nvfp4"),
        # autoquant_kv_cache
        PTQCommand(
            quant="nvfp4,fp8",
            kv_cache_quant="fp8",
            calib_batch_size=4,
            auto_quantize_bits=6.4,
        ),
        PTQCommand(
            quant="nvfp4,fp8",
            kv_cache_quant="nvfp4",
            calib_batch_size=4,
            auto_quantize_bits=6.4,
        ),
        # sm89
        PTQCommand(quant="fp8", min_sm=89),
        PTQCommand(quant="fp8", kv_cache_quant="none", min_sm=89),  # sm100
        PTQCommand(quant="mxfp8", min_sm=100),
        PTQCommand(quant="nvfp4", min_sm=100),
        #
        # multi_gpu
        PTQCommand(quant="fp8", min_gpu=2, min_sm=89),
        PTQCommand(quant="nvfp4", min_gpu=2, min_sm=100),
    ],
    ids=PTQCommand.param_str,
)
def test_ptq_llama(command):
    command.run(TINY_LLAMA_PATH)
