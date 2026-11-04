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

"""Which model types the deprecated TensorRT-LLM export handles as encoder-decoder."""

import pytest

from modelopt.torch.export.trtllm.enc_dec import is_trtllm_enc_dec_export


@pytest.mark.parametrize(
    ("model_type", "expected"),
    [
        ("t5", True),
        ("mt5", True),
        ("mbart", True),
        ("whisper", True),
        # Encoder-decoder, but not supported by the TensorRT-LLM exporter: unified HF export.
        ("longt5", False),
        ("plbart", False),
        ("t5gemma", False),
        ("florence2", False),  # VLM whose language model is BART; the root type decides
        ("qwen3_vl", False),
    ],
)
def test_is_trtllm_enc_dec_export(model_type: str, expected: bool):
    assert is_trtllm_enc_dec_export(model_type) is expected
