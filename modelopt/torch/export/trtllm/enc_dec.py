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

"""Which Hugging Face model types the TensorRT-LLM checkpoint export handles as encoder-decoder.

Like the rest of this package, it goes away with the export in 0.49.0.
"""

__all__ = ["is_trtllm_enc_dec_export"]

# Other encoder-decoder models (e.g. LongT5, PLBart, T5Gemma) use the unified HF export.
_ENC_DEC_MODEL_TYPES = frozenset({"t5", "mt5", "umt5", "bart", "mbart", "whisper"})


def is_trtllm_enc_dec_export(model_type: str | None) -> bool:
    """Return whether the root Hugging Face model_type still exports a TensorRT-LLM checkpoint.

    These are the encoder-decoder families the deprecated TensorRT-LLM exporter supports.
    """
    return model_type in _ENC_DEC_MODEL_TYPES
