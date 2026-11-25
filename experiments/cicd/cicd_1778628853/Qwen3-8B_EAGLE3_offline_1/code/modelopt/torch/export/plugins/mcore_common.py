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

"""Custom mapping from Megatron Core models to their Hugging Face counter part."""

from typing import Any

from .mcore_deepseek import deepseek_causal_lm_export, deepseek_causal_lm_import
from .mcore_gptoss import gptoss_causal_lm_export, gptoss_causal_lm_import
from .mcore_llama import (
    eagle3_deep_llama_causal_lm_export,
    eagle3_llama_causal_lm_export,
    eagle_llama_causal_lm_export,
    llama4_causal_lm_export,
    llama4_causal_lm_import,
    llama_causal_lm_export,
    llama_causal_lm_import,
)
from .mcore_nemotron import (
    nemotron_causal_lm_export,
    nemotron_h_causal_lm_export,
    nemotron_h_causal_lm_import,
)
from .mcore_qwen import (
    qwen3_causal_lm_export,
    qwen3_causal_lm_import,
    qwen25_causal_lm_export,
    qwen25_causal_lm_import,
)

all_mcore_hf_export_mapping: dict[str, Any] = {
    "DeepseekV2ForCausalLM": deepseek_causal_lm_export,
    "DeepseekV3ForCausalLM": deepseek_causal_lm_export,
    "LlamaForCausalLM": llama_causal_lm_export,
    "Llama4ForConditionalGeneration": llama4_causal_lm_export,
    "NemotronForCausalLM": nemotron_causal_lm_export,
    "NemotronHForCausalLM": nemotron_h_causal_lm_export,
    "LlamaForCausalLMEagle": eagle_llama_causal_lm_export,
    "LlamaForCausalLMEagle3": eagle3_llama_causal_lm_export,
    "LlamaForCausalLMEagle3Deep": eagle3_deep_llama_causal_lm_export,
    "Qwen3ForCausalLM": qwen3_causal_lm_export,
    "Qwen3MoeForCausalLM": qwen3_causal_lm_export,
    "Qwen2ForCausalLM": qwen25_causal_lm_export,
    "GptOssForCausalLM": gptoss_causal_lm_export,
}

all_mcore_hf_import_mapping: dict[str, Any] = {
    "LlamaForCausalLM": llama_causal_lm_import,
    "Llama4ForConditionalGeneration": llama4_causal_lm_import,
    "DeepseekV2ForCausalLM": deepseek_causal_lm_import,
    "DeepseekV3ForCausalLM": deepseek_causal_lm_import,
    "NemotronHForCausalLM": nemotron_h_causal_lm_import,
    "Qwen3ForCausalLM": qwen3_causal_lm_import,
    "Qwen3MoeForCausalLM": qwen3_causal_lm_import,
    "Qwen2ForCausalLM": qwen25_causal_lm_import,
    "GptOssForCausalLM": gptoss_causal_lm_import,
}
