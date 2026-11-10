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

"""T5 PTQ modeling (HF model type ``t5``)."""

import torch
from transformers.models.t5.modeling_t5 import T5Attention

from modelopt.torch.quantization.nn import QuantInputBase, QuantModule, TensorQuantizer
from modelopt.torch.quantization.plugins.huggingface import _QUANT_ATTENTION_WRAPPERS
from modelopt.torch.quantization.utils import replace_function

__all__: list[str] = []


class _T5QuantAttention(QuantModule):
    """Attention class for KV Cache quantization compatible with T5 Model."""

    def _quantized_matmul(self, batch1, batch2):
        # T5Attention has two matmul operations, one for the query and key and one for the attention and value.
        # The first matmul is quantized with the q_bmm_quantizer and k_bmm_quantizer. The second matmul is
        # quantized with the v_bmm_quantizer.
        if self.qk_quant_matmul:
            self.qk_quant_matmul = False
            q, k = batch1, batch2
            return torch._matmul(
                self.q_bmm_quantizer(q), self.k_bmm_quantizer(k.transpose(3, 2)).transpose(3, 2)
            )
        else:
            self.qk_quant_matmul = True
            attn, v = batch1, batch2
            return torch._matmul(attn, self.v_bmm_quantizer(v))

    def _setup(self):
        self.q_bmm_quantizer = TensorQuantizer(QuantInputBase.default_quant_desc_input)
        self.k_bmm_quantizer = TensorQuantizer(QuantInputBase.default_quant_desc_input)
        self.v_bmm_quantizer = TensorQuantizer(QuantInputBase.default_quant_desc_input)

    @staticmethod
    def is_compatible_attention(attn):
        return issubclass(attn, T5Attention)

    def forward(self, *args, **kwargs):
        # self.qk_quant_matmul is used to alternate between the two matmul operations for T5Attention
        self.qk_quant_matmul = True
        with replace_function(torch, "matmul", self._quantized_matmul):
            return super().forward(*args, **kwargs)


# After _QuantAttention, which claims T5Attention instead where T5 uses transformers' attention
# interface (transformers 5.15 on).
_QUANT_ATTENTION_WRAPPERS.append(_T5QuantAttention)
