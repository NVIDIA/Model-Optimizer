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

"""Fake quantization of the MoE expert-parallel communication of DeepGEMM's Mega-MoE kernel in vLLM.

Covers ``--moe-backend deep_gemm_mega_moe`` of DeepSeek-V4/V4.1 and Kimi-K3. The kernel fuses
dispatch, expert GEMMs and combine, so the quantizers of vLLM's ``RoutedExperts`` cannot reach it.
``dispatch_quantizer`` fake-quantizes the tokens before the kernel's FP8 staging, which only these
tokens feed. ``combine_quantizer`` fake-quantizes the per-(token, top-k slot) expert outputs the
kernel sends back, which stay in its symmetric buffer: after the kernel, they are quantized and
summed again in the kernel's order, replacing its output.
"""

import importlib

import torch

from ..nn import (
    QuantLinearConvBase,
    QuantModule,
    QuantModuleRegistry,
    SequentialQuantizer,
    TensorQuantizer,
)
from .vllm import create_parallel_state

__all__ = []

# NotImplementedError: some vLLM model packages reject unsupported platforms at import.
try:
    from vllm.models.deepseek_v4.nvidia.model import DeepseekV4MegaMoEExperts
except (ImportError, NotImplementedError):
    DeepseekV4MegaMoEExperts = None
try:
    from vllm.models.kimi_k3.nvidia.model import KimiK3MegaMoEExperts
except (ImportError, NotImplementedError):
    KimiK3MegaMoEExperts = None

# Symmetric buffers whose combine-row layout was checked against the kernel output.
_checked_buffers: set[int] = set()


def _combine_slots(symm_buffer, num_tokens: int) -> torch.Tensor:
    """The kernel's BF16 combine rows as ``[num_topk (+1 fused shared expert), num_tokens, hidden]``.

    They are the last region of the symmetric buffer (DeepGEMM ``layout/mega_moe.cuh``), so their
    offset is the configuration's buffer size minus their size. The buffer can be larger when it is
    shared with a bigger configuration.
    """
    buffer = symm_buffer
    num_slots = buffer.num_topk + (1 if buffer.num_shared_experts else 0)
    max_tokens = buffer.num_max_tokens_per_rank
    num_bytes = num_slots * max_tokens * buffer.hidden * 2
    deep_gemm_c = importlib.import_module(type(buffer).__module__)._C  # the build that sized it
    size = deep_gemm_c.get_symm_buffer_size_for_mega_moe(
        buffer.group.size(),
        buffer.num_experts,
        max_tokens,
        buffer.num_topk,
        buffer.hidden,
        buffer.intermediate_hidden,
        buffer.mma_type,
        buffer.activation,
        buffer.num_shared_experts,
    )[0]
    if buffer.mma_type != "fp8xfp4" or size < num_bytes or size > buffer.buffer.numel():
        raise RuntimeError(f"Unsupported DeepGEMM Mega-MoE buffer layout ({buffer.mma_type}).")
    rows = buffer.buffer[size - num_bytes : size].view(torch.bfloat16)
    return rows.view(num_slots, max_tokens, buffer.hidden)[:, :num_tokens]


def _ordered_sum(slots, valid, num_topk, quantize=None) -> torch.Tensor:
    """Sum the combine rows like the kernel does.

    Valid top-k slots in ascending order, then the fused shared expert's slot, summed in FP32 from
    zero and rounded to BF16 once. A Python loop keeps the order fixed.
    """
    total = torch.zeros(slots.shape[1:], dtype=torch.float32, device=slots.device)
    for k in range(num_topk):
        contribution = torch.where(valid[k], slots[k], 0)
        total += (quantize(contribution) if quantize else contribution).float()
    if slots.shape[0] > num_topk:  # the fused shared expert's local output, not communicated
        total += slots[num_topk].float()
    return total.to(torch.bfloat16)


class _QuantVLLMMegaMoEExperts(QuantModule):
    """``dispatch_quantizer`` and ``combine_quantizer`` for vLLM's DeepGEMM Mega-MoE experts."""

    def _setup(self):
        self.dispatch_quantizer = TensorQuantizer(QuantLinearConvBase.default_quant_desc_input)
        self.combine_quantizer = TensorQuantizer(QuantLinearConvBase.default_quant_desc_input)
        self.dispatch_quantizer.disable()
        self.combine_quantizer.disable()
        self.parallel_state = create_parallel_state()

    def forward(self, hidden_states, *args, **kwargs):
        if self.dispatch_quantizer.is_enabled and self.has_fused_shared_experts:
            raise RuntimeError(
                f"{self.prefix}: the fused shared expert reads the dispatched tokens too; set"
                " VLLM_DISABLE_DSV4_MEGAMOE_SHARED_EXPERT_FUSION=1 to quantize the dispatch."
            )
        output = super().forward(self.dispatch_quantizer(hidden_states), *args, **kwargs)
        if self.combine_quantizer.is_enabled:
            self._check_combine_scale()
            symm_buffer = self.get_symm_buffer()
            slots = _combine_slots(symm_buffer, output.shape[0])
            valid = (symm_buffer.topk_idx[: output.shape[0]] >= 0).T.unsqueeze(-1)
            if id(symm_buffer) not in _checked_buffers:
                self._check_combine_rows(symm_buffer, slots, valid, output)
            output.copy_(_ordered_sum(slots, valid, symm_buffer.num_topk, self.combine_quantizer))
        return output

    def _check_combine_rows(self, symm_buffer, slots, valid, output) -> None:
        """Check that the combine rows reproduce the kernel output bit for bit.

        Eager forwards check each buffer once, on a batch with routed rows (warmup batches are all
        padding). A CUDA graph captured before that check checks every replay on the device instead,
        failing with a device-side assert.
        """
        message = (
            f"{self.prefix}: the DeepGEMM Mega-MoE combine rows do not reproduce the kernel output:"
            " its buffer layout changed, or DG_COMM_KERNEL_DEBUG=1 zeroed it."
        )
        reproduced = (_ordered_sum(slots, valid, symm_buffer.num_topk) == output).all()
        if torch.cuda.is_current_stream_capturing():
            torch._assert_async(reproduced, message)
        elif valid.any():
            if not reproduced:
                raise RuntimeError(message)
            _checked_buffers.add(id(symm_buffer))

    def _check_combine_scale(self) -> None:
        """Require a global scale fixed across ranks while serving.

        Each rank re-quantizes the rows it received from all expert ranks, so a scale computed per
        call would cover other rows than the sending rank's. Calibration collects the amax instead.
        """
        quantizer = self.combine_quantizer
        members = quantizer if isinstance(quantizer, SequentialQuantizer) else (quantizer,)
        if not all(
            q._if_calib
            or q.is_mx_format
            or q._use_constant_amax
            or q._constant_amax is not None
            or q.amax is not None
            for q in members
            if q.is_enabled
        ):
            raise RuntimeError(
                f"{self.prefix}: Mega-MoE combine_quantizer needs a calibrated (algorithm: max) or"
                " constant (constant_amax) global scale."
            )


for _experts_cls, _name in (
    (DeepseekV4MegaMoEExperts, "vllm_DeepseekV4MegaMoEExperts"),
    (KimiK3MegaMoEExperts, "vllm_KimiK3MegaMoEExperts"),  # overrides forward, so registered too
):
    if _experts_cls is not None:
        QuantModuleRegistry.register({_experts_cls: _name})(_QuantVLLMMegaMoEExperts)
