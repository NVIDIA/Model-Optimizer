# Adapted from ResInfer (tensor_quantizer.py / residue_linear.py):
# https://github.com/ShawRong/resinfer/tree/51b906836a69fa43915040314df57bc9f23030ad/quant/modelopt_integration
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2025 Rong Shuo
# SPDX-License-Identifier: Apache-2.0

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

"""Residue-aware quantized Linear."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...qtensor import MXFP4QTensor
from .quant_module import QuantLinearConvBase
from .residue_tensor_quantizer import ResidueTensorQuantizer

__all__ = ["ResidueQuantLinear"]


class ResidueQuantLinear(QuantLinearConvBase):
    """Quantized Linear with selected activation residuals and copied weight channels."""

    def _setup(self):
        """Reuse the existing quantizers, as in SVDQuantLinear."""
        if not isinstance(self.weight_quantizer, ResidueTensorQuantizer):
            self.weight_quantizer.__class__ = ResidueTensorQuantizer

        if not isinstance(self.input_quantizer, ResidueTensorQuantizer):
            self.input_quantizer.__class__ = ResidueTensorQuantizer

        if not hasattr(self.input_quantizer, "_enable_salient_indices"):
            self.input_quantizer._enable_salient_indices = True
        if not hasattr(self.weight_quantizer, "_enable_salient_indices"):
            self.weight_quantizer._enable_salient_indices = True

    def _apply_pre_quant_scale(self, input: torch.Tensor) -> torch.Tensor:
        pre_quant_scale = self.input_quantizer.pre_quant_scale
        if pre_quant_scale is None:
            return input
        return input * pre_quant_scale.to(device=input.device, dtype=input.dtype)

    def _input_qdq_without_pre_quant_scale(self, input: torch.Tensor) -> torch.Tensor:
        """Run input quantizer QDQ after caller has already applied pre_quant_scale."""
        with (
            self.input_quantizer.disable_pre_quant_scale(),
            self.input_quantizer.allow_residue_linear_fake_quant(),
        ):
            return self.input_quantizer(input)

    def _make_residue_input_qdq(
        self,
        input: torch.Tensor,
        salient_indices: torch.Tensor,
    ) -> torch.Tensor:
        input_for_quant = self._apply_pre_quant_scale(input)
        is_mx = self.input_quantizer.block_sizes.get("scale_bits") == (8, 0)
        if is_mx:
            input_for_quant = input_for_quant.float()
        salient_indices = salient_indices.to(device=input.device, dtype=torch.long)

        input_base = self._input_qdq_without_pre_quant_scale(input_for_quant)
        input_residue = input_for_quant.index_select(-1, salient_indices) - input_base.index_select(
            -1, salient_indices
        )
        input_residue_q = self._input_qdq_without_pre_quant_scale(input_residue)

        result = torch.cat([input_base, input_residue_q], dim=-1)
        if is_mx and result.shape[-1] < self.weight.shape[-1]:
            result = F.pad(result, (0, self.weight.shape[-1] - result.shape[-1]))
        return result.to(input.dtype) if is_mx else result

    def _make_residue_weight(
        self,
        input: torch.Tensor,
        input_ext: torch.Tensor,
        salient_indices: torch.Tensor,
    ) -> torch.Tensor:
        weight = self.weight.data if isinstance(self.weight, torch.nn.Parameter) else self.weight
        if weight.shape[-1] == input_ext.shape[-1]:
            return weight
        if weight.shape[-1] != input.shape[-1]:
            raise RuntimeError(
                "Residue weight/input shape mismatch: "
                f"weight K={weight.shape[-1]}, input K={input.shape[-1]}, "
                f"extended input K={input_ext.shape[-1]}."
            )
        weight_indices = salient_indices.to(device=weight.device)
        return torch.cat([weight, weight.index_select(-1, weight_indices)], dim=-1)

    def _apply_weight_pre_quant_scale(
        self,
        weight: torch.Tensor,
        salient_indices: torch.Tensor,
    ) -> torch.Tensor:
        pre_quant_scale = self.weight_quantizer.pre_quant_scale
        if pre_quant_scale is None:
            return weight

        scale = pre_quant_scale.to(device=weight.device, dtype=weight.dtype)
        if scale.ndim == 0 or scale.shape[-1] == 1:
            return weight * scale
        if scale.ndim == 1:
            scale = scale.unsqueeze(0)
        if scale.shape[-1] == weight.shape[-1]:
            return weight * scale
        original_width = getattr(self, "_original_in_features", self.in_features)
        if scale.shape[-1] == original_width:
            scale_indices = salient_indices.to(device=scale.device, dtype=torch.long)
            scale = torch.cat([scale, scale.index_select(-1, scale_indices)], dim=-1)
            scale = F.pad(scale, (0, weight.shape[-1] - scale.shape[-1]), value=1)
            return weight * scale
        raise RuntimeError(
            "weight_quantizer.pre_quant_scale shape does not match residue weight: "
            f"scale K={scale.shape[-1]}, weight K={weight.shape[-1]}, "
            f"salient={salient_indices.numel()}."
        )

    def _forward_residue_qdq(self, input: torch.Tensor) -> torch.Tensor:
        salient_indices = self.input_quantizer.salient_indices
        assert salient_indices is not None and salient_indices.numel() > 0

        input_ext_q = self._make_residue_input_qdq(input, salient_indices)
        weight = self._make_residue_weight(input, input_ext_q, salient_indices)
        weight = self._apply_weight_pre_quant_scale(weight, salient_indices)
        with (
            self.weight_quantizer.disable_pre_quant_scale(),
            self.weight_quantizer.allow_residue_linear_fake_quant(),
        ):
            if (
                self.weight_quantizer.block_sizes.get("scale_bits") == (8, 0)
                and self.weight_quantizer.is_enabled
                and self.weight_quantizer.fake_quant
                and getattr(self.weight_quantizer, "_if_quant", True)
            ):
                # The packer flattens blocks; pad each row so blocks cannot cross rows.
                padding = -weight.shape[-1] % 32
                # Bound the packer's seven-threshold temporaries to a chunk of rows.
                # Larger CUDA chunks amortize kernel launches.
                chunk_elements = 2**22 if weight.is_cuda else 2**18
                rows_per_chunk = max(1, chunk_elements // (weight.shape[-1] + padding))
                weight_q = torch.empty_like(weight)
                for start in range(0, weight.shape[0], rows_per_chunk):
                    chunk = weight[start : start + rows_per_chunk]
                    padded = F.pad(chunk, (0, padding)) if padding else chunk.contiguous()
                    packed, scale = MXFP4QTensor.quantize(padded, 32)
                    dequantized = packed.dequantize(scale=scale, block_sizes={-1: 32})
                    weight_q[start : start + rows_per_chunk] = dequantized[..., : weight.shape[-1]]
            else:
                weight_q = self.weight_quantizer(weight)

        output = F.linear(input_ext_q, weight_q, self.bias)
        return self.output_quantizer(output) if hasattr(self, "output_quantizer") else output

    def forward(self, input, *args, **kwargs):
        """Apply residue QDQ, or collect statistics before channel selection."""
        input_has_salient_attr = hasattr(self.input_quantizer, "salient_indices")
        weight_has_salient_attr = hasattr(self.weight_quantizer, "salient_indices")
        if not input_has_salient_attr and not weight_has_salient_attr:
            # ModelOpt auto-search can activate a plain NVFP4 candidate on the
            # same module. In that case quantizers are regular TensorQuantizer
            # instances and the layer should behave exactly like QuantLinear.
            return super().forward(input, *args, **kwargs)
        if input_has_salient_attr != weight_has_salient_attr:
            raise RuntimeError(
                "Mixed residue/plain quantizer state is invalid: "
                f"input_has_salient_attr={input_has_salient_attr}, "
                f"weight_has_salient_attr={weight_has_salient_attr}."
            )

        salient_indices = self.input_quantizer.salient_indices
        has_salient_indices = salient_indices is not None and salient_indices.numel() > 0

        if not has_salient_indices:
            weight = (
                self.weight.data if isinstance(self.weight, torch.nn.Parameter) else self.weight
            )
            if weight.shape[-1] > input.shape[-1]:
                raise RuntimeError(
                    "Extended residue weight requires input_quantizer.salient_indices "
                    "during forward, but none were loaded."
                )
            active_fake_quant = (
                self.input_quantizer.is_enabled
                and getattr(self.input_quantizer, "_if_quant", True)
                and self.input_quantizer.fake_quant
            ) or (
                self.weight_quantizer.is_enabled
                and getattr(self.weight_quantizer, "_if_quant", True)
                and self.weight_quantizer.fake_quant
            )
            if active_fake_quant and salient_indices is not None:
                with (
                    self.input_quantizer.allow_residue_linear_fake_quant(),
                    self.weight_quantizer.allow_residue_linear_fake_quant(),
                ):
                    return super().forward(input, *args, **kwargs)
            if active_fake_quant:
                raise RuntimeError(
                    "Residue fake-quant forward requires salient_indices. "
                    "This fallback is only valid for calibration collect / normal forward."
                )
            # Keep the normal ModelOpt QuantLinear path for residue calibration's
            # statistics-collection forward. Before postprocess there is no
            # salience metadata or K extension to apply.
            return super().forward(input, *args, **kwargs)

        return self._forward_residue_qdq(input)

    def _validate_fold_weight(self):
        raise NotImplementedError("Residue weight folding requires paired activation/weight QDQ.")

    def fold_weight(self, keep_attrs: bool = False):
        """Reject folding until paired residue weight folding is supported."""
        self._validate_fold_weight()

    @torch.no_grad()
    def materialize_residue_weight(self):
        """Store copied channels explicitly; calibration leaves weights unextended."""
        if hasattr(self, "_original_in_features"):
            return
        indices = self.input_quantizer.salient_indices
        if indices is None:
            raise RuntimeError("Calibrate residue channels before materializing weights.")
        if indices.numel() == 0:
            return
        # Allocate before replacing the parameter so a failed extension leaves it intact.
        weight = self.weight_quantizer.copy_extension(self.weight, indices)
        if self.weight_quantizer.block_sizes.get("scale_bits") == (8, 0):
            weight = F.pad(weight, (0, -weight.shape[-1] % 32))
        self._original_in_features = self.in_features
        self.weight = nn.Parameter(weight, requires_grad=self.weight.requires_grad)
        self.in_features = weight.shape[-1]
