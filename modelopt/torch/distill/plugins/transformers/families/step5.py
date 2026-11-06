# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Step 5 owns its shared expert beside the routed MoE boundary."""

from .base import TransformersFamilyAdapter
from .registry import register_transformers_family

__all__ = []


class _Step5Family(TransformersFamilyAdapter):
    def unit_attributes(self, block, unit_types):
        result = super().unit_attributes(block, unit_types)
        if "mlp" in unit_types and block._modules.get("share_expert") is not None:
            result.append("share_expert")
        return list(dict.fromkeys(result))


register_transformers_family(_Step5Family("step5", ("model.layers",)))
