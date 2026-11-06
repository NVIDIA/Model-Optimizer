# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Kimi K3.1 exposes KDA and sparse attention under the same child name."""

from .base import TransformersFamilyAdapter
from .registry import register_transformers_family

__all__ = []


class _KimiK31Family(TransformersFamilyAdapter):
    def unit_attributes(self, block, unit_types):
        result = []
        for role in unit_types:
            if role in ("attention", "gdn"):
                if (role == "gdn") == bool(block.is_kda):
                    result.append("self_attn")
            else:
                result.extend(super().unit_attributes(block, (role,)))
        return list(dict.fromkeys(result))


register_transformers_family(_KimiK31Family("kimi_k31", ("model.layers",)))
