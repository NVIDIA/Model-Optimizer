# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native NeMo AutoModel bridge for Puzzletron AnyModel checkpoints."""

# Import built-ins for registration. Descriptor modules keep NeMo imports lazy.
from . import models  # noqa: F401
from .auto_model_descriptor import (
    AutoModelDescriptor,
    AutoModelDescriptorFactory,
    ContractAutoModelDescriptor,
)
from .patcher import automodel_patcher

__all__ = [
    "AutoModelDescriptor",
    "AutoModelDescriptorFactory",
    "ContractAutoModelDescriptor",
    "automodel_patcher",
]
