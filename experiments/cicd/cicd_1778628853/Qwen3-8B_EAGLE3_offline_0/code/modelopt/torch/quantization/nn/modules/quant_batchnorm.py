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

"""Quantized batch normalization module."""

import torch.nn as nn

from .quant_module import QuantInputBase, QuantModuleRegistry

QuantModuleRegistry.register({nn.BatchNorm1d: "nn.BatchNorm1d"})(QuantInputBase)
QuantModuleRegistry.register({nn.BatchNorm2d: "nn.BatchNorm2d"})(QuantInputBase)
QuantModuleRegistry.register({nn.BatchNorm3d: "nn.BatchNorm3d"})(QuantInputBase)
