# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""This file contains utility functions used by the quantization backend."""

import torch


def fp8_compatible():
    """Check if the current device supports FP8."""
    return torch.cuda.get_device_capability(0) >= (8, 9)


def fp4_compatible():
    """Check if the current device supports FP4."""
    return torch.cuda.get_device_capability(0) >= (10, 0)
