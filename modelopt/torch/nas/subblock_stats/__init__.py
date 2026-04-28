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
"""Subblock runtime statistics API for ModelOpt NAS.

This module provides utilities for measuring and calculating runtime statistics
of subblocks (e.g., Attention, FFN) within transformer architectures.

Primary API:
    - calc_runtime_for_subblocks: Empirically measures runtime for candidate subblock configurations
"""
from .calc_runtime_stats import calc_runtime_for_subblocks
