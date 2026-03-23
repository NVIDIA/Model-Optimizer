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

# Copyright 2024 Nvidia Corporation, Google Inc, HuggingFace Inc, EleutherAI. All rights reserved.
#
# Small nn helpers for puzzletron pipeline code. Model configs come from HuggingFace ``AutoConfig`` (AnyModel).
# ``LMHead`` is a distinct ``nn.Linear`` subclass so pipeline / FSDP code can target it explicitly
# (see ``validate_runtime_pipeline``).
# mypy: ignore-errors

from torch import nn


class LMHead(nn.Linear):
    """
    Special class to allow FSDP wrapping without affecting other Linear layers in the model.
    """
