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

"""PEFT/LoRA plugins for various frameworks."""

from modelopt.torch.utils import import_plugin

with import_plugin("megatron"):
    # Import TE-grouped MoE plugin BEFORE the non-grouped Megatron plugin so its
    # registrations land first in LoRAModuleRegistry. The registry's resolution rule
    # (modelopt/torch/opt/dynamic.py:_get_registered_nn_class) picks the first
    # registered class whose `forward` identity-matches the target's forward; both
    # the TE-grouped and non-grouped Megatron quant classes inherit their forward
    # from _QuantFunctionalMixin (modelopt/torch/quantization/plugins/custom.py),
    # so they would otherwise tie and the earlier-registered one would win. The
    # non-grouped lookup is unaffected because its `issubclass` check still excludes
    # TE-grouped targets.
    from .megatron_moe import *
    from .megatron import *
