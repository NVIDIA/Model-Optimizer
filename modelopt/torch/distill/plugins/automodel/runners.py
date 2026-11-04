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

"""Family-selected replay implementations; native routing remains authoritative."""

from modelopt.torch.quantization.repair_formats import native_scale_execution

__all__ = ["native_fast_replay"]


def native_fast_replay(module, args, kwargs):
    """Run native grouped experts with fused differentiable IQ scale arithmetic.

    This preserves the model's router, shared experts, clipping and dispatch.
    It never substitutes another family's MoE math or quantization format.
    """
    with native_scale_execution():
        return module(*args, **kwargs)
