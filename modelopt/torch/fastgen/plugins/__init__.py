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

"""Optional plugins for the fastgen subpackage (gated via ``import_plugin``).

``qwen_image`` holds the Qwen-Image pipeline plus the forward-hook helpers that expose
intermediate teacher activations to the DMD2 GAN discriminator. The import is gated so
environments that choose not to install the optional fastgen dependencies still see a
clean package import.
"""

from modelopt.torch.utils import import_plugin

with import_plugin("qwen_image"):
    from .qwen_image import *
