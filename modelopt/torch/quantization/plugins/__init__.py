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

"""Handles quantization plugins to correctly quantize third-party modules.

Please check out the source code of this module for examples of how plugins work and how you can
write your own one. Currently, we support plugins for

- :meth:`apex<modelopt.torch.quantization.plugins.apex>`
- :meth:`attention<modelopt.torch.quantization.plugins.attention>`
- :meth:`diffusers<modelopt.torch.quantization.plugins.diffusers>`
- :meth:`fairscale<modelopt.torch.quantization.plugins.fairscale>`
- :meth:`huggingface<modelopt.torch.quantization.plugins.huggingface>`
- :meth:`megatron<modelopt.torch.quantization.plugins.megatron>`
- :meth:`peft<modelopt.torch.quantization.plugins.peft>`
- :meth:`pytorch_geometric<modelopt.torch.quantization.plugins.pytorch_geometric>`
- :meth:`transformer_engine<modelopt.torch.quantization.plugins.transformer_engine>`
"""

from modelopt.torch.utils import import_plugin

with import_plugin("accelerate"):
    from .accelerate import *

with import_plugin("apex"):
    from .apex import *

from .attention import *
from .custom import *

with import_plugin("diffusers"):
    from .diffusion.diffusers import *

with import_plugin("fairscale"):
    from .fairscale import *

with import_plugin("huggingface"):
    from .huggingface import *

with import_plugin("megatron"):
    from .megatron import *

with import_plugin("nemo"):
    from .nemo import *

with import_plugin("peft"):
    from .peft import *

with import_plugin("torch_geometric"):
    from .pytorch_geometric import *

with import_plugin("transformer_engine"):
    from .transformer_engine import *

with import_plugin("transformers trainer"):
    from .transformers_trainer import *

with import_plugin("transformers"):
    from .transformers import *

with import_plugin("vllm"):
    from .vllm import *

with import_plugin("trl"):
    from .trl import *

with import_plugin("fastvideo"):
    from .diffusion.fastvideo import *
