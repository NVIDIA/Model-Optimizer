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

# Import models to trigger factory registration
from modelopt.torch.puzzletron.anymodel.models.gpt_oss import *
from modelopt.torch.puzzletron.anymodel.models.llama import *
from modelopt.torch.puzzletron.anymodel.models.mistral_small import *
from modelopt.torch.puzzletron.anymodel.models.nemotron_h import *
from modelopt.torch.puzzletron.anymodel.models.nemotron_h_v2 import *
from modelopt.torch.puzzletron.anymodel.models.qwen2 import *
from modelopt.torch.puzzletron.anymodel.models.qwen3 import *
from modelopt.torch.puzzletron.anymodel.models.qwen3_vl import *
