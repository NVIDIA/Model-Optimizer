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

"""Modules with quantization support."""

from .modules.quant_activations import *
from .modules.quant_batchnorm import *
from .modules.quant_conv import *
from .modules.quant_instancenorm import *
from .modules.quant_linear import *
from .modules.quant_module import *
from .modules.quant_pooling import *
from .modules.quant_rnn import *
from .modules.tensor_quantizer import *
