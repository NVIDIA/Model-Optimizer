# Adapted from: https://github.com/huggingface/optimum/blob/0d808ad/optimum/onnx/utils.py

#  Copyright 2022 The HuggingFace Team. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

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

import onnx
from onnx.external_data_helper import ExternalDataInfo, _get_initializer_tensors


def _get_onnx_external_data_tensors(model: onnx.ModelProto) -> list[str]:
    """
    Gets the paths of the external data tensors in the model.
    Note: make sure you load the model with load_external_data=False.
    """
    model_tensors = _get_initializer_tensors(model)
    model_tensors_ext = [
        ExternalDataInfo(tensor).location
        for tensor in model_tensors
        if tensor.HasField("data_location") and tensor.data_location == onnx.TensorProto.EXTERNAL
    ]
    return model_tensors_ext
