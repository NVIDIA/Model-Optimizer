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

"""Base class for ONNX quantizer exporters."""

from abc import ABC, abstractmethod

import onnx


class ONNXQuantExporter(ABC):
    """Base class for ONNX quantizer exporters."""

    @classmethod
    def process_model(cls, onnx_model: onnx.ModelProto) -> onnx.ModelProto:
        """Processes the ONNX model."""
        onnx_model = cls.pre_process(onnx_model)
        onnx_model = cls.compute_scales(onnx_model)
        onnx_model = cls.compress_weights(onnx_model)
        onnx_model = cls.post_process(onnx_model)
        return onnx_model

    @staticmethod
    @abstractmethod
    def pre_process(onnx_model: onnx.ModelProto) -> onnx.ModelProto:
        """Pre-processes the ONNX model. Converts all DQ -> * -> op patterns to DQ -> op."""

    @staticmethod
    @abstractmethod
    def compute_scales(onnx_model: onnx.ModelProto) -> onnx.ModelProto:
        """Computes the scales for the weights in the ONNX model."""

    @staticmethod
    @abstractmethod
    def compress_weights(onnx_model: onnx.ModelProto) -> onnx.ModelProto:
        """Compresses the weights in the ONNX model."""

    @staticmethod
    @abstractmethod
    def post_process(onnx_model: onnx.ModelProto) -> onnx.ModelProto:
        """Post-processes the ONNX model."""
