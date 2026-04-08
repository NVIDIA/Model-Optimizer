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

"""Compatibility shim for ONNX >= 1.20.

ONNX 1.20 removed several deprecated helper functions (float32_to_bfloat16,
float32_to_float8e4m3, etc.). Some downstream packages like onnx_graphsurgeon 0.5.x
still reference these at import time. This module restores them using ml_dtypes so
that onnx_graphsurgeon can be imported without errors.

This module must be imported BEFORE onnx_graphsurgeon.
"""


def patch_onnx_helper_removed_apis():
    """Restore removed ONNX helper functions for backward compatibility."""
    try:
        import onnx.helper

        if not hasattr(onnx.helper, "float32_to_bfloat16"):
            import ml_dtypes
            import numpy as np

            def _float32_to_bfloat16(value):
                return int.from_bytes(
                    np.float32(value).astype(ml_dtypes.bfloat16).tobytes(), "little"
                )

            onnx.helper.float32_to_bfloat16 = _float32_to_bfloat16

        if not hasattr(onnx.helper, "float32_to_float8e4m3"):
            import ml_dtypes
            import numpy as np

            def _float32_to_float8e4m3(value, fn=True, uz=False):
                if fn and not uz:
                    dtype = ml_dtypes.float8_e4m3fn
                elif fn and uz:
                    dtype = ml_dtypes.float8_e4m3fnuz
                else:
                    dtype = ml_dtypes.float8_e4m3fn
                return int(np.float32(value).astype(dtype).view(np.uint8))

            onnx.helper.float32_to_float8e4m3 = _float32_to_float8e4m3

    except ImportError:
        pass


patch_onnx_helper_removed_apis()
