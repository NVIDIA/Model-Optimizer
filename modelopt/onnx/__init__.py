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

"""Model optimization subpackage for onnx."""

import sys
import warnings

MIN_PYTHON_VERSION = (3, 10)

try:
    from . import quantization
    from .logging_config import configure_logging, logger
except ImportError as e:
    raise ImportError(f"{e}\nPlease install optional ``[onnx]`` dependencies.")


if sys.version_info < (3, 11):
    warnings.warn(
        "`modelopt.onnx` package will drop python<3.11 support in a future release",
        DeprecationWarning,
    )

# Check the current Python version
if sys.version_info < MIN_PYTHON_VERSION:
    logger.warning(
        f"This package requires Python {MIN_PYTHON_VERSION[0]}.{MIN_PYTHON_VERSION[1]} or higher. "
        f"You are using Python {sys.version_info[0]}.{sys.version_info[1]}",
    )
