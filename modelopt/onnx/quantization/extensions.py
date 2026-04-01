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

"""Module to load C++ extensions."""

import os
import sys

# TODO: cppimport is no longer maintained, switch to a different library
import cppimport

from modelopt.onnx.logging_config import logger

try:
    logger.info("Loading extension modelopt_round_and_pack_ext...")
    path = os.path.join(os.path.dirname(__file__), "src")
    sys.path.append(path)
    round_and_pack_ext = cppimport.imp("modelopt_round_and_pack_ext")
    sys.path.remove(path)
except Exception as e:
    logger.warning(
        f"{e}\nUnable to load `modelopt_round_and_pack_ext', falling back to python based optimized version. "
        "If you see `copy_file() got an unexpected keyword argument 'dry_run'`, you will need "
        "https://github.com/tbenthompson/cppimport/pull/98 or downgrade setuptools until we have a workaround"
    )
    round_and_pack_ext = None
