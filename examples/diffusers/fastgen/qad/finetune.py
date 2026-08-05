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

"""Entrypoint for FastGen quantization-aware distillation."""

from __future__ import annotations

import os
import sys

_QAD_DIR = os.path.dirname(os.path.abspath(__file__))
_FASTGEN_DIR = os.path.dirname(_QAD_DIR)
if _FASTGEN_DIR not in sys.path:
    sys.path.insert(0, _FASTGEN_DIR)

from nemo_automodel.components.config._arg_parser import parse_args_and_load_config  # noqa: E402

from qad.recipe import QADDiffusionRecipe  # noqa: E402


def main(
    default_config_path: str = ("examples/diffusers/fastgen/qad/configs/qwen_image_nvfp4.yaml"),
) -> None:
    cfg = parse_args_and_load_config(default_config_path)
    recipe = QADDiffusionRecipe(cfg)
    recipe.setup()
    recipe.run_train_validation_loop()


if __name__ == "__main__":
    main()
