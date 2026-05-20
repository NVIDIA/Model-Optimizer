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

"""Entrypoint for the DMD2 Wan 2.2 5B AutoModel example.

Parses the YAML config + CLI overrides with AutoModel's argument parser, then hands
control to :class:`DMD2DiffusionRecipe`.
"""

from __future__ import annotations

from dmd2_recipe import DMD2DiffusionRecipe
from nemo_automodel.components.config._arg_parser import parse_args_and_load_config


def main(
    default_config_path: str = "examples/diffusers/fastgen/configs/dmd2_wan22_5b.yaml",
) -> None:
    cfg = parse_args_and_load_config(default_config_path)
    recipe = DMD2DiffusionRecipe(cfg)
    recipe.setup()
    recipe.run_train_validation_loop()


if __name__ == "__main__":
    main()
