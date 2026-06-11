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

"""Entrypoint for the DMD2 Qwen-Image AutoModel example.

Parses the YAML config + CLI overrides with AutoModel's argument parser, then hands
control to :class:`DMD2DiffusionRecipe`.
"""

from __future__ import annotations

import logging
import os
import sys

# Make this example directory importable as top-level modules (``dmd2_recipe``,
# ``fastgen_data``, ``fastgen_checkpoint``) regardless of the current working directory, so
# the configs' short ``_target_: fastgen_data.build_*`` resolve from a source checkout.
# (Python already puts the script's directory on ``sys.path[0]`` when run as
# ``python .../dmd2_finetune.py``; this makes that explicit and robust to other invocations.)
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from dmd2_recipe import DMD2DiffusionRecipe  # noqa: E402
from nemo_automodel.components.config._arg_parser import parse_args_and_load_config  # noqa: E402


def main(
    default_config_path: str = "examples/diffusers/fastgen/configs/dmd2_qwen_image_smoke.yaml",
) -> None:
    cfg = parse_args_and_load_config(default_config_path)

    # Surface where the data package and ``nemo_automodel`` resolve from, so a misconfigured
    # environment (e.g. a sibling Automodel source checkout shadowing the installed package)
    # is obvious at startup.
    import fastgen_data
    import nemo_automodel

    logging.info(
        "[fastgen] vendored data package: %s",
        os.path.dirname(os.path.abspath(fastgen_data.__file__)),
    )
    logging.info(
        "[fastgen] nemo_automodel resolved from: %s",
        os.path.realpath(nemo_automodel.__file__),
    )

    recipe = DMD2DiffusionRecipe(cfg)
    recipe.setup()
    recipe.run_train_validation_loop()


if __name__ == "__main__":
    main()
