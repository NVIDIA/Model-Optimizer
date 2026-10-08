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

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_FASTGEN_DIR = os.path.dirname(_THIS_DIR)
if _FASTGEN_DIR not in sys.path:
    sys.path.insert(0, _FASTGEN_DIR)

_HELP = """\
usage: finetune.py [--config CONFIG] [CONFIG_OVERRIDE ...]

DMD2 Qwen-Image training with NeMo AutoModel.

options:
  -h, --help       show this help message and exit
  --config CONFIG  YAML config path (default:
                   examples/diffusers/fastgen/dmd2/configs/qwen_image.yaml)

Additional dotted AutoModel config overrides are forwarded unchanged.
"""


def main(
    default_config_path: str = "examples/diffusers/fastgen/dmd2/configs/qwen_image.yaml",
) -> None:
    if any(argument in {"-h", "--help"} for argument in sys.argv[1:]):
        print(_HELP, end="")
        return

    from nemo_automodel.components.config._arg_parser import parse_args_and_load_config

    from dmd2.recipe import DMD2DiffusionRecipe

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
