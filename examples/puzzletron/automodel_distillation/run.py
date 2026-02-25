# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Single entrypoint for AnyModel training pipelines.

Modes: pretrain | kd

Run from this directory so that patch_automodel and recipe are importable. Example:
  torchrun --nproc_per_node=2 -m run --mode pretrain -c ./pretrain.yaml
  torchrun --nproc_per_node=2 -m run --mode kd -c ./kd.yaml

If -c is omitted, a default config path is used per mode.
"""

from __future__ import annotations

import sys

from nemo_automodel.components.config._arg_parser import parse_args_and_load_config
from patch_automodel import apply_patch

# Default config path per mode (used when -c is not passed)
_DEFAULT_CONFIG = {
    "pretrain": "./pretrain.yaml",
    "kd": "./kd.yaml",
}


def _parse_mode() -> str:
    """Parse --mode <mode> from argv; remove it so parse_args_and_load_config does not see it."""
    argv = sys.argv[1:]
    mode = None
    new_argv = []
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok in ("--mode", "-m"):
            if i + 1 >= len(argv):
                raise ValueError("Expected a value after --mode (pretrain | kd)")
            mode = argv[i + 1]
            i += 2
            continue
        new_argv.append(tok)
        i += 1
    if mode is None:
        raise ValueError(
            "Missing --mode. Choose one of: pretrain, kd. "
            "Example: python -m run --mode kd -c kd.yaml"
        )
    if mode not in _DEFAULT_CONFIG:
        raise ValueError(f"Invalid mode '{mode}'. Choose one of: pretrain, kd")
    sys.argv = [sys.argv[0], *new_argv]
    return mode


def main() -> None:
    mode = _parse_mode()
    default_config = _DEFAULT_CONFIG[mode]
    apply_patch()
    cfg = parse_args_and_load_config(default_config)

    if mode == "pretrain":
        from nemo_automodel.recipes.llm.train_ft import TrainFinetuneRecipeForNextTokenPrediction

        recipe = TrainFinetuneRecipeForNextTokenPrediction(cfg)
    elif mode == "kd":
        from recipe import KnowledgeDistillationRecipeForNextTokenPrediction

        recipe = KnowledgeDistillationRecipeForNextTokenPrediction(cfg)
    else:
        raise ValueError(f"Invalid mode '{mode}'. Choose one of: pretrain, kd")

    recipe.setup()
    recipe.run_train_validation_loop()


if __name__ == "__main__":
    main()
