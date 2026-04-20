# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Shared aux-layer selection helpers for compute_hidden_states_* scripts.

Supports a shared ``--aux-layers`` flag that accepts:

- ``"eagle"``  — ``default_eagle_aux_layer_ids(num_hidden_layers)`` from modelopt.
- ``"dflash"`` — ``build_target_layer_ids(num_hidden_layers, 5)`` from modelopt
  (5 draft layers; use the explicit list form for other counts).
- ``"2,5,8"`` — explicit comma-separated layer indices.

Convention: returned values are **0-based transformer layer IDs**. Callers that
index into HuggingFace's ``outputs.hidden_states`` tuple must add ``+1`` because
``hidden_states[0]`` is the embedding output. Callers passing layer IDs to
TRT-LLM / vLLM configs should use the values directly.
"""

import argparse

_DFLASH_DEFAULT_NUM_DRAFT_LAYERS = 5


def add_aux_layers_args(parser: argparse.ArgumentParser) -> None:
    """Register the ``--aux-layers`` flag on ``parser``."""
    parser.add_argument(
        "--aux-layers",
        type=str,
        default="eagle",
        help=(
            "Aux layer indices to capture. One of: "
            "'eagle' (EAGLE-3 default from modelopt), "
            f"'dflash' ({_DFLASH_DEFAULT_NUM_DRAFT_LAYERS}-layer DFlash default from modelopt), "
            "or a comma-separated list like '2,5,8' to override. Default: eagle."
        ),
    )


def resolve_aux_layers(args: argparse.Namespace, num_hidden_layers: int) -> list[int]:
    """Resolve ``args.aux_layers`` to a sorted, de-duped list of 0-based layer IDs."""
    value = args.aux_layers.strip().lower()
    if value == "eagle":
        from modelopt.torch.speculative.plugins.hf_eagle import default_eagle_aux_layer_ids

        return default_eagle_aux_layer_ids(num_hidden_layers)
    if value == "dflash":
        from modelopt.torch.speculative.plugins.modeling_dflash import build_target_layer_ids

        return sorted(
            set(build_target_layer_ids(num_hidden_layers, _DFLASH_DEFAULT_NUM_DRAFT_LAYERS))
        )
    try:
        indices = [int(tok) for tok in args.aux_layers.split(",") if tok.strip()]
    except ValueError as e:
        raise ValueError(
            f"--aux-layers must be 'eagle', 'dflash', or a comma-separated int list, "
            f"got: {args.aux_layers!r}"
        ) from e
    if not indices:
        raise ValueError(f"--aux-layers int list is empty: {args.aux_layers!r}")
    for i in indices:
        if not 0 <= i < num_hidden_layers:
            raise ValueError(f"--aux-layers index {i} out of range [0, {num_hidden_layers})")
    return sorted(set(indices))
