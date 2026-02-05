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

"""Utility to inspect model structure to help implement ModelDescriptor.layer_structure().

This tool loads a model on meta device and displays:
1. Global parameters (embeddings, lm_head, etc.) with their module classes
2. Layer parameters grouped by layer, showing:
   - Layer class (for decoder_layer_cls())
   - Submodule classes (for module_classes in layer_structure())
   - Parameter names (for include_by_name in layer_structure())

Usage:
    python modelopt/torch/puzzletron/anymodel/tools/inspect_model_structure.py \\
        --model_path /path/to/model \\
        --num_layers 3
"""

import argparse
import re

import torch.nn as nn
from transformers import AutoModelForCausalLM


def inspect_model(model_path: str, num_layers: int = 3):
    """Inspect model structure by showing layer submodules and parameters.

    Args:
        model_path: Path to the model checkpoint
        num_layers: Number of decoder layers to display (useful for hybrid models)
    """
    print(f"\n{'=' * 80}")
    print(f"Model: {model_path}")
    print(f"{'=' * 80}\n")

    # Load model on meta device (fast, no memory needed)
    print("Loading model structure (device_map='meta')...\n")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="meta",
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Pattern to detect layer indices: either ".N." in middle or ends with ".N"
    layer_index_pattern = re.compile(r"\.\d+(\.|$)")

    # Build module name -> class name mapping
    module_map = {name: type(module).__name__ for name, module in model.named_modules()}

    # Collect all state dict keys
    state_dict_keys = list(model.state_dict().keys())

    # Separate global vs layer-specific parameters
    global_params = [k for k in state_dict_keys if not layer_index_pattern.search(k)]

    # Display global parameters
    print(f"{'=' * 80}")
    print(f"GLOBAL PARAMETERS:")
    print(f"{'=' * 80}")
    print(f"{'Parameter Name':<70} {'Module Class':<20}")
    print("-" * 90)
    for key in global_params:
        # Get parent module
        parts = key.rsplit(".", 1)
        module_class = module_map.get(parts[0], "?") if len(parts) == 2 else "?"
        print(f"{key:<70} {module_class:<20}")

    # Display layer parameters for each requested layer
    print(f"\n{'=' * 80}")
    if num_layers == 1:
        print(f"LAYER 0 PARAMETERS:")
    else:
        print(f"LAYER PARAMETERS (layers 0-{num_layers - 1}):")
    print(f"{'=' * 80}")

    for layer_idx in range(num_layers):
        # Find layer class
        layer_pattern = re.compile(rf"\.{layer_idx}$")
        layer_module_name = None
        layer_class = None
        for name, class_name in module_map.items():
            if layer_pattern.search(name):
                layer_module_name = name
                layer_class = class_name
                break

        # Print layer header
        if layer_class:
            print(f"\n# Layer {layer_idx}: {layer_class}")
        else:
            print(f"\n# Layer {layer_idx}")

        # Find parameters for this layer
        pattern = re.compile(rf"\.{layer_idx}(\.|$)")
        layer_params = [k for k in state_dict_keys if pattern.search(k)]

        # Group parameters by direct child module
        current_submodule = None
        for param in layer_params:
            # Extract direct child module path (e.g., "model.layers.0.self_attn")
            match = re.search(rf"(.*\.{layer_idx}\.[^.]+)", param)
            if match:
                submodule_path = match.group(1)

                # If this is a new submodule, print its class
                if submodule_path != current_submodule:
                    current_submodule = submodule_path
                    submodule_class = module_map.get(submodule_path, "?")
                    # Extract just the submodule name (e.g., "self_attn")
                    submodule_name = submodule_path.split(".")[-1]
                    print(f"  # {submodule_name} → {submodule_class}")

            print(f"  {param}")


def main():
    parser = argparse.ArgumentParser(
        description="Inspect model structure to help implement layer_structure()",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to model checkpoint (HuggingFace format)",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=3,
        help="Number of decoder layers to display (useful for hybrid models with different layer structures)",
    )

    args = parser.parse_args()
    inspect_model(args.model_path, args.num_layers)


if __name__ == "__main__":
    main()
