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

r"""Compare module output tensors from different model variants.

This module provides:
1. OutputSaveHook - A PyTorch hook to capture module outputs during forward pass
2. Comparison utilities - Compute RMSE and cosine similarity between saved outputs

Usage Example:
--------------

Step 1: Capture outputs from multiple layers:

    from modelopt.torch.nas.plugins.megatron_hooks.compare_module_outputs import (
        OutputSaveHook,
        save_multi_layer_outputs,
    )

    # Register hooks on all target layers
    hooks = {}
    for name, module in model.named_modules():
        if name.endswith('mlp.linear_fc2'):
            hook = OutputSaveHook(layer_name=name)
            module.register_forward_hook(hook)
            hooks[name] = hook

    # Run inference/training
    model(input_data)

    # Save all layer outputs
    save_multi_layer_outputs(hooks, "output_unpruned.pt")

Step 2: Compare outputs from different model variants:

    python compare_module_outputs.py \
        --reference output_unpruned.pt \
        --compare output_l2norm.pt \
        --output-json comparison_stats.json

The saved file format:
{
    'decoder.layers.0.mlp.linear_fc2': Tensor([steps, seq_len, batch, hidden]),
    'decoder.layers.1.mlp.linear_fc2': Tensor([...]),
    ...
    'metadata': {'num_layers': N, 'num_steps': M, 'layer_names': [...]}
}
"""

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F


class OutputSaveHook:
    """Hook to capture and save module outputs during forward pass."""

    def __init__(self, layer_name: str) -> None:
        """Initialize the output save hook.

        Args:
            layer_name: Hierarchical name of the layer (e.g., 'decoder.layers.0.mlp.linear_fc2').
        """
        self.layer_name = layer_name
        self.saved_outputs: list[torch.Tensor] = []

    def __call__(
        self,
        module: nn.Module,
        args: tuple[torch.Tensor, ...],
        output: torch.Tensor | tuple[torch.Tensor, ...],
    ) -> None:
        """Capture and save module output during forward pass.

        Args:
            module: The PyTorch module being hooked.
            args: Input arguments to the module's forward pass.
            output: Output tensor(s) from the module's forward pass.
        """
        # Handle tuple outputs (e.g., output, bias)
        out = output[0] if isinstance(output, tuple) else output
        self.saved_outputs.append(out.detach().cpu())

    def get_stacked_outputs(self) -> torch.Tensor:
        """Stack all saved outputs into a single tensor."""
        return torch.stack(self.saved_outputs)


def save_multi_layer_outputs(hooks: dict[str, OutputSaveHook], path: str) -> None:
    """Save outputs from multiple layers to a single file.

    Args:
        hooks: Dictionary mapping layer names to their hooks.
        path: Path to save the outputs.
    """
    output_dict = {name: hook.get_stacked_outputs() for name, hook in hooks.items()}

    # Add metadata
    output_dict["metadata"] = {
        "num_layers": len(hooks),
        # Number of forward passes (generation steps) - all hooks have same count, so use first hook
        "num_steps": len(next(iter(hooks.values())).saved_outputs) if hooks else 0,
        "layer_names": list(hooks.keys()),
    }

    torch.save(output_dict, path)
    print(f"\nSaved outputs from {len(hooks)} layers to {path}")
    for name, tensor in output_dict.items():
        if name != "metadata":
            print(f"  {name}: {tensor.shape}")


def compute_rmse(tensor1: torch.Tensor, tensor2: torch.Tensor) -> float:
    """Compute Root Mean Square Error between two tensors."""
    mse = torch.mean((tensor1 - tensor2) ** 2)
    rmse = torch.sqrt(mse)
    return rmse.item()


def compute_cosine_similarity(tensor1: torch.Tensor, tensor2: torch.Tensor) -> dict:
    """Compute average cosine similarity between two tensors."""
    # Flatten to 2D for cosine similarity computation
    t1_flat = tensor1.reshape(-1, tensor1.shape[-1])
    t2_flat = tensor2.reshape(-1, tensor2.shape[-1])

    # Compute cosine similarity per position
    cos_sim = F.cosine_similarity(t1_flat, t2_flat, dim=-1)

    return {
        "mean": cos_sim.mean().item(),
        "min": cos_sim.min().item(),
        "max": cos_sim.max().item(),
        "std": cos_sim.std().item(),
    }


def main():
    """Compare module output tensors from different model variants."""
    parser = argparse.ArgumentParser(
        description="Compare module output tensors from different model variants",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--reference",
        type=str,
        required=True,
        help="Path to reference output tensor (e.g., unpruned model)",
    )
    parser.add_argument(
        "--compare",
        type=str,
        required=True,
        help="Path to output tensor to compare against reference",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Path to save comparison statistics as JSON",
    )
    args = parser.parse_args()

    # Load reference data
    print(f"\nLoading reference: {args.reference}")
    ref_data = torch.load(args.reference, map_location="cpu")

    # Load comparison data
    print(f"Loading compare: {args.compare}")
    comp_data = torch.load(args.compare, map_location="cpu")

    # Compare multi-layer outputs
    compare_multi_layer(ref_data, comp_data, args.output_json)


def compare_multi_layer(ref_data: dict, comp_data: dict, output_json: str | None = None):
    """Compare multi-layer outputs."""
    import json

    ref_layers = [k for k in ref_data if k != "metadata"]
    comp_layers = [k for k in comp_data if k != "metadata"]

    if set(ref_layers) != set(comp_layers):
        print("\nERROR: Layer mismatch!")
        print(f"Reference layers: {ref_layers}")
        print(f"Compare layers: {comp_layers}")
        return

    results = {"aggregated": {"rmse": [], "cosine_sim_mean": []}, "per_layer": {}}

    # Per-layer comparison
    for layer_name in sorted(ref_layers):
        ref_tensor = ref_data[layer_name]
        comp_tensor = comp_data[layer_name]

        if ref_tensor.shape != comp_tensor.shape:
            print(f"ERROR: {layer_name} shape mismatch! Skipping.")
            continue

        rmse = compute_rmse(ref_tensor, comp_tensor)
        cos_sim = compute_cosine_similarity(ref_tensor, comp_tensor)

        results["per_layer"][layer_name] = {"rmse": rmse, "cosine_sim": cos_sim}
        results["aggregated"]["rmse"].append(rmse)
        results["aggregated"]["cosine_sim_mean"].append(cos_sim["mean"])

    # Aggregated statistics
    if results["aggregated"]["rmse"]:
        rmse_array = torch.tensor(results["aggregated"]["rmse"])
        cos_sim_array = torch.tensor(results["aggregated"]["cosine_sim_mean"])

        results["aggregated"]["rmse_stats"] = {
            "mean": rmse_array.mean().item(),
            "std": rmse_array.std().item(),
            "min": rmse_array.min().item(),
            "max": rmse_array.max().item(),
        }
        results["aggregated"]["cosine_sim_stats"] = {
            "mean": cos_sim_array.mean().item(),
            "std": cos_sim_array.std().item(),
            "min": cos_sim_array.min().item(),
            "max": cos_sim_array.max().item(),
        }
        results["aggregated"]["num_steps"] = ref_data.get("metadata", {}).get("num_steps", None)
        results["aggregated"]["num_layers"] = len(rmse_array)

    # Save to JSON if requested
    if output_json:
        # Remove raw lists for JSON serialization
        results["aggregated"].pop("rmse", None)
        results["aggregated"].pop("cosine_sim_mean", None)

        with open(output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved comparison results to {output_json}")


if __name__ == "__main__":
    main()
