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

"""Module-shape predicates and MoE quantizer helpers shared by every export backend.

The TensorRT-LLM ``build_*_config`` builders that turn these modules into checkpoint
configs live in :mod:`modelopt.torch.export.trtllm.layer_utils`.
"""

from warnings import warn

import torch
import torch.nn as nn


def get_experts_list(
    module: torch.nn.Module,
    model_type: str,
):
    """Returns list of grouped experts by linear name for given module.

    Args:
        module: MoE block (e.g. MixtralSparseMoeBlock, NemotronHMOE).
        model_type: `type(root_model).__name__.lower()` (may change after ModelOpt quantize).
    """
    experts_list = []

    # Define linear layer names for different model types
    if "mixtralforcausallm" in model_type:
        linear_names = ["w1", "w2", "w3"]
    elif any(
        qwen_variant in model_type
        for qwen_variant in [
            "qwenmoeforcausallm",
            "qwen2moeforcausallm",
            "qwen3moeforcausallm",
            "qwen3nextforcausallm",
        ]
    ):
        linear_names = ["gate_proj", "down_proj", "up_proj"]
    elif "nemotronhforcausallm" in model_type:
        linear_names = ["up_proj", "down_proj"]
    elif "gemma4" in model_type:
        linear_names = ["gate_proj", "down_proj", "up_proj"]
    else:
        raise NotImplementedError(f" {model_type} not supported")

    # Common logic for all supported model types
    experts_list.extend(
        [
            [_get_expert_attr(module.experts, i, linear_name) for i in range(len(module.experts))]
            for linear_name in linear_names
        ]
    )

    return experts_list


def is_layernorm(module: nn.Module) -> bool:
    """Returns whether the module is a layernorm layer."""
    module_name = type(module).__name__
    return any(norm in module_name for norm in ["LayerNorm", "RMSNorm"])


def is_moe(module: nn.Module) -> bool:
    """Returns whether the module is an MOE layer."""
    name = type(module).__name__.lower()
    # Auto-detect common MoE patterns
    if name.endswith("sparsemoeblock") or "moelayer" in name:
        return True
    # Explicit matches for non-standard naming
    if any(key in name for key in ["arcticmoe", "deepseekmoe", "dbrxffn", "nemotronhmoe"]):
        return True
    # Structural detection: modules with router + experts (e.g. Gemma4TextDecoderLayer)
    return (
        hasattr(module, "router")
        and hasattr(module, "experts")
        and isinstance(module.experts, nn.Module)
    )


def is_quantlinear(module: nn.Module) -> bool:
    """Returns whether the module is a quantized linear layer."""
    name = type(module).__name__
    return (
        any(
            keyword in name
            for keyword in ["QuantLinear", "QuantCompressedLinear", "QuantFP8Linear"]
        )
        and "lora" not in name.lower()
        and "ds_kernel" not in name.lower()
    )


def _get_expert_attr(experts: nn.Module, export_id: int, linear_name: str):
    # Generic expert attribute accessor.
    # Works for most MoE models that store experts as a list/ModuleList where
    # each expert has linear layers as direct attributes:
    # experts[0].w1, experts[0].w2, experts[0].w3  (Mixtral)
    # experts[0].gate_proj, experts[0].down_proj, experts[0].up_proj  (Qwen)
    # experts[0].linear_fc1, experts[0].linear_fc2  (Llama MCore)
    return getattr(experts[export_id], linear_name)


def get_expert_linear_names(module: nn.Module) -> list[str]:
    """Get the list of linear names for the experts."""

    def module_match_name_list(module, name_list):
        """Check if the module name matches any of the names in the list.

        e.g. module_match_name_list(QuantQwen3MoeSparseMoeBlock, ['Qwen3MoeSparseMoeBlock']) -> True

        """
        return any(name.lower() in type(module).__name__.lower() for name in name_list)

    # Structural detection: after _export_fused_experts, fused expert modules
    # have per-expert submodules with gate_proj/up_proj/down_proj.
    # Also handles models that originally used this naming (Qwen, DeepSeek, etc.).
    if hasattr(module, "experts"):
        first_proj_attr = getattr(module.experts, "_first_proj_attr", "gate_up_proj")
        if hasattr(module.experts, f"{first_proj_attr}_weight_quantizers"):
            return [first_proj_attr, "down_proj"]

    if module_match_name_list(
        module,
        [
            "Qwen2MoeSparseMoeBlock",
            "Qwen3MoeSparseMoeBlock",
            "Qwen3NextSparseMoeBlock",
            "Qwen3_5MoeSparseMoeBlock",
            "DeepseekMoE",
        ],
    ):
        return ["gate_proj", "down_proj", "up_proj"]
    elif module_match_name_list(module, ["MixtralSparseMoeBlock"]):
        # Old-style Mixtral (iterable experts) uses w1/w2/w3.
        # Fused Mixtral (transformers 5.0+) is already handled by the
        # structural first-projection quantizer check above.
        return ["w1", "w2", "w3"]
    elif module_match_name_list(module, ["MixtralMoeSparseMoeBlock"]):
        # Older transformers naming for Mixtral
        return ["linear_fc1", "linear_fc2"]
    elif module_match_name_list(module, ["DBRXMoeSparseMoeBlock"]):
        return ["w1_linear", "w2_linear", "v1_linear"]
    elif module_match_name_list(module, ["GptOssMoE"]):
        return ["gate_up_proj", "down_proj"]
    elif module_match_name_list(module, ["NemotronHMOE"]):
        # NemotronHMOE experts (NemotronHMLP) use up_proj and down_proj only (no gate).
        return ["up_proj", "down_proj"]
    elif module_match_name_list(module, ["Gemma4TextDecoderLayer"]):
        # Gemma4 MoE experts are unfused into per-expert nn.Linear layers
        return ["gate_proj", "down_proj", "up_proj"]
    else:
        # assuming w1, w2, w3 by default
        return ["w1", "w2", "w3"]


def set_expert_quantizer_amax(
    modules: nn.Module | list[nn.Module],
    quantizer_attrs: str | list[str] | None = None,
    fallback_value: float = 0.5,
    device: torch.device | None = None,
) -> list[nn.Module]:
    """Set amax values for expert quantizers using smart fallback logic.

    Uses smart fallback logic:

    1. Use max from existing quantizers in current batch (best - direct from calibration)
    2. If no existing values found, then:
       - For weight quantizers: calculate from weight statistics
       - For input quantizers: use max from other experts, fallback if none found
    3. Use fallback value as last resort

    This ensures we always have semantically appropriate amax values for export.

    Args:
        modules: Single module or list of modules containing quantizers
        quantizer_attrs: Specific quantizer attributes to handle.
            If None, defaults to ["input_quantizer"] for backward compatibility.
        fallback_value: Final fallback value when other methods fail (default: 0.5)
        device: Target device for tensors (auto-detected if None)

    Returns:
        uncalibrated_modules: a list of uncalibrated experts
    """
    import warnings

    # Normalize inputs
    if not isinstance(modules, list):
        modules = [modules]

    if quantizer_attrs is None:
        quantizer_attrs = ["input_quantizer"]
    elif isinstance(quantizer_attrs, str):
        quantizer_attrs = [quantizer_attrs]

    uncalibrated_modules = []

    # Determine target device if not provided
    if device is None:
        first_module = next(iter(modules))
        if hasattr(first_module, "weight"):
            target_device = first_module.weight.device
        else:
            target_device = torch.device("cpu")
    else:
        target_device = device

    # Collect all valid quantizers
    all_quantizers = []

    for module in modules:
        for attr_name in quantizer_attrs:
            if hasattr(module, attr_name):
                quantizer = getattr(module, attr_name)
                if (
                    quantizer is not None
                    and hasattr(quantizer, "is_enabled")
                    and quantizer.is_enabled
                ):
                    all_quantizers.append((module, attr_name, quantizer))

    target_amax = None

    # Collect ANY existing amax values from current batch (most direct source).
    # Reduce per-quantizer amax to a scalar before stacking — quantizers in
    # static-mode (e.g. NVFP4 with pre-computed per-block _amax) carry tensors
    # whose shapes differ across attrs (gate_up_proj vs down_proj have different
    # output dims), and torch.stack would otherwise fail. The result here is
    # only used as a *fallback* scalar `target_amax` for quantizers missing
    # amax, so a max-of-max is exactly what we want.
    valid_amax_values = []
    for _, attr_name, quantizer in all_quantizers:
        existing_amax = getattr(quantizer, "amax", None)
        if existing_amax is not None:
            # Convert to tensor and add to collection
            if isinstance(existing_amax, torch.Tensor):
                # Meta tensors have no storage; .amax() / .to() would fail.
                if existing_amax.is_meta:
                    continue
                valid_amax_values.append(existing_amax.amax().to(target_device))
            else:
                valid_amax_values.append(
                    torch.tensor(existing_amax, dtype=torch.float32, device=target_device)
                )

    # Use existing values from current batch if any found
    if len(valid_amax_values) > 0:
        target_amax = torch.max(torch.stack(valid_amax_values))

    # If no existing values in current batch, apply type-specific fallback logic
    elif target_amax is None:
        has_input_quantizers = any("input_quantizer" in attr for _, attr, _ in all_quantizers)
        has_weight_quantizers = any("weight_quantizer" in attr for _, attr, _ in all_quantizers)

        if has_weight_quantizers and not has_input_quantizers:
            # For weight quantizers: calculate from weight statistics
            weight_amax_values = []
            for module, _, _ in all_quantizers:
                # Try to find a weight tensor in the module
                weight_tensor = None
                for weight_attr in ["weight", "gate_up_proj", "down_proj"]:
                    if hasattr(module, weight_attr):
                        weight_tensor = getattr(module, weight_attr)
                        break

                if weight_tensor is not None:
                    weight_amax_values.append(torch.max(torch.abs(weight_tensor)))

            if weight_amax_values:
                target_amax = torch.max(torch.stack(weight_amax_values)).item()
        elif has_input_quantizers:
            # For input quantizers: ideally search other experts for existing input amax values
            # TODO: Implement broader expert search - currently function only has access to current batch
            # For now, this will fall through to fallback value
            pass

    # Final fallback
    if target_amax is None:
        target_amax = fallback_value
        has_input_quantizers = any("input_quantizer" in attr for _, attr, _ in all_quantizers)

    # Apply target amax to quantizers that need it
    for module, attr_name, quantizer in all_quantizers:
        # Check if quantizer needs amax (use property for consistency)
        # Also treat zero amax as needing recalibration — a zero amax is never valid
        # and indicates the quantizer wasn't activated during calibration
        amax = getattr(quantizer, "amax", None)
        needs_amax = amax is None or (isinstance(amax, torch.Tensor) and torch.all(amax == 0))

        # Skip dynamic quantizers for input quantizers
        if "input_quantizer" in attr_name and getattr(quantizer, "_dynamic", False):
            needs_amax = False

        if needs_amax:
            # Create tensor with appropriate value (using function-wide target_device)
            if isinstance(target_amax, torch.Tensor):
                amax_tensor = target_amax.clone().to(dtype=torch.float32, device=target_device)
            else:
                amax_tensor = torch.tensor(target_amax, dtype=torch.float32, device=target_device)

            # Set amax value using property for proper validation and tensor handling
            quantizer.amax = amax_tensor

            uncalibrated_modules.append(module)
            amax_val = amax_tensor.item() if isinstance(amax_tensor, torch.Tensor) else amax_tensor

            if len(valid_amax_values) > 0:
                warnings.warn(
                    f"Missing amax value for {attr_name} in {type(module).__name__}. "
                    f"Setting it to {amax_val:.6f} (max from existing quantizers in current batch). "
                    f"This typically occurs when certain experts are not activated during calibration."
                )
            elif amax_val != fallback_value and "input_quantizer" not in attr_name:
                warnings.warn(
                    f"Missing amax value for {attr_name} in {type(module).__name__}. "
                    f"Setting it to {amax_val:.6f} (computed from weights)."
                )

    return uncalibrated_modules


# Gate/up naming pairs for standard (unfused) MoE architectures.
# Fused variants (gate_up_proj, linear_fc1) already share a single quantizer and need no sync.
_GATE_UP_PAIRS = [("gate_proj", "up_proj"), ("w1", "w3")]


def sync_moe_gate_up_amax(model: nn.Module) -> int:
    """Take element-wise max of gate and up weight quantizer amaxes per expert.

    Serving engines fuse gate_proj and up_proj into a single gate_up_proj and
    require a single weight_scale_2. Since weight_scale_2 = amax / (6 * m_fp8)
    (m_fp8=448 normally, 256 for NVFP4 4/6 mode),
    syncing amaxes before quantization ensures the per-block weight_scale values
    are computed against a consistent global scale.

    Only affects standard MoE models with separate gate/up linear layers
    (e.g. Qwen MoE, DeepSeek). Models with already-fused gate_up_proj
    (e.g. Llama4, GptOss) are unaffected.

    Returns:
        Number of expert gate/up pairs whose amaxes were synced.
    """
    synced = 0
    for _, sub_module in model.named_modules():
        if not (is_moe(sub_module) and hasattr(sub_module, "experts")):
            continue
        if not hasattr(sub_module.experts, "__iter__"):
            continue
        for expert in sub_module.experts:
            for gate_name, up_name in _GATE_UP_PAIRS:
                gate_linear = getattr(expert, gate_name, None)
                up_linear = getattr(expert, up_name, None)
                if gate_linear is None or up_linear is None:
                    continue
                gate_wq = getattr(gate_linear, "weight_quantizer", None)
                up_wq = getattr(up_linear, "weight_quantizer", None)
                if gate_wq is None or up_wq is None:
                    break
                gate_amax = getattr(gate_wq, "amax", None)
                up_amax = getattr(up_wq, "amax", None)
                if gate_amax is None or up_amax is None:
                    break
                # Meta tensors have no storage (e.g. CPU-offloaded experts that
                # were never activated during calibration). Skip — there is no
                # real amax data to sync.
                if gate_amax.is_meta or up_amax.is_meta:
                    warn(
                        f"Skipping gate/up amax sync for expert with meta tensors "
                        f"(gate_amax.is_meta={gate_amax.is_meta}, "
                        f"up_amax.is_meta={up_amax.is_meta}). "
                        f"This typically means the expert was CPU-offloaded and "
                        f"not activated during calibration."
                    )
                    break
                if not torch.equal(gate_amax, up_amax):
                    shared_amax = torch.max(gate_amax, up_amax)
                    gate_wq.amax = shared_amax
                    up_wq.amax = shared_amax.clone()
                    synced += 1
                break
    return synced
