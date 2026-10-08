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
"""Export Megatron Core Model to HuggingFace vLLM fakequant checkpoint."""

import os
import tempfile
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
import yaml

from modelopt.torch.export.quant_format import QUANTIZATION_NONE
from modelopt.torch.export.unified_export_megatron import GPTModelExporter
from modelopt.torch.quantization.nn import TensorQuantizer
from modelopt.torch.utils import get_unwrapped_name
from modelopt.torch.utils.distributed import DistributedProcessGroup, is_master

__all__ = ["export_mcore_gpt_to_hf_vllm_fq"]


def _quantizer_configs(module: torch.nn.Module) -> tuple[dict[str, dict], str]:
    """Return quantizer recipes and an unsupported-settings error, if any."""
    configs = {}
    for name, quantizer in module.named_modules():
        if not isinstance(quantizer, TensorQuantizer):
            continue
        is_weight_quantizer = "weight_quantizer" in name
        is_quantizing = quantizer.is_enabled and quantizer._if_quant
        is_activation_quantizing = is_quantizing and not is_weight_quantizer
        is_integer_quantizing = is_activation_quantizing and isinstance(quantizer.num_bits, int)
        # Weight transforms are folded; activation transforms also run when quantization is off.
        unsupported = [
            setting
            for setting, invalid in {
                "rotate": not is_weight_quantizer and quantizer.rotate_is_enabled,
                "pre_quant_scale": not is_weight_quantizer
                and quantizer.pre_quant_scale is not None,
                "fake_quant": is_quantizing and not quantizer.fake_quant,
                "unsigned": is_integer_quantizing and quantizer.unsigned,
                "narrow_range": is_integer_quantizing and quantizer.narrow_range,
                "dynamic_amax": is_activation_quantizing
                and not (
                    quantizer.is_mx_format
                    or quantizer._use_constant_amax
                    or getattr(quantizer, "_amax", None) is not None
                ),
                "bias": is_activation_quantizing and quantizer.bias is not None,
            }.items()
            if invalid
        ]
        if unsupported:
            return {}, (
                f"Unsupported vLLM fakequant quantizer settings for {name or '<root>'}: "
                f"{', '.join(unsupported)}"
            )
        recipe = {"_disabled": is_weight_quantizer or not is_quantizing}
        if not recipe["_disabled"]:
            recipe.update(
                {
                    "_num_bits": quantizer.num_bits,
                    "_axis": quantizer.axis,
                    "_block_sizes": quantizer.block_sizes,
                }
            )
            if quantizer.backend is not None:
                recipe.update(
                    {
                        "_backend": quantizer.backend,
                        "_backend_extra_args": quantizer.backend_extra_args,
                    }
                )
        configs[get_unwrapped_name(name, module)] = recipe
    return configs, ""


def _save_quantizer_state(path: Path, save: Callable[[Path], None]) -> None:
    """Save quantizer state or recipes and share write completion or failure across ranks."""
    failure = ""
    if is_master():
        try:
            save(path)
        except Exception as exc:
            failure = f"{type(exc).__name__}: {exc}"
    failure = DistributedProcessGroup.get_dist_syncd_obj(
        failure,
        DistributedProcessGroup(None),
        lambda failures: next((message for message in failures if message), ""),
    )
    if failure:
        raise RuntimeError(f"Failed to save {path.name}: {failure}")


def _merge_quantizer_states(objs: list[dict | None]) -> dict:
    """Merge replicated recipes or tensors, requiring duplicate keys to agree."""
    merged = {}
    first_rank_by_name = {}
    for rank, state in enumerate(objs):
        if state is None:
            continue
        for name, value in state.items():
            if name in merged:
                previous = merged[name]
                if isinstance(value, torch.Tensor):
                    matches = previous.dtype == value.dtype and torch.equal(previous, value)
                    kind = "tensors"
                else:
                    matches = previous == value
                    kind = "recipes"
                if not matches:
                    raise ValueError(
                        f"Conflicting quantizer {kind} for {name} between ranks "
                        f"{first_rank_by_name[name]} and {rank}"
                    )
            else:
                merged[name] = value
                first_rank_by_name[name] = rank
    return merged


def gather_mcore_vllm_fq_quantizer_recipe(
    quantizer_state_by_name: dict[str, dict],
    save_directory: str | os.PathLike,
) -> None:
    """Sync each rank's captured quantizer configs and save as ``quant_recipe.yaml``.

    Args:
        quantizer_state_by_name: HF-prefixed quantizer name -> resolved config, collected in
            ``VllmFqGPTModelExporter._get_quantized_state``.
        save_directory: Directory for ``quant_recipe.yaml``.
    """
    merged = DistributedProcessGroup.get_dist_syncd_obj(
        quantizer_state_by_name,
        DistributedProcessGroup(None),
        _merge_quantizer_states,
    )

    def save_recipe(path: Path) -> None:
        with open(path, "w") as f:
            yaml.safe_dump(merged, f, sort_keys=False)

    _save_quantizer_state(Path(save_directory) / "quant_recipe.yaml", save_recipe)


def gather_mcore_vllm_fq_quantized_state_dict(
    _model,
    layer_state_dicts: Mapping[Any, dict[str, torch.Tensor]],
    save_directory: str | os.PathLike,
) -> None:
    """Gather quantizer tensors from every per-layer export shard, sync across ranks, and save.

    Megatron export stores one ``OrderedDict`` per decoder layer in ``layer_state_dicts``; the
    ``GPTModelExporter.state_dict`` property only references the last shard after build, so
    quantizer state must be collected from all shards.

    Args:
        _model: Unused; kept for a stable call signature with export entry points.
        layer_state_dicts: Mapping from layer index to that shard's flat export state dict.
        save_directory: Directory for ``quantizer_state.pth``.
    """
    quantizer_state_dict: dict[str, torch.Tensor] = {}
    for sd in layer_state_dicts.values():
        for k, v in sd.items():
            if "quantizer" in k:
                quantizer_state_dict[k] = v.detach().clone().cpu()

    merged_quantizer_state_dict = DistributedProcessGroup.get_dist_syncd_obj(
        quantizer_state_dict,
        DistributedProcessGroup(None),
        _merge_quantizer_states,
    )
    _save_quantizer_state(
        Path(save_directory) / "quantizer_state.pth",
        lambda path: torch.save(merged_quantizer_state_dict, path),
    )


class VllmFqGPTModelExporter(GPTModelExporter):
    """VLLM fakequant GPTModel exporter."""

    _QUANT_RECIPE_MARKER_SUFFIX = "._vllm_fakequant_recipe_marker"

    def __init__(self, *args, **kwargs):
        """Initialize recipe capture before lazy export shards are built."""
        super().__init__(*args, **kwargs)
        self._quantizer_state_for_recipe: dict[str, dict] = {}
        self._quantizer_recipe_markers: list[dict] = []
        self._quantizer_validation_failure = ""
        self._packing_experts = False

    def _store_quantizer_recipe(self, name: str, recipe: dict) -> None:
        """Store one resolved recipe, requiring repeated routes to agree."""
        previous = self._quantizer_state_for_recipe.get(name)
        if previous is not None and previous != recipe:
            raise ValueError(f"Conflicting quantizer recipes routed to {name}")
        self._quantizer_state_for_recipe[name] = recipe

    def _gather_exclude_modules(self) -> list[str]:
        """Validate settings before gathering excluded modules."""
        # All ranks reach this after MTP collection and before writing weights.
        failure = DistributedProcessGroup.get_dist_syncd_obj(
            self._quantizer_validation_failure,
            DistributedProcessGroup(None),
            lambda failures: next((message for message in failures if message), ""),
        )
        if failure:
            raise ValueError(failure)
        return super()._gather_exclude_modules()

    def _extract_quantizer_recipe_markers(
        self, layer_state_dicts: Mapping[Any, dict[str, torch.Tensor]]
    ) -> None:
        """Resolve temporary recipe markers after the normal export mapping has routed them."""
        for state_dict in layer_state_dicts.values():
            for key in list(state_dict):
                if not key.endswith(self._QUANT_RECIPE_MARKER_SUFFIX):
                    continue

                marker = state_dict.pop(key)
                marker_ids = [int(i) for i in marker.detach().cpu().reshape(-1).tolist()]
                recipes = [self._quantizer_recipe_markers[i] for i in marker_ids]
                if any(recipe != recipes[0] for recipe in recipes[1:]):
                    raise ValueError(f"Conflicting packed quantizer recipes routed to {key}")

                recipe_name = key[: -len(self._QUANT_RECIPE_MARKER_SUFFIX)]
                self._store_quantizer_recipe(recipe_name, recipes[0])

    def _get_quantizer_state(self, state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Move routed quantizer tensors and recipe markers out of a weight shard."""
        return {
            key: state_dict.pop(key)
            for key in list(state_dict)
            if "quantizer" in key or key.endswith(self._QUANT_RECIPE_MARKER_SUFFIX)
        }

    def save_pretrained(
        self,
        save_directory: str | os.PathLike,
        pretrained_model_name_or_path: str | os.PathLike,
    ):
        """Save folded weights, quantizer state, and recipes; MTP weights remain unquantized.

        Args:
            save_directory: The directory to save the exported model.
            pretrained_model_name_or_path: The name or path of the pretrained model.
        """
        save_dir = os.fspath(save_directory)
        os.makedirs(save_dir, exist_ok=True)

        assert not (self.is_multimodal and pretrained_model_name_or_path is not None), (
            "Exporting weights in bf16 and amax values is not supported for multimodal models "
            "when pretrained_model_name_or_path is not None"
        )
        assert not self.export_extra_modules, (
            "Exporting extra modules is not supported for vLLM fakequant"
        )

        # Temporary scalar markers carry each recipe through the same export mapping as amax.
        self._quantizer_state_for_recipe = {}
        # Cached shards still reference markers collected when they were built.
        if not self._layer_state_dicts:
            self._quantizer_recipe_markers = []
        quantizer_state_dicts = {
            index: self._get_quantizer_state(state_dict)
            for index, state_dict in enumerate([*self.layer_state_dicts.values(), self._state_dict])
        }
        super().save_pretrained(save_directory, pretrained_model_name_or_path)

        # Save fresh quantizer files after the base exporter copies source files.
        self._extract_quantizer_recipe_markers(quantizer_state_dicts)
        gather_mcore_vllm_fq_quantized_state_dict(self.model, quantizer_state_dicts, save_directory)
        gather_mcore_vllm_fq_quantizer_recipe(self._quantizer_state_for_recipe, save_directory)

    def _get_quantization_format(self, module: torch.nn.Module):
        return QUANTIZATION_NONE

    @contextmanager
    def _collect_packed_quantizers(self):
        """Capture quantizer entries that packed weight mappings do not forward."""
        previous = self._packing_experts
        self._packing_experts = True
        try:
            yield
        finally:
            self._packing_experts = previous

    def _pack_name_remapping(self, *args, **kwargs):
        with self._collect_packed_quantizers():
            return super()._pack_name_remapping(*args, **kwargs)

    def _pack_name_remapping_gpt_oss(self, *args, **kwargs):
        with self._collect_packed_quantizers():
            return super()._pack_name_remapping_gpt_oss(*args, **kwargs)

    def _self_attention_scaling(
        self, module, prefix, k_scale_name="k_scale", v_scale_name="v_scale", is_mtp=False
    ):
        # Fakequant stores K/V ranges and recipes instead of real KV-cache scaling factors.
        if is_mtp:
            prefix = self._mtp_prefix(prefix)
        state, _, _ = self._get_quantized_state(module, prefix=prefix)
        self._state_dict.update({prefix + name: value for name, value in state.items()})

    def _get_quantized_state(
        self,
        module: torch.nn.Module,
        dtype: torch.dtype = torch.float16,
        prefix: str = "",
    ) -> tuple[dict[str, torch.Tensor], str, int]:
        """Return a state_dict, quantization format, and block_size of the module.

        The weight_quantizer is folded into the weight via fake-quantization
        (quantize + dequantize), and its amax is not exported. The vLLM fakequant
        reload path is expected to disable the weight quantizer when the amax is absent.

        Args:
            module: The target module to perform real quantization.
            dtype: The default data type.

        Returns:
            Tuple: state_dict, quantization format, and block_size of the module.
        """
        name_to_value = {}
        qformat: str = self._get_quantization_format(module)
        if prefix.startswith("mtp."):
            for name, quantizer in module.named_modules():
                if isinstance(quantizer, TensorQuantizer) and (
                    (quantizer.is_enabled and quantizer._if_quant)
                    or quantizer.rotate_is_enabled
                    or quantizer.pre_quant_scale is not None
                ):
                    self._quantizer_validation_failure = self._quantizer_validation_failure or (
                        f"MTP quantization is not supported by vLLM fakequant export/reload: {name}"
                    )
                    break
            return self._get_weight_bias(module, dtype), qformat, 0
        quantizer_configs, error = _quantizer_configs(module)
        if error:
            self._quantizer_validation_failure = self._quantizer_validation_failure or error
            # Skip unsupported kernels while peers complete the export collectives.
            return self._get_weight_bias(module, dtype), qformat, 0
        source_prefix = prefix if not prefix or prefix.endswith(".") else prefix + "."
        for qname, qstate in quantizer_configs.items():
            marker_id = len(self._quantizer_recipe_markers)
            self._quantizer_recipe_markers.append(qstate)
            name_to_value[qname + self._QUANT_RECIPE_MARKER_SUFFIX] = torch.tensor(
                marker_id, dtype=torch.int64
            )

        if qformat is None and "norm" not in prefix:
            # Add exclude layers for vllm fakequant config. Note that if the prefix is not an empty
            # string then it usually ends with "." which needs to be removed.
            self.exclude_modules.append(prefix.removesuffix("."))
        block_size = 0
        name_to_value = self._get_weight_bias(module, dtype, name_to_value)
        if "weight" in name_to_value:
            # Use the original device (avoid the CPU round-trip introduced by _get_weight_bias;
            # fake-quantization runs on CUDA and the result is moved to CPU below).
            weight = module.weight.to(dtype)
            # Fold the weight_quantizer into the weight by applying fake-quantization
            # (quantize then dequantize). The weight_quantizer amax is not exported;
            # the vLLM fakequant reload path disables the weight quantizer when absent.
            weight_quantizer = getattr(module, "weight_quantizer", None)
            # Disabled weight quantizers can still apply scaling or rotation.
            if weight_quantizer is not None:
                with torch.no_grad():
                    # NVFP4-like kernels may need CUDA; if weights are CPU after gather, run on
                    # CUDA then ``weight_quantizer.to`` back (full module round-trip).
                    quant_device = (
                        torch.device("cuda", torch.cuda.current_device())
                        if weight.device.type == "cpu" and torch.cuda.is_available()
                        else weight.device
                    )
                    # TensorQuantizer does not expose nn.Module.device (custom __getattr__).
                    param_device = next(weight_quantizer.parameters(), None)
                    buf_device = next(weight_quantizer.buffers(), None)
                    wq_dev = (
                        param_device.device
                        if param_device is not None
                        else (buf_device.device if buf_device is not None else torch.device("cpu"))
                    )
                    need_move = wq_dev != quant_device
                    if need_move:
                        weight_quantizer.to(quant_device)
                    try:
                        weight = weight_quantizer(weight.to(quant_device)).to(dtype)
                    finally:
                        if need_move:
                            weight_quantizer.to(wq_dev)
            name_to_value["weight"] = weight.cpu()

        # Save activation ranges; weight quantizers are folded into the weights above.
        for name, quantizer in module.named_modules():
            if not isinstance(quantizer, TensorQuantizer) or "weight_quantizer" in name:
                continue
            amax = getattr(quantizer, "_amax", None)
            # The constant amax takes precedence over any stored calibration buffer.
            if quantizer._use_constant_amax and not quantizer.is_mx_format:
                amax = quantizer._get_amax(amax if amax is not None else torch.empty(0))
            if amax is not None:
                name = get_unwrapped_name(name, module)
                name_to_value[name + "._amax"] = amax.detach().cpu().clone()
        if self._packing_experts:
            for name, value in self._get_quantizer_state(name_to_value).items():
                key = source_prefix + name
                previous = self._state_dict.get(key)
                if previous is not None:
                    if name.endswith(self._QUANT_RECIPE_MARKER_SUFFIX):
                        previous_recipe = self._quantizer_recipe_markers[int(previous.flatten()[0])]
                        recipe = self._quantizer_recipe_markers[int(value)]
                        if previous_recipe != recipe:
                            self._quantizer_validation_failure = (
                                self._quantizer_validation_failure
                                or f"Conflicting packed quantizer recipes routed to {key}"
                            )
                        value = torch.cat((previous.reshape(-1), value.reshape(-1)))
                    else:
                        # vLLM shares one activation range across fused experts.
                        value = torch.maximum(previous, value)
                self._state_dict[key] = value
        return name_to_value, qformat, block_size


def export_mcore_gpt_to_hf_vllm_fq(
    model: torch.nn.Module,
    pretrained_model_name_or_path: str | os.PathLike,
    export_extra_modules: bool = False,
    dtype: torch.dtype = torch.bfloat16,
    export_dir: Path | str = tempfile.gettempdir(),
    moe_router_dtype: torch.dtype | None = None,
    trust_remote_code: bool = False,
):
    """Export Megatron Core GPTModel to unified checkpoint and save to export_dir.

    Also saves ``quantizer_state.pth`` and ``quant_recipe.yaml`` files,
    for later fakequant reload.

    Args:
        model: The Megatron Core GPTModel instance.
        pretrained_model_name_or_path: Can be either: the *model id* of a
            pretrained model hosted inside a model repo on huggingface.co; or
            a *directory* containing model weights saved using
            [`~PreTrainedModel.save_pretrained`], e.g., `./my_model_directory/`.
        export_extra_modules: If True, export extra modules like medusa_heads or
            eagle_module. Otherwise, only export the base model.
        dtype: The weights data type to export the unquantized layers.
        export_dir: The target export path.

    Raises:
        ValueError: If a quantizer uses settings that cannot be restored.
    """
    exporter = VllmFqGPTModelExporter(
        model,
        pretrained_model_name_or_path,
        export_extra_modules=export_extra_modules,
        dtype=dtype,
        moe_router_dtype=moe_router_dtype,
        trust_remote_code=trust_remote_code,
    )
    exporter.save_pretrained(export_dir, pretrained_model_name_or_path)
