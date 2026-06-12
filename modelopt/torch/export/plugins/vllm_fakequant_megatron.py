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
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from modelopt.torch.export.model_config import QUANTIZATION_NONE
from modelopt.torch.export.unified_export_megatron import GPTModelExporter
from modelopt.torch.quantization.utils import get_quantizer_state_dict
from modelopt.torch.utils.distributed import DistributedProcessGroup, is_master

__all__ = ["export_mcore_gpt_to_hf_vllm_fq"]


def gather_mcore_vllm_fq_quantized_state_dict(
    _model,
    layer_state_dicts: Mapping[Any, dict[str, torch.Tensor]],
    save_directory: str | os.PathLike,
) -> None:
    """Gather quantizer tensors from every per-layer export shard, sync across ranks, and save.

    Megatron export stores one ``OrderedDict`` per decoder layer in ``layer_state_dicts``; the
    ``GPTModelExporter.state_dict`` property only references the last shard after build, so
    quantizer sidecars must be collected from all shards.

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

    def _merge_quantizer_states(objs: list) -> dict:
        merged: dict = {}
        for d in objs:
            if d is not None:
                merged.update(d)
        return merged

    merged_quantizer_state_dict = DistributedProcessGroup.get_dist_syncd_obj(
        quantizer_state_dict,
        DistributedProcessGroup(None),
        _merge_quantizer_states,
    )
    if is_master():
        torch.save(merged_quantizer_state_dict, Path(save_directory) / "quantizer_state.pth")


class VllmFqGPTModelExporter(GPTModelExporter):
    """VLLM fakequant GPTModel exporter."""

    @staticmethod
    def _pop_quantizer_keys(state_dict: dict) -> None:
        """Remove quantizer tensors from an export shard (OrderedDict-safe)."""
        for k in [k for k in state_dict if "quantizer" in k]:
            state_dict.pop(k, None)

    def save_pretrained(
        self,
        save_directory: str | os.PathLike,
        pretrained_model_name_or_path: str | os.PathLike,
    ):
        """Save HF shards + sidecar ``quantizer_state.pth``; then delegate to base export.

        Pipeline-parallel placement of ``config.json``, tokenizer, and multimodal tensors
        remains handled by ``GPTModelExporter.save_pretrained`` (via ``super()``).
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

        gather_mcore_vllm_fq_quantized_state_dict(self.model, self.layer_state_dicts, save_dir)

        self._pop_quantizer_keys(self.state_dict)
        for _layer_sd in self.layer_state_dicts.values():
            self._pop_quantizer_keys(_layer_sd)

        super().save_pretrained(save_directory, pretrained_model_name_or_path)

    def _get_quantization_format(self, module: torch.nn.Module):
        return QUANTIZATION_NONE

    def _get_quantized_state(
        self,
        module: torch.nn.Module,
        dtype: torch.dtype = torch.float16,
        prefix: str = "",
    ) -> tuple[dict[str, torch.Tensor], str, int]:
        """Return a state_dict, quantization format, and block_size of the module.

        Args:
            module: The target module to perform real quantization.
            dtype: The default data type.

        Returns:
            Tuple: state_dict, quantization format, and block_size of the module.
        """
        name_to_value = {}
        qformat: str = self._get_quantization_format(module)
        if qformat is None and "norm" not in prefix:
            # Add exclude layers for vllm fakequant config. Note that if the prefix is not an empty
            # string then it usually ends with "." which needs to be removed.
            self.exclude_modules.append(prefix.removesuffix("."))
        block_size = 0

        if hasattr(module, "weight") and module.weight is not None:
            weight = module.weight.to(dtype).cpu()
            name_to_value["weight"] = weight
        else:
            return name_to_value, qformat, block_size

        if hasattr(module, "bias") and module.bias is not None:
            name_to_value["bias"] = module.bias.to(dtype).cpu()
        for name, param in get_quantizer_state_dict(module).items():
            for key, value in param.items():
                name_to_value[name + "." + key] = value.to(dtype).cpu()
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
