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
from pathlib import Path

import torch

from modelopt.torch.export.model_config import QUANTIZATION_NONE
from modelopt.torch.export.unified_export_megatron import GPTModelExporter
from modelopt.torch.quantization.utils import get_quantizer_state_dict

__all__ = ["export_mcore_gpt_to_hf_vllm_fq"]


def gather_mcore_vllm_fq_quantized_state_dict(
    model, state_dict: dict[str, torch.Tensor], save_directory: str | os.PathLike
):
    """Gather all quantized state dict from all ranks and save them to a file.

    Args:
        state_dict: The state dictionary of the module.
        save_directory: The directory to save the quantized state dict.

    Returns:
        The state dictionary of the module without quantized state.
    """
    quantizer_state_dict = {
        k: v.detach().clone().cpu() for k, v in state_dict.items() if "quantizer" in k
    }

    # Gather all amax dicts to rank 0
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()

    if rank == 0:
        # Rank 0 will collect all amax values
        all_quantizer_state_dicts = [None] * world_size
        torch.distributed.gather_object(quantizer_state_dict, all_quantizer_state_dicts, dst=0)

        # Merge all quantizer state dicts into one
        merged_quantizer_state_dict = {}
        for quantizer_state_dict in all_quantizer_state_dicts:
            if quantizer_state_dict is not None:
                merged_quantizer_state_dict.update(quantizer_state_dict)

        torch.save(merged_quantizer_state_dict, save_directory + "/quantizer_state.pth")
    else:
        # Other ranks just send their amax values
        torch.distributed.gather_object(quantizer_state_dict, None, dst=0)

    torch.distributed.barrier()


class VllmFqGPTModelExporter(GPTModelExporter):
    """VLLM fakequant GPTModel exporter."""

    def save_pretrained(
        self,
        save_directory: str | os.PathLike,
        pretrained_model_name_or_path: str | os.PathLike,
    ):
        os.makedirs(save_directory, exist_ok=True)
        gather_mcore_vllm_fq_quantized_state_dict(self.model, self.state_dict, save_directory)

        # NOTE: `self.state_dict` is an OrderedDict; mutating it while iterating
        # over its keys raises "OrderedDict mutated during iteration".
        keys_to_remove = [k for k in self.state_dict if "quantizer" in k]
        for k in keys_to_remove:
            self.state_dict.pop(k, None)

        assert not (self.is_multimodal and pretrained_model_name_or_path is not None), (
            "Exporting weights in bf16 and amax values is not supported for multimodal models "
            "when pretrained_model_name_or_path is not None"
        )
        assert not self.export_extra_modules, (
            "Exporting extra modules is not supported for vLLM fakequant"
        )
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
    )
    exporter.save_pretrained(export_dir, pretrained_model_name_or_path)
