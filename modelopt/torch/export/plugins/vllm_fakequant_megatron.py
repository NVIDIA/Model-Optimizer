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
    amax_state_dict = {
        k: v.detach().clone().cpu() for k, v in state_dict.items() if k.endswith("_amax")
    }

    # Gather all amax dicts to rank 0
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()

    if rank == 0:
        # Rank 0 will collect all amax values
        all_amax_dicts = [None] * world_size
        torch.distributed.gather_object(amax_state_dict, all_amax_dicts, dst=0)

        # Merge all amax dicts into one
        merged_amax_dict = {}
        for amax_dict in all_amax_dicts:
            if amax_dict is not None:
                merged_amax_dict.update(amax_dict)

        print(f"Total amax entries from all ranks: {len(merged_amax_dict.keys())}")
        torch.save(merged_amax_dict, save_directory + "/quant_amax.pth")
    else:
        # Other ranks just send their amax values
        torch.distributed.gather_object(amax_state_dict, None, dst=0)

    torch.distributed.barrier()


class VllmFqGPTModelExporter(GPTModelExporter):
    """VLLM fakequant GPTModel exporter."""

    def save_pretrained(
        self,
        save_directory: str | os.PathLike,
        pretrained_model_name_or_path: str | os.PathLike | None = None,
    ):
        os.makedirs(save_directory, exist_ok=True)
        gather_mcore_vllm_fq_quantized_state_dict(self.model, self.state_dict, save_directory)
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


def export_mcore_gpt_to_hf_vllm_fq(
    model: torch.nn.Module,
    pretrained_model_name_or_path: str | os.PathLike | None = None,
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
