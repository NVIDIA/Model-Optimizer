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
# mypy: ignore-errors
import torch
from torch import nn
from torch.nn.utils.prune import custom_from_mask

from modelopt.torch.puzzletron.decilm.deci_lm_hf_code.modeling_decilm import DeciLMForCausalLM

"""
Converts a state dictionary from PyTorch's pruning format (with _orig and _mask suffixes)
into a standard format with sparsified weights.
"""


class SparsityMethod:
    def calculate_masks(self, state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Gets a model state_dict, returns a state_dict-like mask_dict with masks"""

    @staticmethod
    def fix_state_dict_inplace(state_dict, verbose=False, change_dtype=False):
        sparsity_masks = {}
        for name in list(state_dict.keys()):
            original_name = name.replace("_orig", "")
            mask_name = original_name + "_mask"
            if name[-4:] == "orig" and mask_name in state_dict:
                val = state_dict[name]
                mask = state_dict[name[:-4] + "mask"]
                val[mask == 0] = 0
                sparsity = (val == 0).sum() / mask.numel()
                sparsity_masks[original_name[:-7]] = mask
                if verbose:
                    print(f"fix_state_dict_inplace: {name} {sparsity=}")
                del state_dict[mask_name]
                del state_dict[name]
                state_dict[original_name] = val
        if change_dtype:
            for name in state_dict:
                state_dict[name] = state_dict[name].to(torch.bfloat16)
        return state_dict, sparsity_masks

    def filter_function(self):
        pass

    def apply_masks(self, model: nn.Module, mask_dict: dict[str, torch.Tensor]) -> None:
        for name, module in model.named_modules():
            if name in mask_dict:
                custom_from_mask(module, "weight", mask_dict[name].to(module.weight.device))
                print(name)
                print(torch.sum(mask_dict[name]) / mask_dict[name].numel())

    def do_sparsity(self, model: DeciLMForCausalLM, mask_dict=None):
        full_name_layers = []
        for block_idx, block_config in enumerate(model.config.block_configs):
            ffn_names = block_config.ffn.sparsify  # layers_to_sparsify_pattern[block_idx]
            att_name = block_config.attention.sparsify
            block = model.model.layers[block_idx]
            if hasattr(block, "mlp"):
                for name, m in block.mlp.named_modules():
                    if isinstance(m, torch.nn.Linear) and self.filter_function(name, ffn_names):
                        full_name_layers.append(
                            "model.layers." + str(block_idx) + "." + "mlp." + name
                        )
            if hasattr(block, "self_attn"):
                for name, m in block.self_attn.named_modules():
                    if isinstance(m, torch.nn.Linear) and self.filter_function(name, att_name):
                        full_name_layers.append(
                            "model.layers." + str(block_idx) + "." + "self_attn." + name
                        )

        if mask_dict is None:
            state_dict_for_sparsifying = {
                k.rstrip(".weight"): v
                for k, v in model.state_dict().items()
                if k.rstrip(".weight") in full_name_layers
            }
            mask_dict = self.calculate_masks(state_dict_for_sparsifying)
        # print('Apply sparsity')
        # print(full_name_layers)
        # print(model.state_dict().keys())
        # print(list(mask_dict.keys()))

        self.apply_masks(model, mask_dict)


class SparsityMethod2o4(SparsityMethod):
    def calculate_masks(self, state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Gets a model state_dict, returns a state_dict-like mask_dict with masks"""
        mask_dict = {}
        for key, val in state_dict.items():
            orig_size = val.shape
            scores = val.flatten() ** 2
            mask = self.create_mask(scores)
            mask = mask.reshape(orig_size)
            mask_dict[key] = mask
        return mask_dict

    def create_mask(self, score, value=0):
        score = score  # .cpu()
        orig_size = score.shape
        score = score.view(-1, 4)
        mask = torch.zeros(score.shape)
        values, indices = torch.topk(score, 2, dim=1)
        rows = torch.arange(mask.size(0)).unsqueeze(-1)
        mask[rows, indices] = 1
        mask = mask.view(orig_size)
        return mask  # dev = score.device, return mask.to(dev)

    @staticmethod
    def filter_function(name, modules_to_sparsify_in_block):
        if modules_to_sparsify_in_block is None:
            return False
        return name in modules_to_sparsify_in_block
