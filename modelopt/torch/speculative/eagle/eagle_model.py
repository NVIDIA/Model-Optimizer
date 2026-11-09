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

"""Eagle model to support eagle decoding."""

import torch

from modelopt.torch.opt.dynamic import DynamicModule


class EagleModel(DynamicModule):
    """Base Eagle Model."""

    def _setup(self):
        self._register_temp_attribute("eagle_module", None)

    def modify(
        self,
        config,
    ):
        """Base Eagle Model modify function. Child class should implement the details."""
        self.eagle_offline = config.eagle_offline
        self.eagle_hidden_state_distillation = config.eagle_hidden_state_distillation
        self.eagle_self_logit_distillation = config.eagle_self_logit_distillation
        self.eagle_freeze_base_model = config.eagle_freeze_base_model
        self.eagle_report_acc = config.eagle_report_acc
        self.eagle_reuse_base_decoder = config.eagle_reuse_base_decoder
        self.eagle_loss_decay_factor = config.eagle_loss_decay_factor
        self.eagle_decoder_type = config.eagle_decoder_type
        self.eagle_ttt_steps = config.eagle_ttt_steps
        self.eagle_mix_hidden_states = config.eagle_mix_hidden_states
        self.eagle_mix_hidden_states_mode = config.eagle_mix_hidden_states_mode
        self.eagle_use_torch_compile = config.eagle_use_torch_compile
        self.eagle_enable_nvtx = config.eagle_enable_nvtx
        self.eagle_export_rope_scaling = config.eagle_export_rope_scaling
        self.eagle_base_lora = config.eagle_base_lora
        self.eagle_base_lora_rank = config.eagle_base_lora_rank
        self.eagle_base_lora_alpha = config.eagle_base_lora_alpha
        self.eagle_base_lora_target_modules = config.eagle_base_lora_target_modules
        self.eagle_base_lora_preservation_loss_weight = (
            config.eagle_base_lora_preservation_loss_weight
        )
        self.eagle_base_lora_warmup_steps = config.eagle_base_lora_warmup_steps
        self.eagle_base_lora_logits_detach_prob = config.eagle_base_lora_logits_detach_prob
        self.eagle_base_lora_start_layer = config.eagle_base_lora_start_layer

    @staticmethod
    def _sample_uniform_hidden_state_history(
        hidden_states_history: list[torch.Tensor],
        *,
        process_group: torch.distributed.ProcessGroup | None = None,
        sequence_parallel: bool = False,
    ) -> torch.Tensor:
        """Sample whole hidden vectors from history in either batch- or sequence-first layout.

        With sequence parallelism, history is sequence-first and local to each rank. Sample
        global indices before slicing so SP preserves the non-SP choices and RNG consumption.
        """
        stacked = torch.stack(hidden_states_history, dim=0)
        num_candidates, dim0, dim1, hidden_size = stacked.shape
        tp_size = (
            torch.distributed.get_world_size(process_group) if process_group is not None else 1
        )
        candidate_indices = torch.randint(
            num_candidates,
            (1, dim0 * tp_size if sequence_parallel else dim0, dim1, 1),
            device=stacked.device,
        )
        if process_group is not None and tp_size > 1:
            # All TP ranks must agree on the global sampling choices.
            torch.distributed.broadcast(
                candidate_indices,
                src=torch.distributed.get_global_rank(process_group, 0),
                group=process_group,
            )
            if sequence_parallel:
                candidate_indices = candidate_indices.chunk(tp_size, dim=1)[
                    torch.distributed.get_rank(process_group)
                ]
        return stacked.gather(
            0,
            candidate_indices.expand(-1, -1, -1, hidden_size),
        ).squeeze(0)
