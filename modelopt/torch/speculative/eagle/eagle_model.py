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

from modelopt.torch.opt.dynamic import DynamicModule


class EagleModel(DynamicModule):
    """Base Eagle Model."""

    def _setup(self):
        self._register_temp_attribute("eagle_module", None)

    def modify(
        self,
        eagle_offline,
        eagle_hidden_state_distillation,
        eagle_self_logit_distillation,
        eagle_freeze_base_model,
        eagle_report_acc,
        eagle_reuse_base_decoder,
        eagle_loss_decay_factor,
        eagle_architecture_config,
        eagle_decoder_type,
    ):
        """Base Eagle Model modify function. Child class should implement the details."""
        self.eagle_offline = eagle_offline
        self.eagle_hidden_state_distillation = eagle_hidden_state_distillation
        self.eagle_self_logit_distillation = eagle_self_logit_distillation
        self.eagle_freeze_base_model = eagle_freeze_base_model
        self.eagle_report_acc = eagle_report_acc
        self.eagle_reuse_base_decoder = eagle_reuse_base_decoder
        self.eagle_loss_decay_factor = eagle_loss_decay_factor
        self.eagle_decoder_type = eagle_decoder_type
