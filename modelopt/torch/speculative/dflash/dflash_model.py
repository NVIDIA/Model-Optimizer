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

"""DFlash model to support block-wise parallel speculative decoding."""

from modelopt.torch.opt.dynamic import DynamicModule


class DFlashModel(DynamicModule):
    """Base DFlash Model."""

    def _setup(self):
        """Register temporary attributes for the DFlash module."""
        self._register_temp_attribute("dflash_module", None)

    def modify(self, config):
        """Base DFlash Model modify function. Child class should implement the details."""
        self.dflash_block_size = config.dflash_block_size
        self.dflash_freeze_base_model = config.dflash_freeze_base_model
        self.dflash_loss_decay_factor = config.dflash_loss_decay_factor
        self.dflash_self_logit_distillation = config.dflash_self_logit_distillation
        self.dflash_num_anchors = config.dflash_num_anchors
        self.dflash_report_acc = config.dflash_report_acc
        self.dflash_use_torch_compile = config.dflash_use_torch_compile
