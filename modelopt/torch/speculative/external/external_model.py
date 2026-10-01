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

"""External draft model to support speculative decoding with a standalone draft."""

from modelopt.torch.opt.dynamic import DynamicModule

__all__ = ["ExternalDraftModel"]


class ExternalDraftModel(DynamicModule):
    """Base external draft model.

    Unlike Eagle/Medusa/DFlash, the converted module *is* the draft: a pretrained
    causal LM that owns its embeddings and lm_head. The base (target) model is
    never grafted onto it; it contributes only cached hidden states, which are
    projected by the target lm_head to obtain teacher logits.
    """

    def _setup(self):
        self._register_temp_attribute("external_offline", True)
        self._register_temp_attribute("external_loss", "soft_ce")
        self._register_temp_attribute("external_report_acc", True)
        self._register_temp_attribute("external_top_k", 20)
        self._register_temp_attribute("external_top_p", 0.95)

    def modify(self, config):
        """Base external draft modify function. Child class should implement the details."""
        self.external_offline = config.external_offline
        self.external_loss = config.external_loss
        self.external_report_acc = config.external_report_acc
        self.external_top_k = config.external_top_k
        self.external_top_p = config.external_top_p
