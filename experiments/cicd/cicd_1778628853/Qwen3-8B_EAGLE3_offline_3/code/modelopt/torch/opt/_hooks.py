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

"""Hooks for Pytorch to ensure compatibility with dynamic modules."""

from contextlib import ExitStack, nullcontext

import torch
import torch.distributed.checkpoint.state_dict as distributed_state_dict
import torch.nn as nn
from torch.distributed.fsdp import _flat_param
from torch.distributed.fsdp._flat_param import FlatParamHandle
from torch.distributed.fsdp._fully_shard import _fsdp_param
from torch.distributed.fsdp._fully_shard._fsdp_param import FSDPParam

from .dynamic import DynamicModule


def _safe_setattr_tensor_or_param_with_dm_check(
    module: nn.Module, param_name: str, tensor_or_param: torch.Tensor | nn.Parameter
):
    """A batched version of _safe_setattr_tensor_or_param ensuring compatibility with DMs.

    This function is called during the creation of the flat param in the FSDP module. We intercept
    this call to ensure that dynamic attributes are correctly preserved without interfering with
    FSDP's management, retrieval, and updates of parameters that might also be dynamic attributes.
    """
    with module.reset_dynamic_attributes() if isinstance(module, DynamicModule) else nullcontext():
        return _flat_param._safe_setattr_tensor_or_param_original(
            module, param_name, tensor_or_param
        )


# set the new function as the replacement
_flat_param._safe_setattr_tensor_or_param_original = _flat_param._safe_setattr_tensor_or_param
_flat_param._safe_setattr_tensor_or_param = _safe_setattr_tensor_or_param_with_dm_check


def _writeback_orig_param(self: FlatParamHandle):
    """A batched version of FlatParamHandle._writeback_orig_params ensuring compatibility with DMs.

    This function is called when we use_orig_params=True. We intercept this call to ensure that
    dynamic attributes are correctly preserved without interfering with FSDP's management,
    retrieval, and updates of parameters that might also be dynamic attributes.
    """
    with ExitStack() as stack:
        for _, module, _ in self.flat_param._param_infos:
            if isinstance(module, DynamicModule):
                stack.enter_context(module.reset_dynamic_attributes())
        return self._writeback_orig_params_original()


FlatParamHandle._writeback_orig_params_original = FlatParamHandle._writeback_orig_params
FlatParamHandle._writeback_orig_params = _writeback_orig_param


def _unsafe_setattr_param_with_dm_check(module: nn.Module, param_name: str, param: nn.Parameter):
    """A batched version of unsafe_setattr_param ensuring compatibility with DMs."""
    with module.reset_dynamic_attributes() if isinstance(module, DynamicModule) else nullcontext():
        return _fsdp_param._unsafe_setattr_param_original(module, param_name, param)


def reset_sharded_param_with_dm_check(self: FSDPParam):
    """A batched version of FSDPParam.reset_sharded_param ensuring compatibility with DMs."""
    module = self._module_info.module
    with module.reset_dynamic_attributes() if isinstance(module, DynamicModule) else nullcontext():
        self._reset_sharded_param_original()


def _get_model_state_dict_with_dm_check(model: nn.Module, *args, **kwargs):
    """A batched version of get_model_state_dict ensuring compatibility with DMs."""
    with ExitStack() as stack:
        for module in model.modules():
            if isinstance(module, DynamicModule):
                stack.enter_context(module.reset_dynamic_attributes())
        return distributed_state_dict.get_model_state_dict_original(model, *args, **kwargs)


_fsdp_param._unsafe_setattr_param_original = _fsdp_param.unsafe_setattr_param
_fsdp_param.unsafe_setattr_param = _unsafe_setattr_param_with_dm_check

FSDPParam._reset_sharded_param_original = FSDPParam.reset_sharded_param
FSDPParam.reset_sharded_param = reset_sharded_param_with_dm_check

distributed_state_dict.get_model_state_dict_original = distributed_state_dict.get_model_state_dict
distributed_state_dict.get_model_state_dict = _get_model_state_dict_with_dm_check
