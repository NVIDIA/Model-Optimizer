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

"""Pytorch plugins needed for NAS and dynamic modules."""

import torch.nn as nn

from ..modules import _DynamicBatchNorm
from ..registry import DMRegistry

__all__ = []


def wrapped_convert_sync_batchnorm(cls: type[nn.Module], /, module: nn.Module, *args, **kwargs):
    """Extend the original convert_sync_batchnorm to handle dynamic modules.

    This method ensures that _DynamicBatchNorm instances are correctly
    converted into DynamicSyncBatchNorm instances.

    .. note::

        We explicitly use ``del`` here following the original implementation!
    """
    # handle vanilla case
    if not isinstance(module, _DynamicBatchNorm):
        return cls.old_convert_sync_batchnorm(module, *args, **kwargs)

    # maintain hparam objects
    hparams = dict(module.named_hparams())

    # set all hparams to max value and retain old active values
    val_actives = {name: hp.active for name, hp in hparams.items()}
    for hp in hparams.values():
        hp.active = hp.max

    # export current module
    module = module.export()

    # convert exported DynamicBN to SyncBN using the original method
    module = cls.old_convert_sync_batchnorm(module, *args, **kwargs)

    # ensure that we indeed got a SyncBN
    assert isinstance(module, nn.SyncBatchNorm), f"Expected SyncBatchNorm, got {type(module)}!"

    # convert SyncBN to DynamicSyncBN
    module = DMRegistry.convert(module)

    # re-use hparams from original dynamic BN to maintain consistency and re-assign active val
    for hp_name, hp in hparams.items():
        setattr(module, hp_name, hp)
        hp.active = val_actives[hp_name]

    # return the new module
    return module


# hook into convert_sync_batchnorm
nn.SyncBatchNorm.old_convert_sync_batchnorm = nn.SyncBatchNorm.convert_sync_batchnorm
nn.SyncBatchNorm.convert_sync_batchnorm = classmethod(wrapped_convert_sync_batchnorm)
