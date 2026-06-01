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
"""Helper (run under torchrun) that reloads a quantized Megatron checkpoint via the bridge and
asserts the ModelOpt quantizers were restored. A non-zero exit fails the calling test.

Usage: torchrun --nproc_per_node <N> _load_megatron_quantized.py <hf_path> <megatron_path> <tp>
"""

import sys

import torch
from megatron.bridge import AutoBridge
from megatron.core.utils import unwrap_model

import modelopt.torch.utils.distributed as dist
from modelopt.torch.quantization.utils import is_quantized


def main(hf_path: str, megatron_path: str, tp: int) -> None:
    bridge = AutoBridge.from_hf_pretrained(hf_path)
    provider = bridge.to_megatron_provider(load_weights=False)
    provider.tensor_model_parallel_size = tp
    provider.pipeline_dtype = torch.bfloat16
    provider.finalize()
    provider.initialize_model_parallel(seed=0)

    model = bridge.load_megatron_model(
        megatron_path,
        mp_overrides={"tensor_model_parallel_size": tp},
        wrap_with_ddp=False,
    )
    unwrapped_model = unwrap_model(model[0])
    assert is_quantized(unwrapped_model), "Loaded Megatron model has no quantizers restored"
    if dist.is_master():
        print("QUANTIZERS_RESTORED_OK")


if __name__ == "__main__":
    dist.setup()
    try:
        main(sys.argv[1], sys.argv[2], int(sys.argv[3]))
    finally:
        dist.cleanup()
