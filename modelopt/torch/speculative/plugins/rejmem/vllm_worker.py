# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""vLLM worker extension that installs the rejection-memory patches.

Load with ``vllm serve ... --worker-extension-cls
modelopt.torch.speculative.plugins.rejmem.vllm_worker.RejMemWorkerExtension``. vLLM imports this module in
every worker process before it builds the model runner, so the patches are in place before the DFlash
speculator exists. Each ``install()`` reads its own environment variables and is a no-op when disabled;
``vllm_weights`` always patches, since its patches only act on draft weight updates.
"""

from typing import Any

import torch

from . import vllm_capture, vllm_consumer, vllm_weights

__all__ = ["RejMemWorkerExtension"]

vllm_consumer.install()
vllm_capture.install()
vllm_weights.install()


class RejMemWorkerExtension:
    """Mixed into vLLM's GPU worker; its methods are reachable through ``collective_rpc``."""

    def rejmem_weight_info(self) -> dict[str, Any]:
        """Draft weight version, GPU UUID and launch-checkpoint manifest, for the trainer's weight push."""
        return {
            "weight_version": vllm_weights.current_weight_version(),
            "device_uuid": str(torch.cuda.get_device_properties(torch.cuda.current_device()).uuid),
            "draft_manifest": vllm_weights.draft_manifest(),
        }

    def rejmem_dropped_memory_rows(self) -> dict[str, int]:
        """Memory rows that found no KV slot since startup (synchronizes the GPU); 0 expected.

        A dict, because vLLM's ``/collective_rpc`` turns results other than dicts and lists into strings.
        """
        return {"dropped_memory_rows": vllm_consumer.dropped_memory_rows()}
