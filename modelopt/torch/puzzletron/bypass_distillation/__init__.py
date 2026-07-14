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

"""AutoModel-native block-local distillation for Puzzletron."""


def launch_bypass_distillation(hydra_cfg, num_nodes: int = 1, node_index: int = 0) -> None:
    """Launch the only supported local-distillation backend."""
    backend = str(hydra_cfg.bypass.get("backend", "automodel")).lower()
    if backend != "automodel":
        raise ValueError(
            f"Unsupported bypass.backend={backend!r}; AutoModel is the only supported backend"
        )
    from ..plugins.automodel.local_kd_launch import launch_local_distillation_automodel

    launch_local_distillation_automodel(
        hydra_cfg,
        num_nodes=num_nodes,
        node_index=node_index,
    )

__all__ = ["launch_bypass_distillation"]
