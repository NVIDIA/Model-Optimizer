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

__all__ = ["run_backbone"]


_CAMERAS_PER_SWEEP = 6
_FEATURE_OUTPUT_NAMES = ("out.0", "out.1")


def run_backbone(version, backbone, history_backbone, stream, images):
    if version == "v1":
        return backbone(stream, img=images.squeeze(0))

    current = images[:, :_CAMERAS_PER_SWEEP].contiguous()
    history = images[:, _CAMERAS_PER_SWEEP : 2 * _CAMERAS_PER_SWEEP].contiguous()
    history_outputs = history_backbone(stream, img=history.squeeze(0))
    history_features = {
        f"prev.{index}": history_outputs[name][:, :_CAMERAS_PER_SWEEP]
        for index, name in enumerate(_FEATURE_OUTPUT_NAMES)
    }
    return backbone(
        stream,
        img=current.squeeze(0),
        **history_features,
    )
