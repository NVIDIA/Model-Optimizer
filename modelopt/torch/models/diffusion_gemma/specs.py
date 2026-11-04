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

"""DiffusionGemma specs (HF model type ``diffusion_gemma``)."""

from ..specs import ModelSpec, PTQSpec, register

__all__: list[str] = []

SEQUENTIAL_DEVICE_MAP = (
    'it ties its encoder and decoder weights, and the balanced "auto" map can split a tied pair '
    "across GPUs, leaving one side on the meta device and breaking generation. Sequential packs "
    "the model onto GPU 0 first, keeping tied modules together for a checkpoint that fits; a "
    "larger one can still spill and split a tied pair, and needs an explicit single-device map"
)

# Structurally encoder-decoder, but generate() runs the whole denoising loop over the prompt and a
# canvas, and returns them concatenated.
GENERATE_INCLUDES_PROMPT = "its generate() returns the prompt and the denoised canvas concatenated"

# transformers ships the model from 5.11.
register(
    ModelSpec(
        model_type="diffusion_gemma",
        min_transformers_version="5.11",
        ptq_spec=PTQSpec(
            sequential_device_map=SEQUENTIAL_DEVICE_MAP,
            generate_includes_prompt=GENERATE_INCLUDES_PROMPT,
        ),
    )
)
