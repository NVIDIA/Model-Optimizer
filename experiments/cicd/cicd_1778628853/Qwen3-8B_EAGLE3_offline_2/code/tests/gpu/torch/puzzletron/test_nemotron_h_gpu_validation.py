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

"""GPU validation for Nemotron-H hybrid model subblock parameter counting.

Requires HuggingFace Hub access to nvidia/NVIDIA-Nemotron-Nano-12B-v2-Base (config only,
no weights are downloaded) and mamba_ssm (CUDA).

Usage:
    pytest -v -s -o addopts= tests/gpu/puzzletron/test_nemotron_h_gpu_validation.py
"""

import copy

import pytest

import modelopt.torch.puzzletron.anymodel.models.nemotron_h_v2.nemotron_h_v2_model_descriptor  # noqa: F401
from modelopt.torch.puzzletron.anymodel.model_descriptor import (
    ModelDescriptor,
    ModelDescriptorFactory,
)
from modelopt.torch.puzzletron.block_config import FFNConfig
from modelopt.torch.puzzletron.subblock_stats.calc_subblock_params_and_memory import (
    calculate_subblock_params,
)
from modelopt.torch.puzzletron.tools.checkpoint_utils import load_model_config

MODEL_ID = "nvidia/NVIDIA-Nemotron-Nano-12B-v2-Base"


@pytest.fixture
def nemotron_descriptor():
    return ModelDescriptorFactory.get("nemotron_h_v2")


@pytest.fixture
def nemotron_config(nemotron_descriptor):
    return load_model_config(
        MODEL_ID, trust_remote_code=nemotron_descriptor.requires_trust_remote_code()
    )


def test_ffn_variants_produce_distinct_params(nemotron_config, nemotron_descriptor):
    """FFN subblocks with different intermediate_size must report different param counts.

    On hybrid models, hybrid_override_pattern must be truncated to match the subblock
    type; otherwise a single-layer model always builds layer 0 (Mamba) and every FFN
    variant reports identical param counts.
    """
    lm_config = nemotron_descriptor.get_language_model_config(nemotron_config)
    pattern = lm_config.hybrid_override_pattern.replace("|", "")
    ffn_indices = [i for i, c in enumerate(pattern) if c in ("-", "E")]
    assert ffn_indices, f"No FFN layers in pattern: {pattern}"

    teacher_size = lm_config.intermediate_size
    sizes = [teacher_size // 4, teacher_size // 2, teacher_size]

    param_counts = {}
    for size in sizes:
        layer_config = copy.deepcopy(nemotron_config)
        ModelDescriptor.truncate_pattern_for_subblock(
            nemotron_descriptor.get_language_model_config(layer_config), ffn_indices[0]
        )

        params = calculate_subblock_params(
            layer_config, FFNConfig(intermediate_size=size), nemotron_descriptor
        )
        param_counts[size] = params
        print(f"  intermediate_size={size:>8d} -> params={params:>12,}")

    assert len(set(param_counts.values())) == len(sizes), (
        f"Expected {len(sizes)} distinct param counts, got: {param_counts}"
    )
