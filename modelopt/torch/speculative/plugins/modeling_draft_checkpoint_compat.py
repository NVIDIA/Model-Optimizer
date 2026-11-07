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

"""Checkpoint key remaps for loading exported and published drafters.

ModelOpt's draft modules define one canonical parameter layout. Serving exports and
published drafters can use different key names; the load hooks here translate those
names back to the module layout.

Each entry is a ``load_state_dict`` pre-hook body: it mutates ``state_dict`` in place to
move released-layout keys onto the canonical names.
"""

import logging
import re

logger = logging.getLogger(__name__)

# DSpark modules keep Markov heads flat; serving checkpoints can nest them.
_DSPARK_NESTED_HEAD_PREFIX = "markov_head."


def remap_dspark_nested_head_keys(state_dict, prefix, *args, **kwargs):
    """Accept DSpark Markov and confidence heads in serving or native layouts."""
    nested = prefix + _DSPARK_NESTED_HEAD_PREFIX
    confidence = prefix + "confidence_head.proj."
    for key in [k for k in state_dict if k.startswith((nested, confidence))]:
        flat = (
            prefix + key[len(nested) :]
            if key.startswith(nested)
            else prefix + "confidence_proj." + key[len(confidence) :]
        )
        nested_value = state_dict.pop(key)
        # A flat key already present wins. The Markov tables are ~14% of the draft's
        # parameters and loading the wrong copy is invisible at runtime, so say which was
        # dropped rather than picking silently.
        if flat in state_dict:
            logger.warning(
                "DSpark: ignoring nested %s because %s is also present in the checkpoint.",
                key,
                flat,
            )
        else:
            state_dict[flat] = nested_value


_LILICORR_VLLM_TO_MODELOPT_REWRITES = (
    (r"lilicorr\.feature_norm(?=\.|$)", "lilicorr.feature_mlp.0"),
    (r"lilicorr\.feature_mlp\.up_proj(?=\.|$)", "lilicorr.feature_mlp.1"),
    (r"lilicorr\.feature_mlp\.down_proj(?=\.|$)", "lilicorr.feature_mlp.3"),
    (r"(lilicorr\.layers\.\d+\.mlp)\.up_proj(?=\.|$)", r"\1.0"),
    (r"(lilicorr\.layers\.\d+\.mlp)\.down_proj(?=\.|$)", r"\1.2"),
)


def remap_lilicorr_vllm_head_keys(state_dict, prefix, *args, **kwargs):
    """Accept vLLM-compatible LiLiCorr exports when warm-starting ModelOpt."""
    for key in list(state_dict):
        modelopt_key = key
        for pattern, replacement in _LILICORR_VLLM_TO_MODELOPT_REWRITES:
            modelopt_key = re.sub(pattern, replacement, modelopt_key)
        if modelopt_key == key:
            continue

        exported_value = state_dict.pop(key)
        if modelopt_key in state_dict:
            logger.warning(
                "LiLiCorr: ignoring exported-layout %s because %s is also present in "
                "the checkpoint.",
                key,
                modelopt_key,
            )
        else:
            state_dict[modelopt_key] = exported_value
