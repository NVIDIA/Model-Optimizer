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


# Serving layout -> this module's native (nn.Sequential) layout. LiLiCorrExporter writes
# the serving names, which is what vLLM and SGLang both require; warm-starting from such
# an export has to translate them back. Values are (serving_suffix, native_suffix).
_LILICORR_SERVING_TO_NATIVE = (
    (
        re.compile(r"^(?P<p>.*)lilicorr\.feature_norm\.(?P<t>weight|bias)$"),
        r"\g<p>lilicorr.feature_mlp.0.\g<t>",
    ),
    (
        re.compile(r"^(?P<p>.*)lilicorr\.feature_mlp\.up_proj\.(?P<t>weight|bias)$"),
        r"\g<p>lilicorr.feature_mlp.1.\g<t>",
    ),
    (
        re.compile(r"^(?P<p>.*)lilicorr\.feature_mlp\.down_proj\.(?P<t>weight|bias)$"),
        r"\g<p>lilicorr.feature_mlp.3.\g<t>",
    ),
    (
        re.compile(r"^(?P<p>.*)lilicorr\.layers\.(?P<i>\d+)\.mlp\.up_proj\.(?P<t>weight|bias)$"),
        r"\g<p>lilicorr.layers.\g<i>.mlp.0.\g<t>",
    ),
    (
        re.compile(r"^(?P<p>.*)lilicorr\.layers\.(?P<i>\d+)\.mlp\.down_proj\.(?P<t>weight|bias)$"),
        r"\g<p>lilicorr.layers.\g<i>.mlp.2.\g<t>",
    ),
)


def remap_lilicorr_serving_keys(state_dict, prefix, *args, **kwargs):
    """Accept the LiLiCorr reranker in serving or native layout.

    ``LiLiCorrHead`` builds its MLPs as ``nn.Sequential`` (positional child names) while
    both serving stacks build the same arithmetic as named ``up_proj``/``down_proj``
    submodules with the feature LayerNorm lifted to a sibling ``feature_norm``.
    ``LiLiCorrExporter`` writes the serving names, so warm-starting from an exported
    drafter arrives in that layout and has to be translated back.
    """
    for key in [k for k in state_dict if k.startswith(prefix)]:
        for pattern, replacement in _LILICORR_SERVING_TO_NATIVE:
            native = pattern.sub(replacement, key)
            if native == key:
                continue
            value = state_dict.pop(key)
            # A native key already present wins, and we say which copy was dropped: the
            # two layouts hold the same tensor, so picking silently would be invisible.
            if native in state_dict:
                logger.warning(
                    "LiLiCorr: ignoring %s because %s is also present in the checkpoint.",
                    key,
                    native,
                )
            else:
                state_dict[native] = value
            break
