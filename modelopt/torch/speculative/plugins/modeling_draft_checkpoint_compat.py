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

"""Per-released-checkpoint key remaps for loading published drafters.

ModelOpt's draft modules define one canonical parameter layout, which is also the export
format. Some published drafters ship the same tensors under different key names; loading
those requires rewriting keys, which is a property of a specific release rather than of the
architecture. Keeping those remaps here — instead of in the module that defines the
architecture — means a new checkpoint quirk adds an entry to this file only.

Each entry is a ``load_state_dict`` pre-hook body: it mutates ``state_dict`` in place to
move released-layout keys onto the canonical names.
"""

import logging

logger = logging.getLogger(__name__)

# Head submodules that some published DSpark checkpoints nest under a `markov_head.` parent.
# ModelOpt keeps them flat (``markov_w1`` / ``markov_w2`` / ...), matching the upstream
# DeepSpec layout; ``nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16-DSpark`` nests them.
_DSPARK_NESTED_HEAD_PREFIX = "markov_head."


def remap_dspark_nested_head_keys(state_dict, prefix, *args, **kwargs):
    """Accept DSpark head weights nested under ``markov_head.`` as well as flat."""
    nested = prefix + _DSPARK_NESTED_HEAD_PREFIX
    for key in [k for k in state_dict if k.startswith(nested)]:
        flat = prefix + key[len(nested) :]
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
