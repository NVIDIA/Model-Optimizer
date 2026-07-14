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

"""Dataset preparation utilities for Puzzletron."""

from .prepare_dataset import *
from .batch import DataLayout, Modality, PackedSequenceMetadata, PuzzletronBatch
from .config import PackingSpec, PuzzletronDataSpec
from .multimodal import (
    batch_from_automodel,
    load_materialized_conversation_dataset,
    load_materialized_intersyn_subset,
    materialize_intersyn_subset,
    materialize_normalized_intersyn_samples,
    normalize_intersyn_multi,
    normalize_intersyn_single,
)

__all__ = [
    "DataLayout",
    "Modality",
    "PackedSequenceMetadata",
    "PuzzletronBatch",
    "PackingSpec",
    "PuzzletronDataSpec",
    "batch_from_automodel",
    "load_materialized_conversation_dataset",
    "load_materialized_intersyn_subset",
    "materialize_intersyn_subset",
    "materialize_normalized_intersyn_samples",
    "normalize_intersyn_multi",
    "normalize_intersyn_single",
]
