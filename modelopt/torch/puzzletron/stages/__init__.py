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

"""Composable Puzzletron stage handlers.

The runner accepts an explicit handler map so stage implementation can migrate
piece by piece without hard-wiring GPU-heavy imports into config preflight.
"""

from . import graph as _graph
from .convert import convert_stage
from .depth import depth_stage
from .diagnostics import (
    activation_diagnostic_stage,
    bypass_diagnostic_stage,
    sort_equivalence_stage,
    width_slice_equivalence_stage,
)
from .future import (
    aiperf_stage,
    distillation_overfit_stage,
    distillation_stage,
    evaluation_stage,
    post_distillation_evaluation_stage,
)
from .graph import *
from .pipeline import (
    activation_stage,
    build_library_stage,
    bypass_overfit_stage,
    bypass_stage,
    mip_stage,
    scoring_stage,
    sort_stage,
    vllm_stats_stage,
)

DEFAULT_HANDLERS = {
    "convert": convert_stage,
    "width_importance": activation_stage,
    "width_sanity": activation_diagnostic_stage,
    "slicing_sanity": width_slice_equivalence_stage,
    "sort_sanity": sort_equivalence_stage,
    "sort": sort_stage,
    "bypass_sanity": bypass_overfit_stage,
    "bypass": bypass_stage,
    "build_library": build_library_stage,
    "vllm_stats": vllm_stats_stage,
    "replacement_scoring": scoring_stage,
    "mip": mip_stage,
    "depth_importance": depth_stage,
    "aiperf": aiperf_stage,
    "global_distillation_sanity": distillation_overfit_stage,
    "global_distillation": distillation_stage,
    "zero_shot_evaluation": evaluation_stage,
    "post_distillation_evaluation": post_distillation_evaluation_stage,
}

__all__ = [
    *_graph.__all__,
    "DEFAULT_HANDLERS",
    "activation_stage",
    "activation_diagnostic_stage",
    "bypass_diagnostic_stage",
    "aiperf_stage",
    "build_library_stage",
    "bypass_overfit_stage",
    "bypass_stage",
    "convert_stage",
    "distillation_stage",
    "distillation_overfit_stage",
    "depth_stage",
    "evaluation_stage",
    "post_distillation_evaluation_stage",
    "mip_stage",
    "scoring_stage",
    "sort_stage",
    "sort_equivalence_stage",
    "width_slice_equivalence_stage",
    "vllm_stats_stage",
]
