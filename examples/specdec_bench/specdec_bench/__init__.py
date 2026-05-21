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

# Methodology version. Bump:
#   - minor (0.X.0) when adding a new metric or strictly-additive provenance field
#   - major (X.0.0) when changing how an existing metric is computed OR its
#     on-disk field names (incompatible with prior consumers / visualizers)
# The visualizer aggregates runs by major version to avoid apple-to-orange
# comparisons across methodology changes.
#
# 1.0.0: rename Request_AR / Category_AR / Average_AR → *_AL across the
#        SpecBench / AcceptanceRate / MTBench metric writers, AND add
#        Joint_Acceptance_Rate to the AcceptanceRate metric. The renamed
#        values were always acceptance LENGTH (mean tokens generated per
#        inference step), not a rate, and the visualizer reads *_AL.
#        Pre-1.0.0 runs in S3 have *_AR and no Joint_AR; they must be
#        re-run or post-processed before comparing.
__version__ = "1.0.0"
