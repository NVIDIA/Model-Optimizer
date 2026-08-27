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

# Single source of truth for the validated `nemo-evaluator-launcher` (the 0.2.x
# path — Steps 1-9 and GDPVal). Sourced by `nel-check.sh` and `nel-gdpval.sh` so
# one edit bumps both; see references/launcher-version.md before changing it.
#
# Assigned unconditionally, never from the environment: a stale ambient value
# (e.g. sourced from `.env`) must not be able to select a different launcher.
# nel-next (`nemo-evaluator` 0.4.x) is a different package and pins separately in
# nel-next.sh.

NEL_VALIDATED_VERSION="0.2.6"
NEL_VALIDATED_SPEC="nemo-evaluator-launcher[all]==${NEL_VALIDATED_VERSION}"
