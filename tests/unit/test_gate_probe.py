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

# Throwaway probe, never merged: fails on purpose so that the unit `linux` check fails and the
# `unit-watch` job of the GPU, example and regression workflows has to cancel the run.


def test_gate_probe_fails_on_purpose():
    raise AssertionError("deliberate failure to exercise the unit-watch cancel path (throwaway PR)")
