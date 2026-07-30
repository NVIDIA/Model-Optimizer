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
"""Re-export ``examples/hf_ptq/example_utils`` so tests can import it via
``from _test_utils.examples.hf_ptq_example_utils import example_utils``
without per-file ``sys.path`` shims.
"""

"""Offline diagnostic reports for Puzzletron artifacts."""

from .campaign_report import generate_campaign_report
from .html_report import generate_replace_block_report, generate_vllm_stats_report

__all__ = ["generate_campaign_report", "generate_replace_block_report", "generate_vllm_stats_report"]
