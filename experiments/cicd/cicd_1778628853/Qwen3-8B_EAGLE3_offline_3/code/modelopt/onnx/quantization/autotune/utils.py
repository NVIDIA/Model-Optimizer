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

"""Utility functions related to Autotune."""

import argparse
import sys
from pathlib import Path

from modelopt.onnx.logging_config import logger

DEFAULT_NUM_SCHEMES = 50
DEFAULT_WARMUP_RUNS = 50
DEFAULT_TIMING_RUNS = 100

MODE_PRESETS = {
    "quick": {"schemes_per_region": 30, "warmup_runs": 10, "timing_runs": 50},
    "default": {
        "schemes_per_region": DEFAULT_NUM_SCHEMES,
        "warmup_runs": DEFAULT_WARMUP_RUNS,
        "timing_runs": DEFAULT_TIMING_RUNS,
    },
    "extensive": {"schemes_per_region": 200, "warmup_runs": 50, "timing_runs": 200},
}


class StoreWithExplicitFlag(argparse.Action):
    """Store the value and set an 'explicit' flag on the namespace so mode presets do not override."""

    def __init__(self, explicit_attr: str, *args, **kwargs):
        """Initialize explicit attribute flag."""
        self._explicit_attr = explicit_attr
        super().__init__(*args, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        """Set attributes."""
        setattr(namespace, self.dest, values)
        setattr(namespace, self._explicit_attr, True)


def validate_file_path(path: str | None, description: str) -> Path | None:
    """Validate that a file path exists.

    Args:
        path: Path string to validate (can be None)
        description: Description of the file for error messages

    Returns:
        Path object if valid, None if path is None

    Raises:
        SystemExit: If path is provided but doesn't exist
    """
    if path is None:
        return None

    path_obj = Path(path)
    if not path_obj.exists():
        logger.error(f"{description} not found: {path_obj}")
        sys.exit(1)

    return path_obj


def get_node_filter_list(node_filter_list_path: str) -> list | None:
    """Extract node filter list from node filters path.

    Args:
        node_filter_list_path: Path to a file containing wildcard patterns to filter ONNX nodes (one pattern per line).

    Returns:
        Node filter list
    """
    node_filter_list = None
    if node_filter_list_path:
        filter_file = validate_file_path(node_filter_list_path, "Node filter list file")
        if filter_file:
            with open(filter_file) as f:
                node_filter_list = [
                    line.strip() for line in f if line.strip() and not line.strip().startswith("#")
                ]
            logger.info(f"Loaded {len(node_filter_list)} filter patterns from {filter_file}")
    return node_filter_list
