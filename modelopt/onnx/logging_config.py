# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Logging configuration for modelopt ONNX quantization."""

import logging
import os
import sys

# Create a parent logger for all ONNX components
logger = logging.getLogger("modelopt.onnx")


def configure_logging(level=logging.INFO, log_file=None):
    """Configure logging for all ONNX components.

    Args:
        level: The logging level to use (default: logging.INFO)
        log_file: Optional path to a log file. If provided, logs will be written to this file
                 in addition to stdout (default: None)
    """
    # Set level for the parent logger and all child loggers
    logger.setLevel(level)

    # Remove any existing handlers to ensure clean configuration
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    formatter = logging.Formatter("%(asctime)s - [modelopt][onnx] - %(levelname)s - %(message)s")

    # Add file handler if log_file is specified
    if log_file:
        try:
            # Create directory if it doesn't exist
            log_dir = os.path.dirname(log_file)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)

            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            logger.info(f"Logging configured to write to file: {log_file}")
        except Exception as e:
            print(
                f"[modelopt][onnx] - ERROR - Failed to setup file logging to {log_file}: {e!s}",
                file=sys.stderr,
            )
            print("[modelopt][onnx] - INFO - Falling back to console logging.", file=sys.stderr)

    # Setup handler to print log in stdout
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Prevent log messages from propagating to the root logger
    logger.propagate = False

    # Ensure all child loggers inherit the level setting
    for name in logging.root.manager.loggerDict:
        if name.startswith("modelopt.onnx"):
            logging.getLogger(name).setLevel(level)


# Configure with default settings if not already configured
configure_logging()
