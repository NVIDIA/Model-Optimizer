# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Logging configuration module for AutoCast.

This module provides centralized logging configuration for all AutoCast components,
including console and file-based logging with customizable log levels. It ensures
consistent logging behavior across all components of the AutoCast tool.
"""

import logging
import os
import sys

# Create a logger for all AutoCast components as a child of modelopt.onnx
# This ensures autocast inherits log level and format when called from quantization
logger = logging.getLogger("modelopt.onnx.autocast")


def configure_logging(level=logging.INFO, log_file=None):
    """Configure logging for all AutoCast components.

    Args:
        level: The logging level to use. Can be a string (e.g., "DEBUG", "INFO") or
               a logging constant (e.g., logging.DEBUG). Default: logging.INFO.
        log_file: Optional path to a log file. If provided, logs will be written to this file
                 in addition to stdout (default: None).
    """
    # Check if parent logger (modelopt.onnx) already has handlers configured
    parent_logger = logging.getLogger("modelopt.onnx")
    parent_has_handlers = len(parent_logger.handlers) > 0

    # If parent is configured and no explicit level provided, inherit parent's level
    if parent_has_handlers and level == logging.INFO:
        level = parent_logger.level

    # Set level for the autocast logger (accepts both string and int)
    logger.setLevel(level)

    # Remove any existing handlers to ensure clean configuration
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(filename)s - %(message)s"
    )

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
            logging.error(f"Failed to setup file logging to {log_file}: {e!s}")

    if parent_has_handlers:
        # Parent logger is configured (called from quantization/other onnx modules)
        # Propagate to parent to use its handlers and format
        logger.propagate = True
    else:
        # Standalone mode (called directly via python3 -m modelopt.onnx.autocast)
        # Add our own console handler with autocast-specific format
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        logger.propagate = False

    # Ensure all child loggers inherit the level setting
    for name in logging.root.manager.loggerDict:
        if name.startswith("modelopt.onnx.autocast"):
            logging.getLogger(name).setLevel(level)


# Configure with default settings if not already configured
configure_logging()
