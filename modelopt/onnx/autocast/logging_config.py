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

# Create a parent logger for all AutoCast components
logger = logging.getLogger("autocast")


def configure_logging(level=logging.INFO, log_file=None):
    """Configure logging for all AutoCast components.

    Args:
        level: The logging level to use (default: logging.INFO).
        log_file: Optional path to a log file. If provided, logs will be written to this file
                 in addition to stdout (default: None).
    """
    # Set level for the parent logger and all child loggers
    logger.setLevel(level)

    # Remove any existing handlers to ensure clean configuration
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Add file handler if log_file is specified
    if log_file:
        try:
            # Create directory if it doesn't exist
            log_dir = os.path.dirname(log_file)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)

            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(filename)s - %(message)s"
            )
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            logger.info(f"Logging configured to write to file: {log_file}")
        except Exception as e:
            logging.error(f"Failed to setup file logging to {log_file}: {e!s}")

    # Allow log messages to propagate to the root logger for testing compatibility
    # This enables pytest's caplog fixture to capture logs while still maintaining
    # our custom formatting through the handlers above
    logger.propagate = True

    # Ensure all child loggers inherit the level setting
    for name in logging.root.manager.loggerDict:
        if name.startswith("autocast"):
            logging.getLogger(name).setLevel(level)


# Configure with default settings if not already configured
configure_logging()
