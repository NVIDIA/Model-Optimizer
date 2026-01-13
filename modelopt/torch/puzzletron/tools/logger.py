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
# mypy: ignore-errors
import inspect
import logging
import os
import sys

import torch.distributed.launch  # noqa: F401

logging.getLogger("fsspec.local").setLevel(logging.ERROR)
logging.getLogger("websockets.client").setLevel(logging.WARN)
logging.getLogger("websockets.server").setLevel(logging.WARN)
logging.getLogger("websockets.server:connection").setLevel(logging.WARN)


class LogColors:
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"

    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    RESET = "\033[0m"


class DistributedLogger(logging.Logger):
    verbosity = logging.ERROR

    def __init__(self, name, level=logging.DEBUG):
        super().__init__(name, level)
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.global_rank = int(os.environ.get("RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))

    def dist_log(self, msg: str, ranks: str = "main"):
        """Log parameter msg with the given ranks.

        Args:
            msg: The message to log.
            ranks: The ranks to log the message to. Choices are:
                "all": log with all ranks
                "main": log with only rank 0 in node 0
                "last": log with only rank -1 in node 0
                "local_main": log with only rank 0 in all nodes
        """
        # print(msg, ranks)
        if ranks not in ["all", "main", "local_main", "last"]:
            raise NotImplementedError(
                f"Could not broadcast msg {msg} - "
                f"ranks parameters choices are ['all', 'main', 'local_main']. Got {ranks}"
            )
        # All ranks to print
        if ranks == "all":
            pass

        # Only main rank at node 0 to print
        elif (
            (ranks == "main" and self.global_rank != 0)
            or (ranks == "last" and self.local_rank != self.world_size - 1)
            or (ranks == "local_main" and self.local_rank != 0)
        ):
            return

        message_source = self.get_caller_location()

        self.info(
            f"{LogColors.GREEN}[rank-{self.global_rank}]{LogColors.RESET}[{message_source}]\t{msg}"
        )

    # def dist_warning(self, msg):
    #     if self.verbosity <= logging.WARNING:
    #         self.warning(f"[rank-{self.global_rank}] " + msg)

    @staticmethod
    def get_caller_location() -> str:
        # Get the caller's stack frame
        frame = inspect.currentframe()

        # f_back -> class method, 2 x f_back -> utils method, 3 x f_back -> original source
        caller_frame = frame.f_back.f_back.f_back

        # Get the filename and line number from the caller's stack frame
        filename = os.path.basename(caller_frame.f_code.co_filename)
        lineno = caller_frame.f_lineno
        return f"{filename}:{lineno}"


# Initialize logger
logging.setLoggerClass(DistributedLogger)
logger = logging.getLogger(__name__)
logger.propagate = False

formatter = logging.Formatter("[%(asctime)s]%(message)s")
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(formatter)
handler.setLevel(logging.DEBUG)
logger.addHandler(handler)

# Manually edit torch logger
torch_logger = logging.getLogger("torch")
torch_logger.handlers = logger.handlers
torch_logger.propagate = False

# Manually edit deepspeed logger

# Show some love to Mac & Windows users who can't easily install deepspeed ;)
# This is allowing running tests on Mac & Windows and train in non-DDP
try:
    from deepspeed.utils import logger as deepspeed_logger

    deepspeed_logger.handlers = logger.handlers
    deepspeed_logger.propagate = False
except ImportError:
    # If deepspeed is not installed - no op
    pass

# Define a custom function to redirect warnings to logger
# def custom_warning_handler(message, category, filename, lineno, file=None, line=None):
#     logger.dist_warning(f'{category.__name__}: {message} (in {filename}, line {lineno})')


# Use the custom warning handler
# warnings.showwarning = custom_warning_handler

logger: DistributedLogger


def aprint(msg: str | None):
    """
    All ranks from all nodes prints
    """
    return logger.dist_log(msg=msg, ranks="all")


def lmprint(msg: str | None):
    """
    All local main ranks prints (rank 0 in each node)
    """
    return logger.dist_log(msg=msg, ranks="local_main")


def mprint(msg: str | None):
    """
    Master prints only (rank 0 in node 0)
    """
    return logger.dist_log(msg=msg, ranks="main")


def lprint(msg: str | None):
    """
    Last rank prints only (rank -1 in node 0)
    """
    return logger.dist_log(msg=msg, ranks="last")
