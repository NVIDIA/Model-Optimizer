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

"""TensorRT server specific constants.

If any constant is shared by both the TensorRT client and TensorRT server,
that should go into client/constants.py module and server should import from there.
Also if we open some of these settings for the clients, they should be moved there.
"""

# Versions
TENSORRT_7_MAJOR_VERSION = 7
TENSORRT_8_MAJOR_VERSION = 8

# Sizes (unit: MB)
ONE_MEBI = 1
ONE_GIBI = 1024

ONE_MEBI_IN_BYTES = 1 << 20
ONE_GIBI_IN_BYTES = 1 << 30

# TensorRT conversion tool names
TRTEXEC = "trtexec"

# trtexec path within docker
TRTEXEC_PATH = "trtexec"
DEFAULT_ARTIFACT_DIR = "modelopt_build/trt_artifacts"

# Default conversion params
DEFAULT_VALIDATION_THRESHOLD = 1e-4
DEFAULT_MAX_BATCH_SIZE = 1
DEFAULT_ACCELERATOR = "GPU"

# With empty tactic source string trtexec will use all the available sources
DEFAULT_TACTIC_SOURCES = ""

# NVTX annotation verbosity (default, verbose or none)
DEFAULT_NVTX_MODE = "none"

# The minimum number of iterations used in kernel selection
DEFAULT_MIN_TIMING = 1

# The number of times averaged in each iteration for kernel selection
DEFAULT_AVG_TIMING = 1

# Default batch size for inference
DEFAULT_BATCH_SIZE = 1

# Default GPU Id
DEFAULT_GPU_ID = 0

# Default maximum workspace size
DEFAULT_MAX_WORKSPACE_SIZE = 256

# trtexec settings.
# Run for N milliseconds to warmup before measuring performance
WARMUP_TIME_MS = 500

# Profiling parameters
DEFAULT_PROFILING_RUNS = 1
DEFAULT_NUM_INFERENCE_PER_RUN = 500

# Layerwise profiling parameters
INPUT_DATA_KEY = "input_data"
OUTPUT_DATA_KEY = "output_data"
UNNAMED_LAYER_KEY = "other"
NODE_NAME_DELIMITER = " + "


# Class representing all TensorRT modes expressed as text
class TRTMode:
    FLOAT32 = "fp32"
    FLOAT16 = "fp16"
    BFLOAT16 = "bf16"
    FLOAT8 = "fp8"
    INT8 = "int8"
    INT4 = "int4"
    STRONGLY_TYPED = "stronglyTyped"
    BEST = "best"


TRT_MODE_FLAGS = {
    TRTMode.FLOAT32: [],
    TRTMode.FLOAT16: ["--fp16"],
    TRTMode.BFLOAT16: ["--bf16"],
    TRTMode.FLOAT8: ["--fp16", "--fp8"],
    TRTMode.INT8: ["--fp16", "--int8"],
    TRTMode.INT4: ["--fp16", "--int4"],
    TRTMode.STRONGLY_TYPED: ["--stronglyTyped"],
    TRTMode.BEST: ["--best"],
}


SHA_256_HASH_LENGTH = 32
