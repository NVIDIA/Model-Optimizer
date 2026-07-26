# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility loaded only by Puzzletron-owned vLLM server subprocesses."""

import importlib.util
import os
import sys
import types

try:
    import torch
except ImportError:
    torch = None

if torch is not None and importlib.util.find_spec("torch._opaque_base") is None:
    # Some NVIDIA Torch 2.11 alpha builds predate torch._opaque_base.  vLLM's
    # LayerName optimization is optional, so disable it and provide the base
    # class needed to import vLLM.  Newer Torch builds retain their native API.
    os.environ["VLLM_USE_LAYERNAME"] = "0"
    opaque_base = types.ModuleType("torch._opaque_base")
    opaque_base.OpaqueBase = object
    sys.modules["torch._opaque_base"] = opaque_base
