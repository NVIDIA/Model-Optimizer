#!/usr/bin/env python3
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

"""
Simple test for Puzzletron DeciLM Bridge.

Usage:
     export PYTHONPATH="/workspace/Megatron-Bridge/src:/workspace/Model-Optimizer:${PYTHONPATH}"
     python test_puzzletron_decilm_bridge.py /workspace/puzzle_dir_decilm/ckpts/teacher
"""

import sys

# Import bridge to register it
import puzzletron_decilm_bridge  # noqa: F401
from megatron.bridge.models.conversion.auto_bridge import AutoBridge
from megatron.bridge.models.conversion.model_bridge import get_model_bridge

print("Testing Puzzletron DeciLM Bridge...")
print()

# Test 1: Check if bridge is registered
bridge = get_model_bridge("DeciLMForCausalLM")
print(f"✓ Bridge registered: {type(bridge).__name__}")

# Test 2: Check if it's in supported models
supported = AutoBridge.list_supported_models()
if "DeciLMForCausalLM" in supported:
    print("✓ DeciLMForCausalLM in supported models list")
else:
    print(f"⚠ DeciLMForCausalLM not in supported models list (found {len(supported)} models)")

# Test 3: Try to load checkpoint (if path provided)
if len(sys.argv) > 1:
    checkpoint_path = sys.argv[1]
    print(f"\nTesting with checkpoint: {checkpoint_path}")

    # Try to create bridge from checkpoint
    bridge = AutoBridge.from_hf_pretrained(checkpoint_path, trust_remote_code=True)
    print("✓ AutoBridge.from_hf_pretrained() succeeded")
    print(f"  Bridge type: {type(bridge._model_bridge).__name__}")

    # Try to get provider (will fail with NotImplementedError)
    provider = bridge.to_megatron_provider(load_weights=False)
    print("✓ Bridge fully working!")

print("\n✓ Basic bridge registration test passed!")
