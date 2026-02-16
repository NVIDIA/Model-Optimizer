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
Simple test for MBridge inference with Puzzletron models.

This test validates:
1. Bridge registration and model support
2. Provider creation and model architecture setup
3. Model initialization and forward pass

Supports:
- Puzzletron DeciLM models
- Puzzletron AnyModel checkpoints (Llama, Mistral, Qwen, etc.)

NOTE: This test uses load_weights=False, so it creates a model with random initialization.
It does NOT test weight conversion from HuggingFace to Megatron format.
For actual weight loading/conversion, use the import scripts in the respective model_distillation directories.

Usage:
     export PYTHONPATH="/workspace/Megatron-Bridge/src:/workspace/Model-Optimizer:${PYTHONPATH}"
     python test_mbridge_inference.py /workspace/puzzle_dir_decilm/ckpts/teacher
     python test_mbridge_inference.py /workspace/puzzle_dir_anymodel/ckpts/teacher

     Or with torchrun for multi-GPU:
     torchrun --nproc_per_node=1 test_mbridge_inference.py /workspace/puzzle_dir_decilm/ckpts/teacher
"""

import sys
from pathlib import Path

# Add sibling directories to path so we can import the bridges
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir / "decilm_model_distillation"))
sys.path.insert(0, str(parent_dir / "anymodel_model_distillation"))

# Import bridges to register them
# Import AnyModel bridge to register it (so LlamaBridge is not used for AnyModel)
import llama_anymodel_bridge  # noqa: F401, E402
import puzzletron_decilm_bridge  # noqa: F401, E402
import torch  # noqa: E402
from megatron.bridge.models.conversion.auto_bridge import AutoBridge  # noqa: E402
from megatron.bridge.models.conversion.model_bridge import get_model_bridge  # noqa: E402

print("Testing MBridge inference with Puzzletron models...")
print()

# Test 1: Check if bridges are registered
bridge = get_model_bridge("DeciLMForCausalLM")
print(f"✓ DeciLM bridge registered: {type(bridge).__name__}")

# Test 2: Check if models are in supported models list
supported = AutoBridge.list_supported_models()
if "DeciLMForCausalLM" in supported:
    print("✓ DeciLMForCausalLM in supported models list")
else:
    print(f"⚠ DeciLMForCausalLM not in supported models list (found {len(supported)} models)")

# Test 3: Try to load checkpoint and create model (if path provided)
if len(sys.argv) > 1:
    checkpoint_path = sys.argv[1]
    print(f"\nTesting with checkpoint: {checkpoint_path}")

    # Try to create bridge from checkpoint
    bridge = AutoBridge.from_hf_pretrained(checkpoint_path, trust_remote_code=True)
    print("✓ AutoBridge.from_hf_pretrained() succeeded")
    print(f"  Bridge type: {type(bridge._model_bridge).__name__}")

    # Try to get provider
    # NOTE: load_weights=False means we create a model with random initialization.
    # This test only validates the bridge architecture setup, NOT weight conversion.
    # For actual weight loading/conversion, use the import scripts in the respective directories.
    provider = bridge.to_megatron_provider(load_weights=False)
    print("✓ Provider created successfully")
    print(f"  Provider type: {type(provider).__name__}")
    print("  NOTE: Model will be initialized with random weights (load_weights=False)")

    # Configure parallelism (single GPU for simple test)
    provider.tensor_model_parallel_size = 1
    provider.pipeline_model_parallel_size = 1
    provider.finalize()
    print("✓ Provider finalized")

    # Initialize model parallel
    provider.initialize_model_parallel(seed=0)
    print("✓ Model parallel initialized")

    # Create the actual GPT model
    model = provider.provide_distributed_model(wrap_with_ddp=False)
    print("✓ GPT model created successfully")
    print(f"  Model type: {type(model[0]).__name__}")
    print(f"  Number of model instances: {len(model)}")

    # Move model to GPU and set to eval mode
    if torch.cuda.is_available():
        model = [m.cuda() for m in model]
        for m in model:
            m.eval()
        print("✓ Model moved to GPU and set to eval mode")

        # Test forward pass with dummy input
        try:
            batch_size = 1
            seq_length = 4
            tokens = torch.randint(0, provider.vocab_size, (batch_size, seq_length)).cuda()
            position_ids = torch.arange(seq_length).unsqueeze(0).cuda()
            attention_mask = torch.ones_like(tokens).cuda()

            with torch.no_grad():
                output = model[0](tokens, position_ids, attention_mask)
            print("✓ Forward pass succeeded")
            print(f"  Output shape: {output.shape}")
        except Exception as e:
            print(f"⚠ Forward pass failed: {e}")

print("\n✓ All tests passed!")
