# Adapted from https://github.com/hemingkx/Spec-Bench/tree/66230f10cb0a02aced5ef3ce1e85163c16160454/model/medusa
# SPDX-FileCopyrightText: Copyright (c) 2024 Heming Xia. All rights reserved.
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

import asyncio
import os
import sys
import time

import torch

from .base import Model

# Medusa dependencies from Spec-Bench
try:
    spec_bench_path = os.path.join(os.getcwd(), "Spec-Bench")
    sys.path.insert(0, spec_bench_path)
    from model.medusa.kv_cache import initialize_past_key_values
    from model.medusa.medusa_choices import mc_sim_7b_63
    from model.medusa.medusa_model import MedusaModel
    from model.medusa.utils import (
        evaluate_posterior,
        generate_candidates,
        generate_medusa_buffers,
        initialize_medusa,
        reset_medusa_mode,
        tree_decoding,
        update_inference_inputs,
    )
except ImportError as e:
    print(f"Medusa dependencies not found: {e}")
    MedusaModel = None


class SpecBenchMedusaModel(Model):
    def __init__(
        self,
        model_dir,
        max_concurrent_requests,
        sampling_kwargs,
        use_draft_logits=False,
        **kwargs,
    ):
        if MedusaModel is None:
            raise ImportError(
                "Medusa dependencies not found. Please ensure Spec-Bench is available."
            )
        assert max_concurrent_requests == 1, "Only support batch size 1 for now!"
        self.medusa_num_heads = kwargs.get("medusa_num_heads", 4)
        self.draft_model_path = kwargs.get("draft_model_dir")
        self.dtype = kwargs.get("dtype", "float16")
        self.max_steps = kwargs.get("max_steps", 512)

        # Medusa decoding parameters
        self.temperature = sampling_kwargs.get("temperature", 0.0)
        self.posterior_threshold = kwargs.get("posterior_threshold", 0.09)
        self.posterior_alpha = kwargs.get("posterior_alpha", 0.3)
        self.medusa_choices = kwargs.get("medusa_choices", mc_sim_7b_63)

        # Convert dtype string to torch dtype
        dtype_map = {
            "float32": torch.float32,
            "float64": torch.float64,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        torch_dtype = dtype_map.get(self.dtype, torch.float16)

        # Load the Medusa model
        # Use single GPU to avoid device mismatch issues with device_map="auto"
        self.device = torch.device(kwargs.get("device", "cuda:0"))
        self.model = MedusaModel.from_pretrained(
            self.draft_model_path,
            model_dir,
            medusa_num_heads=self.medusa_num_heads,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
        )
        self.model = self.model.to(self.device)

        self.sampling_kwargs = sampling_kwargs

    def _medusa_forward(self, input_ids, max_new_tokens, end_id):
        """
        Run Medusa speculative decoding forward pass.

        Returns:
            tuple: (output_ids, new_token_count, num_steps, accept_length_list, timing)
        """
        # Avoid modifying the input_ids in-place
        accept_length_list = []
        input_ids = input_ids.clone()
        timing = [time.perf_counter()]

        # Cache medusa buffers (the fixed patterns for tree attention)
        if (
            hasattr(self.model, "medusa_choices")
            and self.model.medusa_choices == self.medusa_choices
        ):
            medusa_buffers = self.model.medusa_buffers
        else:
            medusa_buffers = generate_medusa_buffers(self.medusa_choices, device=self.device)
        self.model.medusa_buffers = medusa_buffers
        self.model.medusa_choices = self.medusa_choices

        # Initialize the past key and value states
        if hasattr(self.model, "past_key_values"):
            past_key_values = self.model.past_key_values
            past_key_values_data = self.model.past_key_values_data
            current_length_data = self.model.current_length_data
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.model.base_model)
            self.model.past_key_values = past_key_values
            self.model.past_key_values_data = past_key_values_data
            self.model.current_length_data = current_length_data

        input_len = input_ids.shape[1]
        cur_length = input_len
        reset_medusa_mode(self.model)
        medusa_logits, logits = initialize_medusa(
            input_ids, self.model, medusa_buffers["medusa_attn_mask"], past_key_values
        )
        new_token = 0

        for idx in range(self.max_steps):
            candidates, tree_candidates = generate_candidates(
                medusa_logits,
                logits,
                medusa_buffers["tree_indices"],
                medusa_buffers["retrieve_indices"],
            )
            medusa_logits, logits, outputs = tree_decoding(
                self.model,
                tree_candidates,
                past_key_values,
                medusa_buffers["medusa_position_ids"],
                input_ids,
                medusa_buffers["retrieve_indices"],
            )
            best_candidate, accept_length = evaluate_posterior(
                logits,
                candidates,
                self.temperature,
                self.posterior_threshold,
                self.posterior_alpha,
            )
            input_ids, logits, medusa_logits, new_token = update_inference_inputs(
                input_ids,
                candidates,
                best_candidate,
                accept_length,
                medusa_buffers["retrieve_indices"],
                outputs,
                logits,
                medusa_logits,
                new_token,
                past_key_values_data,
                current_length_data,
            )
            accept_length_tree = input_ids.shape[1] - cur_length
            cur_length = accept_length_tree + cur_length
            accept_length_list.append(accept_length_tree)
            timing.append(time.perf_counter())

            if end_id in input_ids[0, input_len:].tolist():
                break
            if new_token > max_new_tokens:
                break

        return input_ids, new_token, idx + 1, accept_length_list, timing

    async def run(self, prompt_ids, max_length, end_id, request_id, turn_id):
        """
        Run inference on the given prompt.

        Args:
            prompt_ids: List of input token IDs
            max_length: Maximum number of new tokens to generate
            end_id: End of sequence token ID
            request_id: Request identifier
            turn_id: Turn identifier

        Returns:
            dict with output_ids, output_logits, and token_times
        """
        output_dict = {}

        # Convert prompt_ids to tensor
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)

        # Run medusa forward pass (synchronously, but wrapped for async interface)
        (
            input_ids_out,
            new_token,
            num_steps,
            accept_length_list,
            timing,
        ) = await asyncio.to_thread(self._medusa_forward, input_ids, max_length, end_id)

        # Extract generated tokens (excluding the prompt)
        original_len = len(prompt_ids)
        generated_tokens = input_ids_out[0, original_len:].tolist()

        # Remove EOS token from output if present
        if end_id in generated_tokens:
            eos_idx = generated_tokens.index(end_id)
            generated_tokens = generated_tokens[:eos_idx]
            # Also adjust accept_length_list and timing
            # Count how many tokens we're removing
            tokens_to_remove = len(input_ids_out[0, original_len:].tolist()) - len(generated_tokens)
            if tokens_to_remove > 0 and len(accept_length_list) > 0:
                # Adjust the last accept length
                accept_length_list[-1] = max(0, accept_length_list[-1] - tokens_to_remove)
                if accept_length_list[-1] == 0:
                    accept_length_list.pop()
                    if len(timing) > 1:
                        timing.pop()

        # Format output_ids as list of list of tokens per step (for beam_width=1)
        reformatted_output_ids = [[]]
        start = 0
        for accept_len in accept_length_list:
            if accept_len > 0:
                reformatted_output_ids[0].append(generated_tokens[start : start + accept_len])
                start += accept_len

        # Handle any remaining tokens
        if start < len(generated_tokens):
            reformatted_output_ids[0].append(generated_tokens[start:])

        output_dict["output_ids"] = reformatted_output_ids
        output_dict["output_logits"] = None
        output_dict["token_times"] = timing

        return output_dict

    def stop(self):
        """Cleanup resources."""
        # Clear cached KV states to free memory
        if hasattr(self.model, "past_key_values"):
            del self.model.past_key_values
            del self.model.past_key_values_data
            del self.model.current_length_data

        # Clear medusa buffers
        if hasattr(self.model, "medusa_buffers"):
            del self.model.medusa_buffers

        # Move model to CPU or delete to free GPU memory
        if hasattr(self, "model") and self.model is not None:
            del self.model
            torch.cuda.empty_cache()
