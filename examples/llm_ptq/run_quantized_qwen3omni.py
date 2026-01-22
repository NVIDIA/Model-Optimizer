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

# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Script to load and run a quantized Qwen3Omni model from mto checkpoint."""

import argparse
import time

import torch
from qwen_omni_utils import process_mm_info
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

import modelopt.torch.opt as mto


def main(args):
    print(f"Loading base model from {args.model_path}...")
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype="auto",
        device_map="cuda",
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    )

    print(f"Restoring quantized state from {args.checkpoint_path}...")
    model = mto.restore(model, args.checkpoint_path)

    model.disable_talker()

    print("Loading processor...")
    processor = Qwen3OmniMoeProcessor.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )

    # Build conversation with user prompt
    prompt = args.prompt or "What is the capital of France?"
    conversation = [{"role": "user", "content": [{"type": "text", "text": f"{prompt}"}]}]
    conversations = [conversation]

    # Set whether to use audio in video
    use_audio_in_video = True

    # Preparation for inference
    texts = processor.apply_chat_template(conversations, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conversations, use_audio_in_video=use_audio_in_video)

    inputs = processor(
        text=texts,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=use_audio_in_video,
    )
    inputs = inputs.to(model.device).to(model.dtype)

    print(f"\nPrompt: {prompt}")
    print("Generating...")

    start_time = time.time()
    with torch.no_grad():
        text_ids, _ = model.generate(
            **inputs,
            thinker_return_dict_in_generate=True,
            use_audio_in_video=use_audio_in_video,
            max_new_tokens=args.max_new_tokens,
            return_audio=False,
        )
    end_time = time.time()
    print(f"Time taken for generation: {end_time - start_time:.2f} seconds")

    # Decode the generated tokens
    generated_text = processor.batch_decode(
        text_ids.sequences[:, inputs["input_ids"].shape[1] :],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    print(f"\nGenerated: {generated_text[0]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run quantized Qwen3Omni model")
    parser.add_argument(
        "--model_path",
        type=str,
        default="Qwen/Qwen3-Omni-30B-A3B-Instruct",
        help="Path to the base Qwen3Omni model (HF format)",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="/home/scratch.arasane_hw/models/qwen3omni_nvfp4_qkv_disabled_text_bs512_calib512.pt",
        help="Path to the mto.save() quantized checkpoint",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Text prompt for generation",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum new tokens to generate",
    )

    args = parser.parse_args()
    main(args)
