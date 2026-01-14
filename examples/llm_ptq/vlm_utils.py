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

"""Utility functions for Vision-Language Model (VLM) inference and testing."""

import os

from PIL import Image
from transformers import AutoImageProcessor, AutoProcessor, GenerationConfig


def run_vl_preview_generation(model, tokenizer, model_path, stage_name):
    """Run preview generation for VL models using sample images.

    Args:
        model: The VL model
        tokenizer: The tokenizer
        model_path: Path to the model (for loading image processor)
        stage_name: Description of the stage (e.g., "before quantization")

    Returns:
        Generated response text for logging/comparison
    """
    try:
        print(f"Loading sample images for {stage_name} preview...")

        # Load sample images from the images directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        images_dir = os.path.join(script_dir, "images")

        # Check if images directory exists
        if not os.path.exists(images_dir):
            print(f"❌ Warning: Images directory not found at {images_dir}")
            print("   VL preview generation requires sample images to test vision capabilities.")
            print("   Skipping VL preview generation.")
            return None

        # Use single image for VL preview to avoid shape mismatch issues
        image_files = ["example1a.jpeg", "example1b.jpeg", "example.jpg", "test.jpg", "sample.png"]
        image = None
        missing_files = []
        for img_file in image_files:
            img_path = os.path.join(images_dir, img_file)
            if os.path.exists(img_path):
                try:
                    image = Image.open(img_path)
                    print(f"  ✅ Successfully loaded: {img_file}")
                    break  # Use the first available image
                except Exception as e:
                    print(f"  ⚠️  Warning: Could not open {img_file}: {e}")
                    missing_files.append(f"{img_file} (corrupted)")
            else:
                missing_files.append(img_file)

        if image is None:
            print(f"❌ Warning: No valid sample images found in {images_dir}")
            print(f"   Searched for: {', '.join(image_files)}")
            if missing_files:
                print(f"   Missing/invalid files: {', '.join(missing_files)}")
            print("   VL preview generation requires sample images to test vision capabilities.")
            print("   Skipping VL preview generation.")
            return None

        # Check if this is Nemotron-Parse early to set up proper generation config
        config = model.config
        architectures = getattr(config, "architectures", [])
        is_nemotron_parse = any("nemotronparse" in arch.lower() for arch in architectures)

        # Generate response
        question = "Describe this image briefly."  # Updated for single image

        # Use model's GenerationConfig for Nemotron-Parse, dict for others
        if is_nemotron_parse:
            try:
                generation_config = GenerationConfig.from_pretrained(
                    model_path, trust_remote_code=True
                )
                print("Using Nemotron-Parse GenerationConfig from model")
            except Exception as e:
                print(f"Warning: Could not load GenerationConfig: {e}, using defaults")
                generation_config = {
                    "max_new_tokens": 50,
                    "do_sample": False,
                    "eos_token_id": tokenizer.eos_token_id,
                }
        else:
            generation_config = {
                "max_new_tokens": 50,
                "do_sample": False,
                "eos_token_id": tokenizer.eos_token_id,
            }

        print(f"Generating VL response ({stage_name})...")

        # Try to detect the VL model has chat method or generate method
        if hasattr(model, "chat"):
            image_processor = AutoImageProcessor.from_pretrained(model_path, trust_remote_code=True)

            image_features = image_processor([image])  # Pass as list with single image

            # Move image features to the same device as the model
            model_device = model.device
            for key, value in image_features.items():
                if hasattr(value, "to"):  # Check if it's a tensor
                    image_features[key] = value.to(model_device)
                    print(f"    Moved {key} to {model_device}")

            response = model.chat(
                tokenizer=tokenizer,
                question=question,
                generation_config=generation_config,
                **image_features,
            )
        else:
            processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

            # Check if this is Nemotron-Parse (uses task prompts instead of chat templates)
            config = model.config
            architectures = getattr(config, "architectures", [])
            is_nemotron_parse = any("nemotronparse" in arch.lower() for arch in architectures)

            if is_nemotron_parse:
                # Nemotron-Parse uses a specific task prompt format
                # See: https://huggingface.co/nvidia/NVIDIA-Nemotron-Parse-v1.1#usage-example
                prompt = "</s><s><predict_bbox><predict_classes><output_markdown>"
                print(f"Using Nemotron-Parse task prompt: {prompt}")
            else:
                # Other VL models use chat templates
                messages = [
                    {"role": "system", "content": "/no_think"},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "image": "",
                            },
                            {
                                "type": "text",
                                "text": question,
                            },
                        ],
                    },
                ]

                # Apply chat template
                prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )

            # Process inputs using the processor with single image
            inputs = processor(
                text=[prompt],
                images=[image],  # Pass single image as list
                return_tensors="pt",
            )

            # Move inputs to the same device as the model
            model_device = model.device
            inputs = inputs.to(model_device)
            print(f"    Moved inputs to {model_device}")

            # Verify we have pixel_values for the vision encoder
            if not hasattr(inputs, 'pixel_values') or inputs.pixel_values is None:
                raise ValueError("Processor did not generate pixel_values. Check processor configuration.")

            # Generate response using model.generate
            if isinstance(generation_config, GenerationConfig):
                # For Nemotron-Parse with GenerationConfig object
                generated_ids = model.generate(
                    pixel_values=inputs.pixel_values,
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    generation_config=generation_config,
                )
            else:
                # For other models with dict generation config
                generated_ids = model.generate(
                    pixel_values=inputs.pixel_values,
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    **generation_config,
                )

            # Decode the response (trim input tokens like in the working example)
            if generated_ids is None:
                raise ValueError("Model generate returned None")

            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]

            # For Nemotron-Parse, use tokenizer.batch_decode instead of processor.batch_decode
            if is_nemotron_parse and hasattr(tokenizer, 'batch_decode'):
                output_text = tokenizer.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
            else:
                output_text = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )

            if output_text is None or len(output_text) == 0:
                raise ValueError("Decoding returned empty output")

            response = output_text[0]

        print(f"✅ VL generation {stage_name} successful!")
        print(f"Question: {question}")
        print(f"Response: {response}")

        # Return the response for comparison/logging
        return response

    except Exception as e:
        print(f"❌ VL preview generation {stage_name} failed: {e}")
        print("This may indicate issues with the quantized model")
        return None


def run_text_only_generation(model, tokenizer, question, generation_config, model_path):
    """Run text-only generation for VL models, supporting both chat and generate methods.

    Args:
        model: The VL model
        tokenizer: The tokenizer
        question: The text question to ask
        generation_config: Generation configuration
        model_path: Path to the model (for loading processor if needed)

    Returns:
        Generated response text or None if failed
    """
    try:
        if hasattr(model, "chat"):
            # Use model.chat with None for images (text-only mode)
            response = model.chat(tokenizer, None, question, generation_config, history=None)
            return response
        else:
            processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

            # Create text-only messages
            messages = [
                {"role": "system", "content": "/no_think"},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": question,
                        },
                    ],
                },
            ]

            # Apply chat template
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            # Process text-only inputs
            inputs = processor(
                text=[prompt],
                images=None,  # No images for text-only
                return_tensors="pt",
            )

            # Move inputs to the same device as the model
            model_device = model.device
            inputs = inputs.to(model_device)

            # Generate response using model.generate
            generated_ids = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                **generation_config,
            )

            # Decode the response (trim input tokens like in the working example)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            return output_text[0]

    except Exception as e:
        print(f"Text-only generation failed: {e}")
        return None
