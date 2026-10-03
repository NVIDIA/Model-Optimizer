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

"""Whisper model configuration utilities for ONNX Runtime GenAI.

This module provides utilities for generating configuration files required
by ONNX Runtime GenAI for Whisper models:

- audio_processor_config.json: Defines the audio preprocessing pipeline
  (AudioDecoder -> STFT -> LogMelSpectrum)
- genai_config.json: Specifies model architecture, I/O tensor names,
  and inference settings for encoder-decoder models
"""

import json
import os
from typing import Any

from ...logging_config import logger

# ---------------------------------------------------------------------------
# Audio processor config
# ---------------------------------------------------------------------------


def generate_audio_processor_config(
    num_mel_bins: int = 128,
    n_fft: int = 400,
    hop_length: int = 160,
    chunk_size: int = 30,
) -> dict:
    """Generate audio processor configuration for Whisper.

    Args:
        num_mel_bins: Number of mel frequency bins.
            - 80 for whisper-tiny/base/small/medium/large/large-v2
            - 128 for whisper-large-v3 and whisper-large-v3-turbo
        n_fft: FFT size (default 400 for 16kHz audio with 25ms window).
        hop_length: Hop length for STFT (default 160 for 10ms hop).
        chunk_size: Audio chunk size in seconds (default 30).

    Returns:
        Audio processor configuration dictionary.
    """
    return {
        "feature_extraction": {
            "sequence": [
                {"operation": {"name": "audio_decoder", "type": "AudioDecoder"}},
                {
                    "operation": {
                        "name": "STFT",
                        "type": "STFTNorm",
                        "attrs": {
                            "n_fft": n_fft,
                            "frame_length": n_fft,
                            "hop_length": hop_length,
                        },
                    }
                },
                {
                    "operation": {
                        "name": "log_mel_spectrogram",
                        "type": "LogMelSpectrum",
                        "attrs": {
                            "chunk_size": chunk_size,
                            "hop_length": hop_length,
                            "n_fft": n_fft,
                            "n_mel": num_mel_bins,
                        },
                    }
                },
            ]
        }
    }


def save_audio_processor_config(
    output_dir: str,
    hf_model_id: str | None = None,
    num_mel_bins: int | None = None,
    overwrite: bool = False,
    trust_remote_code: bool = False,
) -> str:
    """Save audio_processor_config.json to output directory.

    Args:
        output_dir: Directory to save the config file.
        hf_model_id: HuggingFace model ID to extract num_mel_bins from config.
            If provided, num_mel_bins is extracted from the model config.
        num_mel_bins: Number of mel bins. Used if hf_model_id is not provided.
            Default is 128 (for whisper-large-v3 models).
        overwrite: Whether to overwrite existing file.

    Returns:
        Path to the saved config file.
    """
    output_path = os.path.join(output_dir, "audio_processor_config.json")

    # Check if file already exists
    if os.path.exists(output_path) and not overwrite:
        logger.info(f"audio_processor_config.json already exists at {output_dir}")
        return output_path

    # Determine num_mel_bins
    if hf_model_id is not None:
        from transformers import WhisperConfig

        config = WhisperConfig.from_pretrained(hf_model_id, trust_remote_code=trust_remote_code)
        num_mel_bins = config.num_mel_bins
        logger.info(f"Extracted num_mel_bins={num_mel_bins} from {hf_model_id}")
    elif num_mel_bins is None:
        num_mel_bins = 128  # Default for whisper-large-v3
        logger.info(f"Using default num_mel_bins={num_mel_bins}")

    # Generate config
    audio_processor_cfg = generate_audio_processor_config(num_mel_bins=num_mel_bins)

    # Save to file
    os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(audio_processor_cfg, f, indent=4)

    logger.info(f"Saved audio_processor_config.json to {output_dir}")
    return output_path


# ---------------------------------------------------------------------------
# GenAI config
# ---------------------------------------------------------------------------


def generate_genai_config(
    encoder_filename: str,
    decoder_filename: str,
    hf_model_id: str | None = None,
    trust_remote_code: bool = False,
    # Model config (auto-detected from HuggingFace if model_id provided)
    num_decoder_layers: int | None = None,
    num_encoder_layers: int | None = None,
    num_attention_heads: int | None = None,
    hidden_size: int | None = None,
    vocab_size: int | None = None,
    bos_token_id: int | None = None,
    eos_token_id: int | None = None,
    pad_token_id: int | None = None,
    decoder_start_token_id: int | None = None,
    context_length: int | None = None,
    # Provider options
    provider: str = "cuda",
    enable_cuda_graph: bool = False,
    # Search options
    num_beams: int = 1,
    max_length: int | None = None,
    past_present_share_buffer: bool = True,
    # Decoder input/output naming patterns
    decoder_past_key_pattern: str = "past_key_self_%d",
    decoder_past_value_pattern: str = "past_value_self_%d",
    decoder_cross_past_key_pattern: str = "past_key_cross_%d",
    decoder_cross_past_value_pattern: str = "past_value_cross_%d",
    decoder_present_key_pattern: str = "present_key_self_%d",
    decoder_present_value_pattern: str = "present_value_self_%d",
    # Encoder output naming patterns
    encoder_cross_present_key_pattern: str = "present_key_cross_%d",
    encoder_cross_present_value_pattern: str = "present_value_cross_%d",
) -> dict[str, Any]:
    """Generate genai_config.json configuration for Whisper models.

    This config is required by ONNX Runtime GenAI for running encoder-decoder
    models like Whisper. It specifies model architecture, I/O tensor names,
    and inference settings.

    Args:
        encoder_filename: Filename of the encoder ONNX model.
        decoder_filename: Filename of the decoder ONNX model.
        hf_model_id: HuggingFace model ID to auto-detect config parameters.
        num_decoder_layers: Number of decoder layers.
        num_encoder_layers: Number of encoder layers.
        num_attention_heads: Number of attention heads.
        hidden_size: Hidden dimension size.
        vocab_size: Vocabulary size.
        bos_token_id: Beginning of sequence token ID.
        eos_token_id: End of sequence token ID.
        pad_token_id: Padding token ID.
        decoder_start_token_id: Decoder start token ID.
        context_length: Maximum context/sequence length.
        provider: Execution provider ("cuda", "cpu", "NvTensorRtRtx").
        enable_cuda_graph: Whether to enable CUDA graph optimization.
        num_beams: Number of beams for beam search.
        max_length: Maximum generation length.
        past_present_share_buffer: Whether to share KV cache buffer.
        decoder_past_key_pattern: Pattern for decoder past key input names.
        decoder_past_value_pattern: Pattern for decoder past value input names.
        decoder_cross_past_key_pattern: Pattern for cross-attention past key names.
        decoder_cross_past_value_pattern: Pattern for cross-attention past value names.
        decoder_present_key_pattern: Pattern for decoder present key output names.
        decoder_present_value_pattern: Pattern for decoder present value output names.
        encoder_cross_present_key_pattern: Pattern for encoder cross-attention key outputs.
        encoder_cross_present_value_pattern: Pattern for encoder cross-attention value outputs.

    Returns:
        Dictionary containing the complete genai_config.json structure.
    """
    # Load config from HuggingFace if model_id provided
    if hf_model_id is not None:
        from transformers import WhisperConfig

        logger.info(f"Loading config from HuggingFace: {hf_model_id}")
        config = WhisperConfig.from_pretrained(hf_model_id, trust_remote_code=trust_remote_code)

        # Extract values from HF config
        if num_decoder_layers is None:
            num_decoder_layers = config.decoder_layers
        if num_encoder_layers is None:
            num_encoder_layers = config.encoder_layers
        if num_attention_heads is None:
            num_attention_heads = config.decoder_attention_heads
        if hidden_size is None:
            hidden_size = config.d_model
        if vocab_size is None:
            vocab_size = config.vocab_size
        if bos_token_id is None:
            bos_token_id = config.bos_token_id
        if eos_token_id is None:
            eos_token_id = config.eos_token_id
        if pad_token_id is None:
            pad_token_id = config.pad_token_id
        if decoder_start_token_id is None:
            decoder_start_token_id = config.decoder_start_token_id
        if context_length is None:
            context_length = getattr(config, "max_target_positions", None) or config.max_length
        if max_length is None:
            max_length = getattr(config, "max_target_positions", None) or config.max_length

    # Compute head_size
    head_size = hidden_size // num_attention_heads

    # Build provider options (lowercase provider names as expected by GenAI)
    if provider == "cuda":
        provider_options = [{"cuda": {}}]
    elif provider == "cpu":
        provider_options = []
    elif provider == "NvTensorRtRtx":
        provider_options = [
            {"NvTensorRtRtx": {"enable_cuda_graph": "1" if enable_cuda_graph else "0"}}
        ]
    else:
        provider_options = [{provider.lower(): {}}]

    # Build config
    genai_config = {
        "model": {
            "bos_token_id": bos_token_id,
            "context_length": context_length,
            "decoder_start_token_id": decoder_start_token_id,
            "speech": {"config_filename": "audio_processor_config.json"},
            "decoder": {
                "session_options": {
                    "log_id": "onnxruntime-genai",
                    "provider_options": provider_options,
                },
                "filename": decoder_filename,
                "head_size": head_size,
                "hidden_size": hidden_size,
                "inputs": {
                    "input_ids": "input_ids",
                    "past_key_names": decoder_past_key_pattern,
                    "past_value_names": decoder_past_value_pattern,
                    "cross_past_key_names": decoder_cross_past_key_pattern,
                    "cross_past_value_names": decoder_cross_past_value_pattern,
                    "attention_mask": "attention_mask",
                    "position_ids": "position_ids",
                },
                "outputs": {
                    "logits": "logits",
                    "present_key_names": decoder_present_key_pattern,
                    "present_value_names": decoder_present_value_pattern,
                },
                "num_attention_heads": num_attention_heads,
                "num_hidden_layers": num_decoder_layers,
                "num_key_value_heads": num_attention_heads,  # Whisper uses MHA, not GQA
            },
            "encoder": {
                "session_options": {
                    "log_id": "onnxruntime-genai",
                    "provider_options": provider_options,
                },
                "filename": encoder_filename,
                "head_size": head_size,
                "hidden_size": hidden_size,
                "inputs": {"audio_features": "audio_features"},
                "outputs": {
                    "encoder_outputs": "encoder_hidden_states",
                    "cross_present_key_names": encoder_cross_present_key_pattern,
                    "cross_present_value_names": encoder_cross_present_value_pattern,
                },
                "num_attention_heads": num_attention_heads,
                "num_hidden_layers": num_encoder_layers,
                "num_key_value_heads": num_attention_heads,
            },
            "eos_token_id": eos_token_id,
            "pad_token_id": pad_token_id,
            "type": "whisper",
            "vocab_size": vocab_size,
        },
        "search": {
            "diversity_penalty": 0.0,
            "do_sample": False,
            "early_stopping": True,
            "length_penalty": 1.0,
            "max_length": max_length,
            "min_length": 0,
            "no_repeat_ngram_size": 0,
            "num_beams": num_beams,
            "num_return_sequences": 1,
            "past_present_share_buffer": past_present_share_buffer,
            "repetition_penalty": 1.0,
            "temperature": 1.0,
            "top_k": 1,
            "top_p": 1.0,
        },
    }

    return genai_config


def save_genai_config(
    output_dir: str,
    encoder_filename: str,
    decoder_filename: str = "decoder_with_past_model.onnx",
    hf_model_id: str | None = None,
    overwrite: bool = False,
    provider: str = "cuda",
    trust_remote_code: bool = False,
    **kwargs,
) -> str:
    """Save genai_config.json to output directory.

    Args:
        output_dir: Directory to save the config file.
        encoder_filename: Filename of the encoder ONNX model.
        decoder_filename: Filename of the decoder ONNX model.
            Default is "decoder_with_past_model.onnx".
        hf_model_id: HuggingFace model ID for auto-detecting config.
        overwrite: Whether to overwrite existing file.
        provider: Execution provider.
        **kwargs: Additional arguments passed to generate_genai_config.

    Returns:
        Path to the saved config file.
    """
    output_path = os.path.join(output_dir, "genai_config.json")

    # Check if file already exists
    if os.path.exists(output_path) and not overwrite:
        logger.info(f"genai_config.json already exists at {output_dir}")
        return output_path

    # Generate config
    genai_cfg = generate_genai_config(
        encoder_filename=encoder_filename,
        decoder_filename=decoder_filename,
        hf_model_id=hf_model_id,
        trust_remote_code=trust_remote_code,
        provider=provider,
        **kwargs,
    )

    # Save to file
    os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(genai_cfg, f, indent=4)

    logger.info(f"Saved genai_config.json to {output_dir}")
    return output_path


def update_genai_config_encoder(
    config_path: str,
    encoder_filename: str,
    encoder_cross_present_key_pattern: str = "present_key_cross_%d",
    encoder_cross_present_value_pattern: str = "present_value_cross_%d",
) -> dict[str, Any]:
    """Update an existing genai_config.json with encoder settings.

    Useful when you already have a config from ONNX Runtime export and
    want to update the encoder section after running encoder surgery.

    Args:
        config_path: Path to existing genai_config.json.
        encoder_filename: New encoder filename.
        encoder_cross_present_key_pattern: Pattern for cross-attention key outputs.
        encoder_cross_present_value_pattern: Pattern for cross-attention value outputs.

    Returns:
        Updated configuration dictionary.
    """
    with open(config_path) as f:
        config = json.load(f)

    # Update encoder section
    if "model" in config and "encoder" in config["model"]:
        config["model"]["encoder"]["filename"] = encoder_filename
        config["model"]["encoder"]["outputs"]["cross_present_key_names"] = (
            encoder_cross_present_key_pattern
        )
        config["model"]["encoder"]["outputs"]["cross_present_value_names"] = (
            encoder_cross_present_value_pattern
        )

    # Save updated config
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)

    logger.info(f"Updated encoder section in {config_path}")
    return config


def update_genai_config_decoder(
    config_path: str,
    decoder_filename: str,
    decoder_past_key_pattern: str = "past_key_values.%d.decoder.key",
    decoder_past_value_pattern: str = "past_key_values.%d.decoder.value",
    decoder_present_key_pattern: str = "present.%d.decoder.key",
    decoder_present_value_pattern: str = "present.%d.decoder.value",
) -> dict[str, Any]:
    """Update an existing genai_config.json with decoder settings.

    Useful when you've run decoder surgery (MHA/GQA fusion) and need
    to update the decoder section.

    Args:
        config_path: Path to existing genai_config.json.
        decoder_filename: New decoder filename.
        decoder_past_key_pattern: Pattern for past key input names.
        decoder_past_value_pattern: Pattern for past value input names.
        decoder_present_key_pattern: Pattern for present key output names.
        decoder_present_value_pattern: Pattern for present value output names.

    Returns:
        Updated configuration dictionary.
    """
    with open(config_path) as f:
        config = json.load(f)

    # Update decoder section
    if "model" in config and "decoder" in config["model"]:
        config["model"]["decoder"]["filename"] = decoder_filename
        config["model"]["decoder"]["inputs"]["past_key_names"] = decoder_past_key_pattern
        config["model"]["decoder"]["inputs"]["past_value_names"] = decoder_past_value_pattern
        config["model"]["decoder"]["outputs"]["present_key_names"] = decoder_present_key_pattern
        config["model"]["decoder"]["outputs"]["present_value_names"] = decoder_present_value_pattern

    # Save updated config
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)

    logger.info(f"Updated decoder section in {config_path}")
    return config
