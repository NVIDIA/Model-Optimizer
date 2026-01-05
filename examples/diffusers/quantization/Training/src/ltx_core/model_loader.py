"""
Model loader for LTX models using in-house implementations.

This module provides functionality to load LTX model components from safetensors checkpoints,
supporting both video-only (LTXV) and audio-video (LTX-2) models.

"""

from __future__ import annotations

import json
from contextlib import suppress
from enum import Enum
from pathlib import Path

import safetensors.torch as st
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.loaders.single_file_utils import load_single_file_checkpoint
from pydantic import BaseModel, ConfigDict
from torch import nn
from transformers import T5EncoderModel, T5Tokenizer

from ltx_core.models.av_model import LTXAVModel
from ltx_core.models.model import LTXVModel
from ltx_core.models.text_encoders.gemma_emb_connector import GemmaConnector
from ltx_core.models.text_encoders.gemma_encoder import GemmaTextEncoder
from ltx_core.models.text_encoders.t5_encoder import T5TextEncoder
from ltx_core.models.vae.audio_vae import AudioVAE
from ltx_core.models.vae.causal_video_autoencoder import VideoVAE
from ltx_core.models.vocoders.vocoder import Vocoder
from ltxv_trainer import logger


class ModelType(str, Enum):
    """Types of LTX models."""

    LTXV = "ltxv"
    LTX_2 = "ltx_2"


class LtxModelComponents(BaseModel):
    """Container for all LTX model components."""

    text_encoder: nn.Module | None = None
    transformer: nn.Module
    video_vae: nn.Module | None
    audio_vae: nn.Module | None = None
    vocoder: nn.Module | None = None
    connector: GemmaConnector | None = None
    scheduler: object | None = None
    model_type: ModelType

    model_config = ConfigDict(arbitrary_types_allowed=True)


def load_model(  # noqa: PLR0912, PLR0915
    checkpoint_path: str | Path,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    with_video_vae: bool = True,
    with_audio_vae: bool = True,
    with_vocoder: bool = True,
    with_text_encoder: bool = True,
    with_connector: bool = True,
    text_encoder_path: str | Path | None = None,
) -> LtxModelComponents:
    """
    Load LTX model components from a safetensors checkpoint.

    This function loads the complete model including transformer, VAEs, and vocoder
    from a single safetensors file. It automatically detects whether the model
    supports audio-video or video-only generation. Supports both local paths and
    HTTP(S) URLs for checkpoint sources.

    Args:
        checkpoint_path: Path to the safetensors checkpoint file (local path or HTTP(S) URL)
        device: Device to load models on ("cuda", "cpu", etc.)
        dtype: Data type for model weights (torch.float32, torch.float16, etc.)
        with_video_vae: Whether to load the video VAE component
        with_audio_vae: Whether to load the audio VAE component (AV models only)
        with_vocoder: Whether to load the vocoder component (AV models only)
        with_text_encoder: Whether to load the text encoder
        with_connector: Whether to load the Gemma connector (LTX-2 models only)
        text_encoder_path: Path to text encoder. For T5 (LTXV models), can be None to load from HF.
                          For Gemma (LTX-2 models), must be path to Gemma model directory.

    Returns:
        LtxModelComponents containing all loaded model components

    Example:
        >>> # Load from local path
        >>> components = load_model(
        ...     "path/to/ltx2_checkpoint.safetensors",
        ...     device="cuda",
        ...     dtype=torch.bfloat16,
        ... )
        >>> # Load from HTTP URL
        >>> components = load_model(
        ...     "https://example.com/models/ltx2_checkpoint.safetensors",
        ...     device="cuda",
        ...     dtype=torch.bfloat16,
        ... )
        >>> # Use the transformer for inference
        >>> output = components.transformer(...)
        >>> # Decode video with VAE
        >>> video = components.video_vae.decode(output)
    """
    checkpoint_path_str = str(checkpoint_path)
    is_url = checkpoint_path_str.startswith(("http://", "https://"))

    if not is_url:
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    logger.info(f"Loading model from {checkpoint_path_str}")

    # Load state dict and metadata
    state_dict = load_single_file_checkpoint(checkpoint_path_str)

    with st.safe_open(checkpoint_path_str, framework="pt") as f:  # TODO: load from URL
        metadata = f.metadata()
    # print(f"{metadata=}")

    # Parse config from metadata
    config = {}
    if metadata and "config" in metadata:
        with suppress(json.JSONDecodeError):
            config = json.loads(metadata["config"])
    # print(json.dumps(config, indent=2))

    # Detect model type
    model_type = _detect_model_type(config, state_dict)
    logger.debug(f"Detected model type: {model_type.value}")

    # Load transformer (always required)
    logger.debug("Loading transformer...")
    transformer = load_transformer(state_dict, config=config, model_type=model_type, device=device, dtype=dtype)

    # Load video VAE
    video_vae = None
    if with_video_vae:
        logger.debug("Loading video VAE...")
        video_vae = load_video_vae(state_dict, config=config, device=device, dtype=dtype)

    # Load audio VAE (only for AV models)
    # Note: Audio VAE uses float32 for quality (bfloat16 causes audio artifacts)
    audio_vae = None
    if with_audio_vae and model_type == ModelType.LTX_2:
        logger.debug("Loading audio VAE...")
        audio_vae = load_audio_vae(state_dict, config=config, device=device, dtype=torch.float32)
    elif with_audio_vae and model_type == ModelType.LTXV:
        logger.debug("Skipping audio VAE (not available for video-only model)")

    # Load vocoder (only for AV models)
    vocoder = None
    if with_vocoder and model_type == ModelType.LTX_2:
        logger.debug("Loading vocoder...")
        vocoder = load_vocoder(state_dict, config=config, device=device, dtype=dtype)
    elif with_vocoder and model_type == ModelType.LTXV:
        logger.debug("Skipping vocoder (not available for video-only model)")

    # Load text encoder (optional)
    text_encoder = None
    if with_text_encoder:
        logger.debug("Loading text encoder...")
        # Determine encoder type based on model type
        if model_type == ModelType.LTX_2:
            # Gemma requires both paths
            if text_encoder_path is None:
                raise ValueError("text_encoder_path must be provided for Gemma encoder (LTX-2 models)")
            text_encoder = load_text_encoder(
                encoder_path=text_encoder_path,
                encoder_type="gemma",
                ltx_checkpoint_path=checkpoint_path,
                device=device,
                dtype=dtype,
            )
        else:
            # T5 can load from HuggingFace or local path
            text_encoder = load_text_encoder(
                encoder_path=text_encoder_path, encoder_type="t5", device=device, dtype=dtype
            )

    # Load connector (only for Gemma models)
    connector = None
    if with_connector and model_type == ModelType.LTX_2 and text_encoder_path is not None:
        logger.debug("Loading Gemma connector...")
        connector = load_gemma_connector(
            checkpoint_path=checkpoint_path,
            device=device,
            dtype=dtype,
        )
    elif with_connector and model_type == ModelType.LTXV:
        logger.debug("Skipping connector (not available for T5 encoder)")

    # Create scheduler (skeleton for now)
    scheduler = create_scheduler(config)

    return LtxModelComponents(
        transformer=transformer,
        video_vae=video_vae,
        audio_vae=audio_vae,
        vocoder=vocoder,
        text_encoder=text_encoder,
        connector=connector,
        scheduler=scheduler,
        model_type=model_type,
    )


def load_text_encoder(
    encoder_path: str | Path | None = None,
    encoder_type: str = "t5",
    ltx_checkpoint_path: str | Path | None = None,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    load_in_8bit: bool = False,
    max_length: int = 256,
) -> nn.Module:
    """
    Load text encoder (T5 or Gemma).

    Args:
        encoder_path: Path to text encoder model directory. For T5, can be None to load from HF.
                     For Gemma, must be path to Gemma model directory.
        encoder_type: Type of encoder ("t5" or "gemma")
        ltx_checkpoint_path: For Gemma only - path to LTX checkpoint containing embeddings connector
        device: Device to load model on
        dtype: Data type for model weights
        load_in_8bit: Whether to load in 8-bit precision (T5 only)
        max_length: Maximum sequence length for tokenization

    Returns:
        Loaded text encoder

    Raises:
        ValueError: If required paths are missing for the encoder type
    """
    if encoder_type == "t5":
        return _load_t5_text_encoder(
            encoder_path=encoder_path,
            device=device,
            dtype=dtype,
            load_in_8bit=load_in_8bit,
            max_length=max_length,
        )
    elif encoder_type == "gemma":
        if encoder_path is None:
            raise ValueError("`encoder_path` must be provided for Gemma encoder")
        if ltx_checkpoint_path is None:
            raise ValueError("`ltx_checkpoint_path` must be provided for Gemma encoder")
        return _load_gemma_text_encoder(
            gemma_model_path=encoder_path,
            device=device,
            dtype=dtype,
            max_length=max_length,
        )
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}. Must be 't5' or 'gemma'")


def load_transformer(
    checkpoint_or_state: str | Path | dict,
    config: dict | None = None,
    model_type: ModelType | None = None,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> nn.Module:
    """
    Load transformer model from checkpoint file or state dict.

    Args:
        checkpoint_or_state: Either a path to checkpoint file or a state dictionary
        config: Model configuration (required if checkpoint_or_state is a dict)
        model_type: Model type (auto-detected if checkpoint_or_state is a path)
        device: Device to load model on
        dtype: Data type for model weights

    Returns:
        Loaded transformer model

    Raises:
        ValueError: If config is missing required fields or arguments are invalid
        FileNotFoundError: If checkpoint file doesn't exist
    """
    # Load from checkpoint or use provided state dict
    if isinstance(checkpoint_or_state, (str, Path)):
        state_dict, config, _ = _load_checkpoint_and_config(checkpoint_or_state)
        model_type = _detect_model_type(config, state_dict)
    else:
        state_dict = checkpoint_or_state
        if config is None:
            raise ValueError("config must be provided when passing a state_dict")
        if model_type is None:
            raise ValueError("model_type must be provided when passing a state_dict")

    # Extract transformer weights and config
    transformer_state = _extract_weights(state_dict, "model.diffusion_model.", "transformer")
    transformer_state = {k: v for k, v in transformer_state.items() if "embeddings_connector" not in k}
    transformer_config = _validate_and_get_config(config, "transformer")

    # Remove metadata fields that aren't constructor arguments
    transformer_config = {k: v for k, v in transformer_config.items() if not k.startswith("_")}

    # Create model based on type
    model_class = LTXAVModel if model_type == ModelType.LTX_2 else LTXVModel

    # Create model on meta device to avoid memory allocation during initialization
    with torch.device("meta"):
        model = model_class(**transformer_config)

    # Load weights
    model.load_state_dict(transformer_state, assign=True)

    return model.to(device=device, dtype=dtype)


def load_video_vae(
    checkpoint_or_state: str | Path | dict,
    config: dict | None = None,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> nn.Module:
    """
    Load video VAE from checkpoint file or state dict.

    Args:
        checkpoint_or_state: Either a path to checkpoint file or a state dictionary
        config: Model configuration (required if checkpoint_or_state is a dict)
        device: Device to load model on
        dtype: Data type for model weights

    Returns:
        Loaded video VAE model

    Raises:
        ValueError: If config is missing required fields or arguments are invalid
        FileNotFoundError: If checkpoint file doesn't exist
    """
    # Load from checkpoint or use provided state dict
    if isinstance(checkpoint_or_state, (str, Path)):
        state_dict, config, _ = _load_checkpoint_and_config(checkpoint_or_state)
    else:
        state_dict = checkpoint_or_state
        if config is None:
            raise ValueError("config must be provided when passing a state_dict")

    # Extract VAE weights and config
    vae_state = _extract_weights(state_dict, "vae.", "video VAE")
    vae_config = _validate_and_get_config(config, "vae")

    # Create model on meta device to avoid memory allocation during initialization
    with torch.device("meta"):
        vae = VideoVAE(config=vae_config)

    # Load weights (this will allocate memory and load the actual weights)
    missing, unexpected = vae.load_state_dict(vae_state, assign=True)

    if missing:
        logger.warning(f"Missing keys in video VAE: {missing}")
    if unexpected:
        logger.warning(f"Unexpected keys in video VAE: {unexpected}")

    return vae.to(device=device, dtype=dtype)


def load_audio_vae(
    checkpoint_or_state: str | Path | dict,
    config: dict | None = None,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.float32,
) -> nn.Module:
    """
    Load audio VAE from checkpoint file or state dict.

    Note: Audio VAE defaults to float32 because audio processing is sensitive
    to numerical precision - bfloat16 causes metallic/robotic artifacts.

    Args:
        checkpoint_or_state: Either a path to checkpoint file or a state dictionary
        config: Model configuration (required if checkpoint_or_state is a dict)
        device: Device to load model on
        dtype: Data type for model weights

    Returns:
        Loaded audio VAE model

    Raises:
        ValueError: If config is missing required fields or arguments are invalid
        FileNotFoundError: If checkpoint file doesn't exist
    """
    # Load from checkpoint or use provided state dict
    if isinstance(checkpoint_or_state, (str, Path)):
        state_dict, config, metadata = _load_checkpoint_and_config(checkpoint_or_state)
    else:
        state_dict = checkpoint_or_state
        if config is None:
            raise ValueError("config must be provided when passing a state_dict")
        # When state_dict is provided directly, metadata should be in config
        metadata = {"config": config}

    # Extract audio VAE weights and metadata
    audio_vae = AudioVAE(state_dict=state_dict, metadata=metadata)

    return audio_vae.to(device=device, dtype=dtype)


def load_vocoder(
    checkpoint_or_state: str | Path | dict,
    config: dict | None = None,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> nn.Module:
    """
    Load vocoder from checkpoint file or state dict.

    Args:
        checkpoint_or_state: Either a path to checkpoint file or a state dictionary
        config: Model configuration (required if checkpoint_or_state is a dict)
        device: Device to load model on
        dtype: Data type for model weights

    Returns:
        Loaded vocoder model

    Raises:
        ValueError: If config is missing required fields or arguments are invalid
        FileNotFoundError: If checkpoint file doesn't exist
    """
    # Load from checkpoint or use provided state dict
    if isinstance(checkpoint_or_state, (str, Path)):
        state_dict, config, _ = _load_checkpoint_and_config(checkpoint_or_state)
    else:
        state_dict = checkpoint_or_state
        if config is None:
            raise ValueError("config must be provided when passing a state_dict")

    # Extract vocoder weights and config
    vocoder_state = _extract_weights(state_dict, "vocoder.", "vocoder")
    vocoder_config = _validate_and_get_config(config, "vocoder")

    # Create model on meta device to avoid memory allocation during initialization
    with torch.device("meta"):
        vocoder = Vocoder(vocoder_config)

    # Load weights (this will allocate memory and load the actual weights)
    vocoder.load_state_dict(vocoder_state, assign=True)

    return vocoder.to(device=device, dtype=dtype)


def create_scheduler(config: dict) -> FlowMatchEulerDiscreteScheduler:  # noqa: ARG001
    """
    Create scheduler for inference and training.

    Args:
        config: Model configuration containing scheduler parameters

    Returns:
        FlowMatchEulerDiscreteScheduler instance
    """

    # Load scheduler from the latest LTXV model
    # This is compatible with both LTXV and LTX-2 models
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        "Lightricks/LTX-Video-0.9.7-dev",
        subfolder="scheduler",
    )
    return scheduler


# Helper functions for loading


def _load_checkpoint_and_config(checkpoint_path: str | Path) -> tuple[dict, dict, dict]:
    """Load state dict, config, and metadata from checkpoint file or URL.

    Returns:
        tuple: (state_dict, config, metadata)
    """
    checkpoint_path_str = str(checkpoint_path)
    is_url = checkpoint_path_str.startswith(("http://", "https://"))

    if not is_url:
        checkpoint_path_obj = Path(checkpoint_path)
        if not checkpoint_path_obj.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path_obj}")

    state_dict = load_single_file_checkpoint(checkpoint_path_str)
    with st.safe_open(checkpoint_path_str, framework="pt") as f:
        metadata = f.metadata()

    if not metadata or "config" not in metadata:
        raise ValueError(f"Checkpoint {checkpoint_path} is missing config metadata")

    config = {}
    with suppress(json.JSONDecodeError):
        config = json.loads(metadata["config"])

    if not config:
        raise ValueError(f"Failed to parse config from checkpoint {checkpoint_path}")

    return state_dict, config, metadata


def _detect_model_type(config: dict, state_dict: dict) -> ModelType:
    """Detect if model is audio-video or video-only."""
    # Check config first
    transformer_config = config.get("transformer", {})
    class_name = transformer_config.get("_class_name", "")

    if class_name == "AVTransformer3DModel":
        return ModelType.LTX_2

    # Fallback: check for audio-specific keys in state dict
    has_audio_keys = any(k.startswith("audio_vae.") for k in state_dict)
    return ModelType.LTX_2 if has_audio_keys else ModelType.LTXV


def _extract_weights(state_dict: dict, prefix: str, component_name: str) -> dict:
    """Extract weights with given prefix from state dict."""
    extracted = {k.replace(prefix, ""): v for k, v in state_dict.items() if k.startswith(prefix)}

    if not extracted:
        raise ValueError(f"No {component_name} weights found in state dict (expected keys with prefix '{prefix}')")

    return extracted


def _validate_and_get_config(config: dict, section: str) -> dict:
    """Validate that config section exists and return it."""
    if section not in config:
        raise ValueError(f"Config is missing '{section}' section")

    return config[section]


def _load_t5_text_encoder(
    encoder_path: str | Path | None,
    device: str | torch.device,
    dtype: torch.dtype,
    load_in_8bit: bool,
    max_length: int,
) -> nn.Module:
    """Load T5 text encoder from HuggingFace or local path."""
    # Default HF repo for T5
    hf_repo = "Lightricks/LTX-Video"

    # Load from HuggingFace if no path provided, otherwise from local path
    source = hf_repo if encoder_path is None else str(encoder_path)

    # Load tokenizer
    tokenizer = T5Tokenizer.from_pretrained(source, subfolder="tokenizer")

    # Load text encoder
    if load_in_8bit:
        try:
            # noinspection PyUnusedImports
            from transformers import BitsAndBytesConfig  # noqa: PLC0415

            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            text_encoder = T5EncoderModel.from_pretrained(
                source, subfolder="text_encoder", quantization_config=quantization_config
            )
        except ImportError:
            logger.warning("`bitsandbytes` not available, loading in full precision")
            text_encoder = T5EncoderModel.from_pretrained(source, subfolder="text_encoder", dtype=dtype)
    else:
        text_encoder = T5EncoderModel.from_pretrained(source, subfolder="text_encoder", dtype=dtype)

    # Create encoder with the proper T5TextEncoder class
    encoder = T5TextEncoder(tokenizer=tokenizer, text_encoder=text_encoder, max_length=max_length)

    # Move to device if not using 8-bit (8-bit models handle device placement automatically)
    if not load_in_8bit:
        encoder.to(device=device)

    return encoder


def _load_gemma_text_encoder(
    gemma_model_path: str | Path,
    device: str | torch.device,
    dtype: torch.dtype,
    max_length: int,
) -> nn.Module:
    """Load Gemma text encoder"""

    print(f"Loading GemmaTextEncoder from {gemma_model_path}")
    encoder = GemmaTextEncoder(
        model_path=gemma_model_path,
        max_length=max_length,
        dtype=dtype,
    )

    encoder.to(device=device)
    return encoder


def load_gemma_connector(
    checkpoint_path: str | Path,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> GemmaConnector:
    """Load Gemma connector (video and audio embeddings connectors) from checkpoint.

    Args:
        checkpoint_path: Path to the safetensors checkpoint file (local path or HTTP(S) URL)
        device: Device to load connectors on
        dtype: Data type for connector weights

    Returns:
        GemmaConnector instance with loaded video and audio connectors
    """
    connector = GemmaConnector(
        checkpoint_path=checkpoint_path,
        dtype=dtype,
    )

    connector.to(device=device)
    return connector
