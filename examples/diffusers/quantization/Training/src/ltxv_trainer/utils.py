import io  # noqa: I001
import subprocess
from fractions import Fraction
from pathlib import Path

import torch
import torchvision.io
from PIL import ExifTags, Image, ImageCms, ImageOps
from PIL.Image import Image as PilImage
from torch import Tensor

from ltxv_trainer import logger


import decord  # Note: Decord must be imported after torch

# Configure decord to use PyTorch tensors
decord.bridge.set_bridge("torch")


def get_gpu_memory_gb(device: torch.device) -> float:
    """Get current GPU memory usage in GB using nvidia-smi"""
    try:
        device_id = device.index if device.index is not None else 0
        result = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,nounits,noheader",
                "-i",
                str(device_id),
            ],
            encoding="utf-8",
        )
        return float(result.strip()) / 1024  # Convert MB to GB
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError) as e:
        logger.error(f"Failed to get GPU memory from nvidia-smi: {e}")
        # Fallback to torch
        return torch.cuda.memory_allocated(device) / 1024**3


def open_image_as_srgb(image_path: str | Path | io.BytesIO) -> PilImage:
    """
    Opens an image file, applies rotation (if it's set in metadata) and converts it
    to the sRGB color space respecting the original image color space .
    """
    exif_colorspace_srgb = 1

    with Image.open(image_path) as img_raw:
        img = ImageOps.exif_transpose(img_raw)

    input_icc_profile = img.info.get("icc_profile")

    # Try to convert to sRGB if the image has ICC profile metadata
    srgb_profile = ImageCms.createProfile(colorSpace="sRGB")
    if input_icc_profile is not None:
        input_profile = ImageCms.ImageCmsProfile(io.BytesIO(input_icc_profile))
        srgb_img = ImageCms.profileToProfile(img, input_profile, srgb_profile, outputMode="RGB")
    else:
        # Try fall back to checking EXIF
        exif_data = img.getexif()
        if exif_data is not None:
            # Assume sRGB if no ICC profile and EXIF has no ColorSpace tag
            color_space_value = exif_data.get(ExifTags.Base.ColorSpace.value)
            if color_space_value is not None and color_space_value != exif_colorspace_srgb:
                raise ValueError(
                    "Image has colorspace tag in EXIF but it isn't set to sRGB,"
                    " conversion is not supported."
                    f" EXIF ColorSpace tag value is {color_space_value}",
                )

        srgb_img = img.convert("RGB")

        # Set sRGB profile in metadata since now the image is assumed to be in sRGB.
        srgb_profile_data = ImageCms.ImageCmsProfile(srgb_profile).tobytes()
        srgb_img.info["icc_profile"] = srgb_profile_data

    return srgb_img


def read_video(video_path: str | Path, target_frames: int | None = None) -> tuple[Tensor, float]:
    """Load and sample frames from a video file.

    Args:
        video_path: Path to the video file
        target_frames: Target number of frames to sample. If None, loads all frames.

    Returns:
        Video tensor with shape [F, C, H, W] in range [0, 1] and frames per second (fps).

    Raises:
        ValueError: If video has fewer frames than target_frames
    """
    # Load video using decord
    video_reader = decord.VideoReader(str(video_path))
    fps = video_reader.get_avg_fps()

    total_frames = len(video_reader)

    if target_frames is None:
        # Load all frames
        indices = list(range(total_frames))
        frames = video_reader.get_batch(indices).float() / 255.0  # [F, H, W, C]
    else:
        # Sample frames uniformly to match target frame count
        if total_frames < target_frames:
            raise ValueError(f"Video has {total_frames} frames, but {target_frames} frames are required")

        # Calculate frame indices to sample
        indices = torch.linspace(0, total_frames - 1, target_frames).long()
        frames = video_reader.get_batch(indices.tolist()).float() / 255.0  # [F, H, W, C]

    frames = frames.permute(0, 3, 1, 2)  # [F, H, W, C] -> [F, C, H, W]

    return frames, fps


def save_video(
    video_tensor: torch.Tensor,
    output_path: Path | str,
    fps: float = 24.0,
    audio: torch.Tensor | None = None,
    audio_sample_rate: int | None = None,
) -> None:
    """Save a video tensor to a file, optionally with audio.

    Args:
        video_tensor: Video tensor of shape [C, F, H, W] or [F, C, H, W] in range [0, 1] or [0, 255]
        output_path: Path to save the video
        fps: Frames per second for the output video
        audio: Optional audio tensor of shape [C, samples] in range [-1, 1]
        audio_sample_rate: Sample rate for the audio (required if audio is provided)
    """
    output_path = Path(output_path)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Handle [C, F, H, W] format from pipeline (C=3 for RGB)
    # vs [F, C, H, W] format (F is typically > 3)
    if video_tensor.shape[0] == 3 and video_tensor.shape[1] > 3:
        # [C, F, H, W] -> [F, C, H, W]
        video_tensor = video_tensor.permute(1, 0, 2, 3)

    # Convert to uint8 and correct format for torchvision.io.write_video
    if video_tensor.max() <= 1:
        video_tensor = video_tensor * 255
    video_tensor = video_tensor.to(torch.uint8)
    video_tensor = video_tensor.permute(0, 2, 3, 1)  # [F, C, H, W] -> [F, H, W, C]

    # Prepare write_video kwargs
    write_kwargs: dict = {
        "video_codec": "h264",
        "options": {"crf": "18"},
    }

    # Add audio if provided
    if audio is not None:
        if audio_sample_rate is None:
            raise ValueError("audio_sample_rate must be provided when audio is given")

        # Ensure audio is float32 and in range [-1, 1]
        audio = audio.float().clamp(-1.0, 1.0).cpu()

        write_kwargs["audio_array"] = audio
        write_kwargs["audio_fps"] = audio_sample_rate
        write_kwargs["audio_codec"] = "aac"

    # Save video
    torchvision.io.write_video(
        str(output_path),
        video_tensor.cpu(),
        fps=Fraction(fps).limit_denominator(1000),
        **write_kwargs,
    )
