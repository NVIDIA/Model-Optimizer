#!/usr/bin/env python3

"""
Compute latent representations for video generation training.

This module provides functionality for processing video and image files, including:
- Loading videos/images from various file formats (CSV, JSON, JSONL)
- Resizing, cropping, and transforming media
- MediaDataset for video-only preprocessing workflows
- BucketSampler for grouping videos by resolution

Can be used as a standalone script:
    python scripts/process_videos.py dataset.csv --resolution-buckets 768x768x25 --output-dir /path/to/output
"""

import json  # noqa: I001
import math
from pathlib import Path
from typing import Any


import numpy as np
import pandas as pd
import torch
import typer
from pillow_heif import register_heif_opener
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
import torchaudio
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import crop, resize, to_tensor
from transformers.utils.logging import disable_progress_bar

from ltx_core.model_loader import load_audio_vae, load_video_vae
from ltxv_trainer import logger
from ltxv_trainer.ltxv_utils import encode_video
from ltxv_trainer.utils import open_image_as_srgb

# Should be imported after `torch` to avoid compatibility issues.
import decord  # type: ignore

decord.bridge.set_bridge("torch")
disable_progress_bar()

# Register HEIF/HEIC support
register_heif_opener()

# Constants for validation
VAE_SPATIAL_FACTOR = 32
VAE_TEMPORAL_FACTOR = 8

# Audio constants
AUDIO_LATENT_CHANNELS = 8
AUDIO_FREQUENCY_BINS = 16


app = typer.Typer(
    pretty_exceptions_enable=False,
    no_args_is_help=True,
    help="Process videos/images and save latent representations for video generation training.",
)


class MediaDataset(Dataset):
    """
    Dataset for processing video and image files.

    This dataset is designed for media preprocessing workflows where you need to:
    - Load and preprocess videos/images
    - Apply resizing and cropping transformations
    - Handle different resolution buckets
    - Filter out invalid media files
    - Optionally extract audio from video files
    """

    def __init__(
        self,
        dataset_file: str | Path,
        main_media_column: str,
        video_column: str,
        resolution_buckets: list[tuple[int, int, int]],
        reshape_mode: str = "center",
        with_audio: bool = False,
    ) -> None:
        """
        Initialize the media dataset.

        Args:
            dataset_file: Path to CSV/JSON/JSONL metadata file
            video_column: Column name for video paths in the metadata file
            resolution_buckets: List of (frames, height, width) tuples
            reshape_mode: How to crop videos ("center", "random")
            with_audio: Whether to extract audio from video files
        """
        super().__init__()

        self.dataset_file = Path(dataset_file)
        self.main_media_column = main_media_column
        self.resolution_buckets = resolution_buckets
        self.reshape_mode = reshape_mode
        self.with_audio = with_audio

        # First load main media paths
        self.main_media_paths = self._load_video_paths(main_media_column)

        # Then load reference video paths
        self.video_paths = self._load_video_paths(video_column)

        # Filter out videos with insufficient frames
        self._filter_valid_videos()

        self.max_target_frames = max(self.resolution_buckets, key=lambda x: x[0])[0]

        # Set up video transforms
        self.transforms = transforms.Compose(
            [
                transforms.Lambda(lambda x: x.clamp_(0, 1)),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )

    def __len__(self) -> int:
        return len(self.video_paths)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Get a single video/image with metadata, and optionally audio."""
        if isinstance(index, list):
            # Special case for BucketSampler - return cached data
            return index

        video_path: Path = self.video_paths[index]

        # Compute relative path of the video
        data_root = self.dataset_file.parent
        relative_path = str(video_path.relative_to(data_root))
        media_relative_path = str(self.main_media_paths[index].relative_to(data_root))

        if video_path.suffix.lower() in [".png", ".jpg", ".jpeg"]:
            media_tensor = self._preprocess_image(video_path)
            fps = 1.0
            audio_data = None  # Images don't have audio
        else:
            media_tensor, fps = self._preprocess_video(video_path)

            # Extract audio if enabled
            if self.with_audio:
                # Calculate target duration from the processed video frames
                # This ensures audio is trimmed to match the exact video duration
                target_duration = media_tensor.shape[0] / fps
                audio_data = self._extract_audio(video_path, target_duration)
            else:
                audio_data = None

        result = {
            "video": media_tensor,
            "relative_path": relative_path,
            "main_media_relative_path": media_relative_path,
            "video_metadata": {
                "num_frames": media_tensor.shape[0],
                "height": media_tensor.shape[2],
                "width": media_tensor.shape[3],
                "fps": fps,
            },
        }

        # Add audio data if available
        if audio_data is not None:
            result["audio"] = audio_data

        return result

    @staticmethod
    def _extract_audio(video_path: Path, target_duration: float) -> dict[str, torch.Tensor | int] | None:
        """Extract audio track from a video file, trimmed to match video duration."""
        try:
            # torchaudio can extract audio from video files directly
            # waveform shape: [channels, samples]
            waveform, sample_rate = torchaudio.load(str(video_path))

            # Trim or pad to target duration
            target_samples = int(target_duration * sample_rate)
            current_samples = waveform.shape[-1]

            if current_samples > target_samples:
                # Trim to target duration
                waveform = waveform[..., :target_samples]
            elif current_samples < target_samples:
                # Pad with zeros to target duration
                padding = target_samples - current_samples
                waveform = torch.nn.functional.pad(waveform, (0, padding))
                logger.warning(f"Padded audio to {target_duration:.2f} seconds for {video_path}")

            return {"waveform": waveform, "sample_rate": sample_rate}

        except Exception as e:
            logger.debug(f"Could not extract audio from {video_path}: {e}")
            return None

    def _load_video_paths(self, column: str) -> list[Path]:
        """Load video paths from the specified data source."""
        if self.dataset_file.suffix == ".csv":
            return self._load_video_paths_from_csv(column)
        elif self.dataset_file.suffix == ".json":
            return self._load_video_paths_from_json(column)
        elif self.dataset_file.suffix == ".jsonl":
            return self._load_video_paths_from_jsonl(column)
        else:
            raise ValueError("Expected `dataset_file` to be a path to a CSV, JSON, or JSONL file.")

    def _load_video_paths_from_csv(self, column: str) -> list[Path]:
        """Load video paths from a CSV file."""
        df = pd.read_csv(self.dataset_file)
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in CSV file")

        data_root = self.dataset_file.parent
        video_paths = [data_root / Path(line.strip()) for line in df[column].tolist()]

        # Validate that all paths exist
        invalid_paths = [path for path in video_paths if not path.is_file()]
        if invalid_paths:
            raise ValueError(f"Found {len(invalid_paths)} invalid video paths. First few: {invalid_paths[:5]}")

        return video_paths

    def _load_video_paths_from_json(self, column: str) -> list[Path]:
        """Load video paths from a JSON file."""
        with open(self.dataset_file, "r", encoding="utf-8") as file:
            data = json.load(file)

        if not isinstance(data, list):
            raise ValueError("JSON file must contain a list of objects")

        data_root = self.dataset_file.parent
        video_paths = []
        for entry in data:
            if column not in entry:
                raise ValueError(f"Key '{column}' not found in JSON entry")
            video_paths.append(data_root / Path(entry[column].strip()))

        # Validate that all paths exist
        invalid_paths = [path for path in video_paths if not path.is_file()]
        if invalid_paths:
            raise ValueError(f"Found {len(invalid_paths)} invalid video paths. First few: {invalid_paths[:5]}")

        return video_paths

    def _load_video_paths_from_jsonl(self, column: str) -> list[Path]:
        """Load video paths from a JSONL file."""
        data_root = self.dataset_file.parent
        video_paths = []
        with open(self.dataset_file, "r", encoding="utf-8") as file:
            for line in file:
                entry = json.loads(line)
                if column not in entry:
                    raise ValueError(f"Key '{column}' not found in JSONL entry")
                video_paths.append(data_root / Path(entry[column].strip()))

        # Validate that all paths exist
        invalid_paths = [path for path in video_paths if not path.is_file()]
        if invalid_paths:
            raise ValueError(f"Found {len(invalid_paths)} invalid video paths. First few: {invalid_paths[:5]}")

        return video_paths

    def _filter_valid_videos(self) -> None:
        """Filter out videos with insufficient frames."""
        original_length = len(self.video_paths)
        valid_video_paths = []
        valid_main_media_paths = []
        min_frames_required = min(self.resolution_buckets, key=lambda x: x[0])[0]

        for i, video_path in enumerate(self.video_paths):
            if video_path.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                valid_video_paths.append(video_path)
                valid_main_media_paths.append(self.main_media_paths[i])
                continue

            try:
                video_reader = decord.VideoReader(uri=video_path.as_posix())
                if len(video_reader) >= min_frames_required:
                    valid_video_paths.append(video_path)
                    valid_main_media_paths.append(self.main_media_paths[i])
                else:
                    logger.warning(
                        f"Skipping video at {video_path} - has {len(video_reader)} frames, "
                        f"which is less than the minimum required frames ({min_frames_required})"
                    )
            except Exception as e:
                logger.warning(f"Failed to read video at {video_path}: {e!s}")

        # Update both path lists to maintain synchronization
        self.video_paths = valid_video_paths
        self.main_media_paths = valid_main_media_paths

        if len(self.video_paths) < original_length:
            logger.warning(
                f"Filtered out {original_length - len(self.video_paths)} videos with insufficient frames. "
                f"Proceeding with {len(self.video_paths)} valid videos."
            )

    def _preprocess_image(self, path: Path) -> torch.Tensor:
        """Preprocess a single image by resizing and applying transforms."""
        image = open_image_as_srgb(path)
        image = to_tensor(image)
        image = image.unsqueeze(0)  # Add frame dimension to match video format [1,C,H,W]

        # Find nearest resolution bucket and resize
        nearest_bucket = self._get_resolution_bucket_for_item(image)
        _, target_height, target_width = nearest_bucket
        image_resized = self._resize_and_crop(image, target_height, target_width)

        # Apply transforms and ensure single frame
        image = self.transforms(image_resized)
        image = image.unsqueeze(0)  # Add frame dimension [1,C,H,W]
        return image

    def _preprocess_video(self, path: Path) -> tuple[torch.Tensor, float]:
        """Preprocess a video by loading, resizing, and applying transforms."""
        video_reader = decord.VideoReader(uri=path.as_posix())
        fps = video_reader.get_avg_fps()

        video_num_frames = min(len(video_reader), self.max_target_frames)
        frame_indices = list(range(video_num_frames))
        video = video_reader.get_batch(frame_indices)

        # Ensure video is a PyTorch tensor
        if isinstance(video, decord.ndarray.NDArray):
            video = torch.from_numpy(video.asnumpy())

        # Convert to float and normalize to [0,1]
        video = video.float() / 255.0

        # Permute [T,H,W,C] -> [T,C,H,W]
        video = video.permute(0, 3, 1, 2).contiguous()

        nearest_bucket = self._get_resolution_bucket_for_item(video)
        target_num_frames, target_height, target_width = nearest_bucket
        frames_resized = self._resize_and_crop(video, target_height, target_width)

        # Trim video to target number of frames
        frames_resized = frames_resized[:target_num_frames]

        # Apply transforms to each frame
        video = torch.stack([self.transforms(frame) for frame in frames_resized], dim=0)
        return video, fps

    def _get_resolution_bucket_for_item(self, media_tensor: torch.Tensor) -> tuple[int, int, int]:
        """Get the nearest resolution bucket for the given media tensor."""
        num_frames, _, height, width = media_tensor.shape

        def distance(bucket: tuple[int, int, int]) -> tuple:
            bucket_num_frames, bucket_height, bucket_width = bucket
            # Lexicographic key:
            # 1) minimize aspect-ratio diff (in log-scale, for invariance to shorter/longer ARs)
            # 2) prefer buckets with more frames (by using negative)
            # 3) prefer buckets with larger spatial area (by using negative)
            return (
                abs(math.log(width / height) - math.log(bucket_width / bucket_height)),
                -bucket_num_frames,
                -(bucket_height * bucket_width),
            )

        # Keep only buckets with <= available frames
        relevant_buckets = [b for b in self.resolution_buckets if b[0] <= num_frames]
        if not relevant_buckets:
            raise ValueError(f"No resolution buckets have <= {num_frames} frames. Available: {self.resolution_buckets}")

        # Find the bucket with the minimal distance (according to the function above) to the media item's shape.
        nearest_bucket = min(relevant_buckets, key=distance)

        return nearest_bucket

    def _resize_and_crop(self, media_tensor: torch.Tensor, target_height: int, target_width: int) -> torch.Tensor:
        """Resize and crop tensor to target size."""
        # Get current dimensions
        current_height, current_width = media_tensor.shape[2], media_tensor.shape[3]

        # Calculate aspect ratios to determine which dimension to resize first
        current_aspect = current_width / current_height
        target_aspect = target_width / target_height

        # Resize while maintaining aspect ratio - scale to make the smaller dimension fit
        if current_aspect > target_aspect:
            # Current is wider than target, so scale by height
            new_width = int(current_width * target_height / current_height)
            media_tensor = resize(
                media_tensor,
                size=[target_height, new_width],  # type: ignore
                interpolation=InterpolationMode.BICUBIC,
            )
        else:
            # Current is taller than target, so scale by width
            new_height = int(current_height * target_width / current_width)
            media_tensor = resize(
                media_tensor,
                size=[new_height, target_width],
                interpolation=InterpolationMode.BICUBIC,
            )

        # Update dimensions after resize
        current_height, current_width = media_tensor.shape[2], media_tensor.shape[3]
        media_tensor = media_tensor.squeeze(0)

        # Calculate how much we need to crop from each dimension
        delta_h = current_height - target_height
        delta_w = current_width - target_width

        # Determine crop position based on reshape mode
        if self.reshape_mode == "random":
            # Random crop position
            top = np.random.randint(0, delta_h + 1)
            left = np.random.randint(0, delta_w + 1)
        elif self.reshape_mode == "center":
            # Center crop
            top, left = delta_h // 2, delta_w // 2
        else:
            raise ValueError(f"Unsupported reshape mode: {self.reshape_mode}")

        # Perform the final crop to exact target dimensions
        media_tensor = crop(media_tensor, top=top, left=left, height=target_height, width=target_width)
        return media_tensor


def compute_video_latents(  # noqa: PLR0913, PLR0915
    dataset_file: str | Path,
    video_column: str,
    resolution_buckets: list[tuple[int, int, int]],
    output_dir: str,
    model_source: str,
    main_media_column: str | None = None,
    reshape_mode: str = "center",
    batch_size: int = 1,
    device: str = "cuda",
    vae_tiling: bool = False,
    with_audio: bool = False,
    audio_output_dir: str | None = None,
) -> None:
    """
    Process videos and save latent representations.

    Args:
        dataset_file: Path to metadata file (CSV/JSON/JSONL) containing video paths
        video_column: Column name for video paths in the metadata file
        resolution_buckets: List of (frames, height, width) tuples
        output_dir: Directory to save video latents
        model_source: Model source for VAE (must be LTX-2 checkpoint if with_audio=True)
        reshape_mode: How to crop videos ("center", "random")
        main_media_column: Column name for main media paths (if different from video_column)
        batch_size: Batch size for processing
        device: Device to use for computation
        vae_tiling: Whether to enable VAE tiling
        with_audio: Whether to extract and encode audio from videos
        audio_output_dir: Directory to save audio latents (required if with_audio=True)
    """
    # Validate audio parameters
    if with_audio and audio_output_dir is None:
        raise ValueError("audio_output_dir must be provided when with_audio=True")

    console = Console()

    # Create dataset
    dataset = MediaDataset(
        dataset_file=dataset_file,
        main_media_column=main_media_column or video_column,
        video_column=video_column,
        resolution_buckets=resolution_buckets,
        reshape_mode=reshape_mode,
        with_audio=with_audio,
    )
    logger.info(f"Loaded {len(dataset)} valid media files")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Set up audio output directory if needed
    audio_output_path = None
    if with_audio:
        audio_output_path = Path(audio_output_dir)
        audio_output_path.mkdir(parents=True, exist_ok=True)

    # Load video VAE
    with console.status(f"[bold]Loading video VAE from [cyan]{model_source}[/]...", spinner="dots"):
        vae = load_video_vae(
            checkpoint_or_state=model_source,
            device=device,
            dtype=torch.bfloat16,
        )

    if vae_tiling:
        vae.enable_tiling()

    # Load audio VAE if needed
    audio_vae = None
    if with_audio:
        with console.status(f"[bold]Loading audio VAE from [cyan]{model_source}[/]...", spinner="dots"):
            audio_vae = load_audio_vae(
                checkpoint_or_state=model_source,
                device=device,
                dtype=torch.float32,  # Audio VAE needs float32 for quality
            )

    # Create dataloader
    # Note: batch_size=1 required when with_audio because audio extraction can fail for some videos,
    # and the default collate function can't handle mixed None/dict values across a batch.
    if with_audio and batch_size > 1:
        logger.warning("Audio processing requires batch_size=1. Overriding batch_size to 1.")
        batch_size = 1
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Track audio statistics
    audio_success_count = 0
    audio_skip_count = 0

    # Process batches
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Processing videos", total=len(dataloader))

        for batch in dataloader:
            # Encode videos
            with torch.inference_mode():
                video_latents = encode_video(vae=vae, image_or_video=batch["video"])

            # Save latents for each item in batch
            for i in range(len(batch["relative_path"])):
                output_rel_path = Path(batch["main_media_relative_path"][i]).with_suffix(".pt")
                output_file = output_path / output_rel_path

                # Create output directory maintaining structure
                output_file.parent.mkdir(parents=True, exist_ok=True)

                latent_data = {
                    "latents": video_latents["latents"][i].cpu().contiguous(),
                    "num_frames": video_latents["num_frames"],
                    "height": video_latents["height"],
                    "width": video_latents["width"],
                    "fps": batch["video_metadata"]["fps"][i].item(),
                }

                torch.save(latent_data, output_file)

                # Process audio if enabled (audio is already extracted by the dataset)
                if with_audio:
                    audio_batch = batch.get("audio")
                    if audio_batch is not None:
                        # Extract the i-th item from batched audio data
                        # DataLoader collates [channels, samples] -> [batch, channels, samples]
                        audio_data = {
                            "waveform": audio_batch["waveform"][i],
                            "sample_rate": audio_batch["sample_rate"][i].item(),
                        }

                        # Encode audio
                        audio_latents = None
                        try:
                            with torch.inference_mode():
                                audio_latents = encode_audio(audio_vae, audio_data)
                        except Exception as e:
                            print(f"Error: failed to encode audio for {output_rel_path} {e}")

                        # Save audio latents
                        audio_output_file = audio_output_path / output_rel_path
                        audio_output_file.parent.mkdir(parents=True, exist_ok=True)

                        if audio_latents is not None:
                            audio_latent_data = {
                                "latents": audio_latents["latents"].cpu().contiguous(),
                                "num_time_steps": audio_latents["num_time_steps"],
                                "frequency_bins": audio_latents["frequency_bins"],
                                "duration": audio_latents["duration"],
                            }

                            torch.save(audio_latent_data, audio_output_file)
                            audio_success_count += 1
                        else:
                            audio_skip_count += 1
                    else:
                        # Video has no audio track
                        audio_skip_count += 1

            progress.advance(task)

    # Log summary
    logger.info(f"Processed {len(dataset)} videos. Latents saved to {output_path}")
    if with_audio:
        logger.info(
            f"Audio processing: {audio_success_count} videos with audio, "
            f"{audio_skip_count} videos without audio (skipped)"
        )


def encode_audio(
    audio_vae: torch.nn.Module,
    audio_data: dict[str, torch.Tensor | int],
) -> dict[str, torch.Tensor | int]:
    """Encode audio waveform into latent representation.

    Args:
        audio_vae: AudioVAE model for encoding
        audio_data: Dict with {"waveform": Tensor[channels, samples], "sample_rate": int}

    Returns:
        Dict containing packed audio latents and shape information:
        {
            "latents": Tensor[seq_len, channels],  # Packed format matching video latents
            "num_time_steps": int,
            "frequency_bins": int,
            "duration": float,
        }
    """
    device = next(audio_vae.parameters()).device
    dtype = next(audio_vae.parameters()).dtype

    waveform = audio_data["waveform"].to(device=device, dtype=dtype)
    sample_rate = audio_data["sample_rate"]

    # Add batch dimension if needed: [channels, samples] -> [batch, channels, samples]
    if waveform.dim() == 2:
        waveform = waveform.unsqueeze(0)

    # Calculate duration
    duration = waveform.shape[-1] / sample_rate

    # AudioVAE.encode expects {"waveform": Tensor[batch, channels, samples], "sample_rate": int}
    latents = audio_vae.encode({"waveform": waveform, "sample_rate": sample_rate})

    # latents shape: [batch, channels, time, freq] = [1, 8, T, 16]
    _, channels, time_steps, freq_bins = latents.shape

    # Pack to [seq_len, channels] format (matching video latent format)
    latents_packed = latents.squeeze(0)  # [channels, time, freq]
    latents_packed = latents_packed.permute(1, 2, 0)  # [time, freq, channels]
    latents_packed = latents_packed.reshape(-1, channels)  # [time * freq, channels]

    return {
        "latents": latents_packed,
        "num_time_steps": time_steps,
        "frequency_bins": freq_bins,
        "duration": duration,
    }


def parse_resolution_buckets(resolution_buckets_str: str) -> list[tuple[int, int, int]]:
    """Parse resolution buckets from string format to list of tuples (frames, height, width)"""
    resolution_buckets = []
    for bucket_str in resolution_buckets_str.split(";"):
        w, h, f = map(int, bucket_str.split("x"))

        if w % VAE_SPATIAL_FACTOR != 0 or h % VAE_SPATIAL_FACTOR != 0:
            raise typer.BadParameter(
                f"Width and height must be multiples of {VAE_SPATIAL_FACTOR}, got {w}x{h}",
                param_hint="resolution-buckets",
            )

        if f % VAE_TEMPORAL_FACTOR != 1:
            raise typer.BadParameter(
                f"Number of frames must be a multiple of {VAE_TEMPORAL_FACTOR} plus 1, got {f}",
                param_hint="resolution-buckets",
            )

        resolution_buckets.append((f, h, w))
    return resolution_buckets


@app.command()
def main(
    dataset_file: str = typer.Argument(
        ...,
        help="Path to metadata file (CSV/JSON/JSONL) containing video paths",
    ),
    resolution_buckets: str = typer.Option(
        ...,
        help='Resolution buckets in format "WxHxF;WxHxF;..." (e.g. "768x768x25;512x512x49")',
    ),
    output_dir: str = typer.Option(
        ...,
        help="Output directory to save video latents",
    ),
    video_column: str = typer.Option(
        default="media_path",
        help="Column name in the dataset JSON/JSONL/CSV file containing video paths",
    ),
    batch_size: int = typer.Option(
        default=4,
        help="Batch size for processing",
    ),
    device: str = typer.Option(
        default="cuda",
        help="Device to use for computation",
    ),
    vae_tiling: bool = typer.Option(
        default=False,
        help="Enable VAE tiling for larger video resolutions",
    ),
    model_source: str = typer.Option(
        default="/path/to/.safetensors",  # TODO: set to actual LTX-2 model source
        help="Model source - path to safetensors checkpoint file or HuggingFace repo",
    ),
    reshape_mode: str = typer.Option(
        default="center",
        help="How to crop videos: 'center' or 'random'",
    ),
) -> None:
    """Process videos/images and save latent representations for video generation training.

    This script processes videos and images from metadata files and saves latent representations
    that can be used for training video generation models. The output latents will maintain
    the same folder structure and naming as the corresponding media files.

    Examples:
        # Process videos from a CSV file
        python scripts/process_videos.py dataset.csv --resolution-buckets 768x768x25 --output-dir ./latents

        # Process videos from a JSON file with custom video column
        python scripts/process_videos.py dataset.json
            --resolution-buckets 768x768x25 --output-dir ./latents --video-column "video_path"

        # Enable VAE tiling to save GPU VRAM
        python scripts/process_videos.py dataset.csv
            --resolution-buckets 1024x1024x25 --output-dir ./latents --vae-tiling

        # Use a local checkpoint file
        python scripts/process_videos.py dataset.csv
            --resolution-buckets 768x768x25 --output-dir ./latents
            --model-source /path/to/checkpoint.safetensors
    """

    # Validate dataset file exists
    if not Path(dataset_file).is_file():
        raise typer.BadParameter(f"Dataset file not found: {dataset_file}")

    # Parse resolution buckets
    parsed_resolution_buckets = parse_resolution_buckets(resolution_buckets)

    if len(parsed_resolution_buckets) > 1:
        logger.warning(
            "Using multiple resolution buckets. "
            "When training with multiple resolution buckets, you must use a batch size of 1."
        )

    # Process latents
    compute_video_latents(
        dataset_file=dataset_file,
        video_column=video_column,
        resolution_buckets=parsed_resolution_buckets,
        output_dir=output_dir,
        model_source=model_source,
        reshape_mode=reshape_mode,
        batch_size=batch_size,
        device=device,
        vae_tiling=vae_tiling,
    )


if __name__ == "__main__":
    app()
