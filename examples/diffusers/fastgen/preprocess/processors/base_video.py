# Vendored from NVIDIA-NeMo/Automodel @ e42584e3 (Apache-2.0):
#   https://github.com/NVIDIA-NeMo/Automodel/blob/e42584e303397e9bd34643407b8a57d7def88ce9/tools/diffusion/processors/base_video.py
# Vendored into the fastgen example so preprocessing runs against stock nemo_automodel
# without the un-packaged AutoModel ``tools/`` tree. Self-contained (relative ``.base`` import). Original license below.
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Base class for video model preprocessing.

Extends BaseModelProcessor with video-specific functionality for models like
Wan2.1 and HunyuanVideo.
"""

from abc import abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from .base import BaseModelProcessor


class BaseVideoProcessor(BaseModelProcessor):
    """
    Abstract base class for video model preprocessing.

    Extends BaseModelProcessor with video-specific methods for:
    - Video loading and frame extraction
    - Video VAE encoding
    - Frame count constraints (e.g., 4n+1 for HunyuanVideo)
    - First frame handling for image-to-video models
    """

    @property
    @abstractmethod
    def supported_modes(self) -> List[str]:
        """
        Return supported input modes.

        Returns:
            List of supported modes: 'video' for video files, 'frames' for image sequences
        """
        pass

    @property
    def frame_constraint(self) -> Optional[str]:
        """
        Return frame count constraint.

        Returns:
            Frame constraint string (e.g., '4n+1') or None if no constraint
        """
        return None

    @property
    def quantization(self) -> int:
        """
        VAE quantization requirement.

        Video models typically use 8 due to 3D VAE temporal compression.
        Override in subclasses if different.

        Returns:
            Resolution quantization factor (default 8 for video models)
        """
        return 8

    @abstractmethod
    def encode_video(
        self,
        video_tensor: torch.Tensor,
        models: Dict[str, Any],
        device: str,
        deterministic: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        """
        Encode video tensor to latent space.

        Args:
            video_tensor: Video tensor of shape (1, C, T, H, W), normalized to [-1, 1]
            models: Dict of loaded models from load_models()
            device: Device to use for encoding
            deterministic: If True, use mean instead of sampling from latent distribution
            **kwargs: Additional model-specific arguments

        Returns:
            Latent tensor (shape varies by model, typically (1, C, T', H', W'))
        """
        pass

    @abstractmethod
    def load_video(
        self,
        video_path: str,
        target_size: Tuple[int, int],
        num_frames: Optional[int] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Load video from file and preprocess.

        Args:
            video_path: Path to video file
            target_size: Target (height, width)
            num_frames: Number of frames to extract (None = all frames)
            **kwargs: Additional loading options

        Returns:
            Tuple of:
                - video_tensor: Tensor of shape (1, C, T, H, W), normalized to [-1, 1]
                - first_frame: First frame as numpy array (H, W, C) in uint8 for caching
        """
        pass

    def adjust_frame_count(self, frames: np.ndarray, target_frames: int) -> np.ndarray:
        """
        Adjust frame count to meet model constraints.

        Override in subclasses that have specific frame count requirements
        (e.g., HunyuanVideo requires 4n+1 frames).

        Args:
            frames: Array of frames (T, H, W, C)
            target_frames: Target number of frames

        Returns:
            Adjusted frames array with target_frames frames
        """
        current_frames = len(frames)
        if current_frames == target_frames:
            return frames

        # Default: uniform sampling to reach target frame count
        indices = np.linspace(0, current_frames - 1, target_frames).astype(int)
        return frames[indices]

    def encode_image(
        self,
        image_tensor: torch.Tensor,
        models: Dict[str, Any],
        device: str,
    ) -> torch.Tensor:
        """
        Encode single image by treating it as a 1-frame video.

        Default implementation wraps image as video and delegates to encode_video.

        Args:
            image_tensor: Image tensor of shape (1, C, H, W), normalized to [-1, 1]
            models: Dict of loaded models from load_models()
            device: Device to use for encoding

        Returns:
            Latent tensor
        """
        # Add temporal dimension: (1, C, H, W) -> (1, C, 1, H, W)
        video_tensor = image_tensor.unsqueeze(2)
        return self.encode_video(video_tensor, models, device)

    def verify_latent(
        self,
        latent: torch.Tensor,
        models: Dict[str, Any],
        device: str,
    ) -> bool:
        """
        Verify that a latent can be decoded.

        Default implementation checks for NaN/Inf values.
        Override for model-specific verification.

        Args:
            latent: Encoded latent tensor
            models: Dict of loaded models from load_models()
            device: Device to use for verification

        Returns:
            True if verification passes, False otherwise
        """
        try:
            # Basic sanity checks
            if torch.isnan(latent).any():
                return False
            if torch.isinf(latent).any():
                return False
            return True
        except Exception:
            return False

    def validate_latent_shape(
        self,
        latent: torch.Tensor,
        expected_channels: int,
        spatial_downscale: int = 8,
        temporal_downscale: int = 4,
        input_shape: Optional[Tuple[int, int, int, int, int]] = None,
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate latent tensor shape based on expected dimensions.

        This helper validates that the encoded latent has the expected shape
        given the input dimensions and model-specific downscale factors.

        Args:
            latent: Encoded latent tensor, expected shape (B, C, T', H', W')
            expected_channels: Expected number of latent channels
            spatial_downscale: Spatial downscale factor (default 8)
            temporal_downscale: Temporal downscale factor (default 4)
            input_shape: Optional input shape (B, C, T, H, W) for dimension validation

        Returns:
            Tuple of (is_valid, error_message)
            - is_valid: True if shape is valid
            - error_message: Description of issue if invalid, None if valid
        """
        # Check basic tensor properties
        if latent.ndim != 5:
            return False, f"Expected 5D tensor (B, C, T, H, W), got {latent.ndim}D"

        B, C, T, H, W = latent.shape

        # Check channel count
        if C != expected_channels:
            return False, f"Expected {expected_channels} channels, got {C}"

        # Check for invalid values
        if torch.isnan(latent).any():
            return False, "Latent contains NaN values"
        if torch.isinf(latent).any():
            return False, "Latent contains Inf values"

        # Validate dimensions against input shape if provided
        if input_shape is not None:
            _, _, in_T, in_H, in_W = input_shape
            expected_T = max(1, (in_T + temporal_downscale - 1) // temporal_downscale)
            expected_H = in_H // spatial_downscale
            expected_W = in_W // spatial_downscale

            if T != expected_T:
                return False, f"Expected temporal dim {expected_T}, got {T}"
            if H != expected_H:
                return False, f"Expected height {expected_H}, got {H}"
            if W != expected_W:
                return False, f"Expected width {expected_W}, got {W}"

        return True, None

    def load_video_frames(
        self,
        video_path: str,
        target_size: Tuple[int, int],
        num_frames: Optional[int] = None,
        resize_mode: str = "bilinear",
        center_crop: bool = True,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Load video frames using OpenCV with resizing and optional center crop.

        This is a utility method that can be used by subclass implementations.

        Args:
            video_path: Path to video file
            target_size: Target (height, width)
            num_frames: Number of frames to extract (None = all)
            resize_mode: Interpolation mode for resizing
            center_crop: Whether to center crop to target aspect ratio

        Returns:
            Tuple of:
                - frames: numpy array (T, H, W, C) in uint8
                - info: Dict with video metadata (fps, original_size, etc.)
        """
        import cv2

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Determine which frames to extract
        if num_frames is not None and num_frames < total_frames:
            # Uniform sampling
            frame_indices = np.linspace(0, total_frames - 1, num_frames).astype(int)
        else:
            frame_indices = np.arange(total_frames)

        target_height, target_width = target_size

        # Map resize modes to OpenCV interpolation
        interp_map = {
            "bilinear": cv2.INTER_LINEAR,
            "bicubic": cv2.INTER_CUBIC,
            "nearest": cv2.INTER_NEAREST,
            "area": cv2.INTER_AREA,
            "lanczos": cv2.INTER_LANCZOS4,
        }
        interpolation = interp_map.get(resize_mode, cv2.INTER_LINEAR)

        frames = []
        current_idx = 0

        for target_idx in frame_indices:
            # Seek to frame if needed
            if current_idx != target_idx:
                cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)

            ret, frame = cap.read()
            if not ret:
                break

            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Resize and optionally center crop
            if center_crop:
                # Calculate scale to cover target area
                scale = max(target_width / orig_width, target_height / orig_height)
                new_width = int(orig_width * scale)
                new_height = int(orig_height * scale)

                frame = cv2.resize(frame, (new_width, new_height), interpolation=interpolation)

                # Center crop
                start_x = (new_width - target_width) // 2
                start_y = (new_height - target_height) // 2
                frame = frame[start_y : start_y + target_height, start_x : start_x + target_width]
            else:
                # Direct resize (may change aspect ratio)
                frame = cv2.resize(frame, (target_width, target_height), interpolation=interpolation)

            frames.append(frame)
            current_idx = target_idx + 1

        cap.release()

        frames = np.array(frames, dtype=np.uint8)

        info = {
            "fps": fps,
            "total_frames": total_frames,
            "extracted_frames": len(frames),
            "original_size": (orig_width, orig_height),
            "target_size": (target_width, target_height),
        }

        return frames, info

    def frames_to_tensor(self, frames: np.ndarray) -> torch.Tensor:
        """
        Convert numpy frames array to normalized tensor.

        Args:
            frames: numpy array (T, H, W, C) in uint8

        Returns:
            Tensor of shape (1, C, T, H, W) normalized to [-1, 1]
        """
        # (T, H, W, C) -> (T, C, H, W)
        tensor = torch.from_numpy(frames).float().permute(0, 3, 1, 2)

        # Normalize to [-1, 1]
        tensor = tensor / 255.0
        tensor = (tensor - 0.5) / 0.5

        # Add batch dimension: (T, C, H, W) -> (1, C, T, H, W)
        tensor = tensor.permute(1, 0, 2, 3).unsqueeze(0)

        return tensor
