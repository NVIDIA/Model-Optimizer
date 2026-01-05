import torch
from torch import Tensor, nn


def pack_latents(unpacked_latents: Tensor) -> Tensor:
    """Convert unpacked latents to packed format.

    This is used to convert model outputs from [B, C, F, H, W] format back to
    the packed [B, seq_len, C] format expected by the training loss computation.

    Args:
        unpacked_latents: Tensor of shape [B, C, F, H, W]

    Returns:
        Packed latents of shape [B, seq_len, C] where seq_len = F*H*W
    """
    B, C, F, H, W = unpacked_latents.shape  # noqa: N806

    # Permute to [B, F, H, W, C]
    packed = unpacked_latents.permute(0, 2, 3, 4, 1)

    # Reshape to [B, F*H*W, C]
    packed = packed.reshape(B, F * H * W, C)

    return packed


def unpack_latents(packed_latents: Tensor, num_frames: int, height: int, width: int) -> Tensor:
    """Convert packed latents to unpacked format.

    This is used to convert training data from [B, seq_len, C] format to the
    [B, C, F, H, W] format expected by ltx_core transformer models.

    Args:
        packed_latents: Tensor of shape [B, seq_len, C] where seq_len = F*H*W
        num_frames: Number of latent frames (F)
        height: Height of latent frames (H)
        width: Width of latent frames (W)

    Returns:
        Unpacked latents of shape [B, C, F, H, W]
    """
    B, seq_len, C = packed_latents.shape  # noqa: N806
    assert seq_len == num_frames * height * width, f"Expected seq_len={num_frames * height * width}, got {seq_len}"

    # Reshape to [B, F, H, W, C]
    unpacked = packed_latents.reshape(B, num_frames, height, width, C)

    # Permute to [B, C, F, H, W]
    unpacked = unpacked.permute(0, 4, 1, 2, 3)

    return unpacked


def encode_video(
    vae: nn.Module,
    image_or_video: Tensor,
    dtype: torch.dtype | None = None,
) -> dict[str, Tensor | int]:
    """Encodes input images/videos into latent representations with the VAE.

    Args:
        vae: VAE model for encoding
        image_or_video: Input tensor of shape [B,C,F,H,W] or [B,C,1,H,W]
        dtype: Target dtype for tensors

    Returns:
        Dict containing latents and shape information
    """
    device = next(vae.parameters()).device

    if image_or_video.ndim == 4:
        image_or_video = image_or_video.unsqueeze(2)
    assert image_or_video.ndim == 5, f"Expected 5D tensor, got {image_or_video.ndim}D tensor"

    # Determine VAE dtype from its parameters
    vae_dtype = next(vae.parameters()).dtype
    image_or_video = image_or_video.to(device=device, dtype=vae_dtype)
    image_or_video = image_or_video.permute(0, 2, 1, 3, 4).contiguous()  # [B, C, F, H, W] -> [B, F, C, H, W]

    # Encode image/video - VAE returns normalized latents directly
    latents = vae.encode(image_or_video)
    latents = latents.to(dtype=dtype) if dtype is not None else latents
    _, _, num_frames, height, width = latents.shape

    # Pack latents to sequence format [B, F*H*W, C]
    latents = pack_latents(latents)
    return {"latents": latents, "num_frames": num_frames, "height": height, "width": width}


def decode_video(
    vae: nn.Module,
    latents: Tensor,
    num_frames: int,
    height: int,
    width: int,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """Decodes latent representations back into videos with the VAE.

    This function reverses the encoding process performed by encode_video().
    It takes the packed latents and shape information and reconstructs the original video.

    Args:
        vae: VAE model for decoding
        latents: Latent tensor as saved by encode_video() [B, seq_len, C]
        num_frames: Number of latents frames in the latent tensor
        height: Height of the latent representation
        width: Width of the latent representation
        dtype: Target dtype for tensors

    Returns:
        Decoded video tensor of shape [B, C, F, H, W]
    """
    device = next(vae.parameters()).device
    vae_dtype = next(vae.parameters()).dtype
    latents = latents.to(device=device, dtype=vae_dtype)

    # Add batch dimension if not present
    if latents.dim() == 2:
        # Latents are [seq_len, C], add batch dimension
        latents = latents.unsqueeze(0)  # -> [1, seq_len, C]
    elif latents.dim() == 1:
        # Latents are [seq_len], add batch and channel dimensions
        latents = latents.unsqueeze(0).unsqueeze(-1)  # -> [1, seq_len, 1]

    # Unpack the latents from [B, seq_len, C] to [B, C, F, H, W]
    latents = unpack_latents(latents, num_frames, height, width)

    # Decode the latents - VAE handles denormalization and noise internally
    # The decode method will:
    # 1. Denormalize using per_channel_statistics
    # 2. Add noise if timestep_conditioning is True (using internal decode_noise_scale)
    # 3. Decode to video
    video = vae.decode(latents)  # Returns [B, C, F, H, W]

    # Permute back to [B, F, C, H, W] to match encode_video input format
    video = video.permute(0, 2, 1, 3, 4).contiguous()  # [B, C, F, H, W] -> [B, F, C, H, W]

    # Normalize video to [0, 1] range
    video = video * 0.5 + 0.5
    video = video.to(dtype=dtype) if dtype is not None else video

    return video


def prepare_video_coordinates(
    num_frames: int,
    height: int,
    width: int,
    batch_size: int,
    device: torch.device | None = None,
    start_end: bool = True,
) -> Tensor:
    """Prepare video coordinates for positional embeddings.

    Args:
        num_frames: Number of frames in latent space
        height: Height in latent space
        width: Width in latent space
        batch_size: Batch size
        device: Target device for tensors
        start_end: If True, return [B, 3, seq_len, 2] format with start/end coords.
                  This matches ltx_core's SymmetricPatchifier format.

    Returns:
        Video coordinates tensor in PIXEL SPACE (not latent space).
        Shape: [B, 3, seq_len, 2] if start_end=True, else [B, 3, seq_len]
        where seq_len = num_frames * height * width.

        The coordinates are scaled by VAE compression factors:
        - Temporal: latent_idx * 8
        - Spatial: latent_idx * 32

        Frame-rate scaling (1/fps) is NOT applied here - the transformer does it internally.
    """
    if device is None:
        device = torch.device("cpu")

    # VAE scale factors (not ROPE scale factors!)
    temporal_scale = 8.0
    spatial_scale = 32.0

    # Create base coordinate tensors in LATENT space
    frame_indices = torch.arange(num_frames, device=device, dtype=torch.float32)
    height_indices = torch.arange(height, device=device, dtype=torch.float32)
    width_indices = torch.arange(width, device=device, dtype=torch.float32)

    # Create meshgrid
    grid_f, grid_h, grid_w = torch.meshgrid(
        frame_indices,
        height_indices,
        width_indices,
        indexing="ij",
    )

    # Flatten to (F*H*W,) for each coordinate
    frame_coords = grid_f.flatten()
    height_coords = grid_h.flatten()
    width_coords = grid_w.flatten()

    # Convert to PIXEL space by applying VAE scale factors
    pixel_f = frame_coords * temporal_scale  # [0, 8, 16, 24, ...]
    pixel_h = height_coords * spatial_scale  # [0, 32, 64, 96, ...]
    pixel_w = width_coords * spatial_scale  # [0, 32, 64, 96, ...]

    # Stack to (3, sequence_length) and expand for batch
    coords_start = torch.stack([pixel_f, pixel_h, pixel_w], dim=0)
    coords_start = coords_start.unsqueeze(0).expand(batch_size, -1, -1)

    if start_end:
        # Create end coordinates (start + patch_size * scale_factor)
        # For patch_size=1: end = start + scale_factor
        coords_end = coords_start.clone()
        coords_end[:, 0, :] += temporal_scale  # +8 for temporal
        coords_end[:, 1, :] += spatial_scale  # +32 for height
        coords_end[:, 2, :] += spatial_scale  # +32 for width

        # Stack to [B, 3, seq_len, 2]
        coords = torch.stack([coords_start, coords_end], dim=-1)
    else:
        coords = coords_start

    return coords
