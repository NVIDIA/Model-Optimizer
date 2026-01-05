import math
from functools import partial
from typing import List, Optional, Tuple, Union

import torch
from einops import rearrange
from torch import nn

from ltx_core.models.ops.conv_nd_factory import make_conv_nd, make_linear_nd
from ltx_core.models.ops.pixel_norm import PixelNorm

from ..model import PixArtAlphaCombinedTimestepSizeEmbeddings


class Encoder(nn.Module):
    r"""
    The `Encoder` layer of a variational autoencoder that encodes its input into a latent representation.

    Args:
        dims (`int` or `Tuple[int, int]`, *optional*, defaults to 3):
            The number of dimensions to use in convolutions.
        in_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        out_channels (`int`, *optional*, defaults to 3):
            The number of output channels.
        blocks (`List[Tuple[str, int]]`, *optional*, defaults to `[("res_x", 1)]`):
            The blocks to use. Each block is a tuple of the block name and the number of layers.
        base_channels (`int`, *optional*, defaults to 128):
            The number of output channels for the first convolutional layer.
        norm_num_groups (`int`, *optional*, defaults to 32):
            The number of groups for normalization.
        patch_size (`int`, *optional*, defaults to 1):
            The patch size to use. Should be a power of 2.
        norm_layer (`str`, *optional*, defaults to `group_norm`):
            The normalization layer to use. Can be either `group_norm` or `pixel_norm`.
        latent_log_var (`str`, *optional*, defaults to `per_channel`):
            The number of channels for the log variance. Can be either `per_channel`, `uniform`, `constant` or `none`.
    """

    def __init__(
        self,
        dims: Union[int, Tuple[int, int]] = 3,
        in_channels: int = 3,
        out_channels: int = 3,
        blocks: List[Tuple[str, int | dict]] = [("res_x", 1)],
        base_channels: int = 128,
        norm_num_groups: int = 32,
        patch_size: Union[int, Tuple[int]] = 1,
        norm_layer: str = "group_norm",  # group_norm, pixel_norm
        latent_log_var: str = "per_channel",
        spatial_padding_mode: str = "zeros",
    ):
        super().__init__()
        self.patch_size = patch_size
        self.norm_layer = norm_layer
        self.latent_channels = out_channels
        self.latent_log_var = latent_log_var
        self.blocks_desc = blocks

        in_channels = in_channels * patch_size**2
        output_channel = base_channels

        self.conv_in = make_conv_nd(
            dims=dims,
            in_channels=in_channels,
            out_channels=output_channel,
            kernel_size=3,
            stride=1,
            padding=1,
            causal=True,
            spatial_padding_mode=spatial_padding_mode,
        )

        self.down_blocks = nn.ModuleList([])

        for block_name, block_params in blocks:
            input_channel = output_channel
            if isinstance(block_params, int):
                block_params = {"num_layers": block_params}

            if block_name == "res_x":
                block = UNetMidBlock3D(
                    dims=dims,
                    in_channels=input_channel,
                    num_layers=block_params["num_layers"],
                    resnet_eps=1e-6,
                    resnet_groups=norm_num_groups,
                    norm_layer=norm_layer,
                    spatial_padding_mode=spatial_padding_mode,
                )
            elif block_name == "res_x_y":
                output_channel = block_params.get("multiplier", 2) * output_channel
                block = ResnetBlock3D(
                    dims=dims,
                    in_channels=input_channel,
                    out_channels=output_channel,
                    eps=1e-6,
                    groups=norm_num_groups,
                    norm_layer=norm_layer,
                    spatial_padding_mode=spatial_padding_mode,
                )
            elif block_name == "compress_time":
                block = make_conv_nd(
                    dims=dims,
                    in_channels=input_channel,
                    out_channels=output_channel,
                    kernel_size=3,
                    stride=(2, 1, 1),
                    causal=True,
                    spatial_padding_mode=spatial_padding_mode,
                )
            elif block_name == "compress_space":
                block = make_conv_nd(
                    dims=dims,
                    in_channels=input_channel,
                    out_channels=output_channel,
                    kernel_size=3,
                    stride=(1, 2, 2),
                    causal=True,
                    spatial_padding_mode=spatial_padding_mode,
                )
            elif block_name == "compress_all":
                block = make_conv_nd(
                    dims=dims,
                    in_channels=input_channel,
                    out_channels=output_channel,
                    kernel_size=3,
                    stride=(2, 2, 2),
                    causal=True,
                    spatial_padding_mode=spatial_padding_mode,
                )
            elif block_name == "compress_all_x_y":
                output_channel = block_params.get("multiplier", 2) * output_channel
                block = make_conv_nd(
                    dims=dims,
                    in_channels=input_channel,
                    out_channels=output_channel,
                    kernel_size=3,
                    stride=(2, 2, 2),
                    causal=True,
                    spatial_padding_mode=spatial_padding_mode,
                )
            elif block_name == "compress_all_res":
                output_channel = block_params.get("multiplier", 2) * output_channel
                block = SpaceToDepthDownsample(
                    dims=dims,
                    in_channels=input_channel,
                    out_channels=output_channel,
                    stride=(2, 2, 2),
                    spatial_padding_mode=spatial_padding_mode,
                )
            elif block_name == "compress_space_res":
                output_channel = block_params.get("multiplier", 2) * output_channel
                block = SpaceToDepthDownsample(
                    dims=dims,
                    in_channels=input_channel,
                    out_channels=output_channel,
                    stride=(1, 2, 2),
                    spatial_padding_mode=spatial_padding_mode,
                )
            elif block_name == "compress_time_res":
                output_channel = block_params.get("multiplier", 2) * output_channel
                block = SpaceToDepthDownsample(
                    dims=dims,
                    in_channels=input_channel,
                    out_channels=output_channel,
                    stride=(2, 1, 1),
                    spatial_padding_mode=spatial_padding_mode,
                )
            else:
                raise ValueError(f"unknown block: {block_name}")

            self.down_blocks.append(block)

        # out
        if norm_layer == "group_norm":
            self.conv_norm_out = nn.GroupNorm(num_channels=output_channel, num_groups=norm_num_groups, eps=1e-6)
        elif norm_layer == "pixel_norm":
            self.conv_norm_out = PixelNorm()
        elif norm_layer == "layer_norm":
            self.conv_norm_out = LayerNorm(output_channel, eps=1e-6)

        self.conv_act = nn.SiLU()

        conv_out_channels = out_channels
        if latent_log_var == "per_channel":
            conv_out_channels *= 2
        elif latent_log_var == "uniform" or latent_log_var == "constant":
            conv_out_channels += 1
        elif latent_log_var != "none":
            raise ValueError(f"Invalid latent_log_var: {latent_log_var}")
        self.conv_out = make_conv_nd(
            dims,
            output_channel,
            conv_out_channels,
            3,
            padding=1,
            causal=True,
            spatial_padding_mode=spatial_padding_mode,
        )

        self.gradient_checkpointing = False

    def forward(self, sample: torch.FloatTensor) -> torch.FloatTensor:
        r"""The forward method of the `Encoder` class."""

        sample = patchify(sample, patch_size_hw=self.patch_size, patch_size_t=1)
        sample = self.conv_in(sample)

        checkpoint_fn = (
            partial(torch.utils.checkpoint.checkpoint, use_reentrant=False)
            if self.gradient_checkpointing and self.training
            else lambda x: x
        )

        for down_block in self.down_blocks:
            sample = checkpoint_fn(down_block)(sample)

        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        if self.latent_log_var == "uniform":
            last_channel = sample[:, -1:, ...]
            num_dims = sample.dim()

            if num_dims == 4:
                # For shape (B, C, H, W)
                repeated_last_channel = last_channel.repeat(1, sample.shape[1] - 2, 1, 1)
                sample = torch.cat([sample, repeated_last_channel], dim=1)
            elif num_dims == 5:
                # For shape (B, C, F, H, W)
                repeated_last_channel = last_channel.repeat(1, sample.shape[1] - 2, 1, 1, 1)
                sample = torch.cat([sample, repeated_last_channel], dim=1)
            else:
                raise ValueError(f"Invalid input shape: {sample.shape}")
        elif self.latent_log_var == "constant":
            sample = sample[:, :-1, ...]
            approx_ln_0 = -30  # this is the minimal clamp value in DiagonalGaussianDistribution objects
            sample = torch.cat(
                [sample, torch.ones_like(sample, device=sample.device) * approx_ln_0],
                dim=1,
            )

        return sample

class Decoder(nn.Module):
    r"""
    The `Decoder` layer of a variational autoencoder that decodes its latent representation into an output sample.

    Args:
        dims (`int` or `Tuple[int, int]`, *optional*, defaults to 3):
            The number of dimensions to use in convolutions.
        in_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        out_channels (`int`, *optional*, defaults to 3):
            The number of output channels.
        blocks (`List[Tuple[str, int]]`, *optional*, defaults to `[("res_x", 1)]`):
            The blocks to use. Each block is a tuple of the block name and the number of layers.
        base_channels (`int`, *optional*, defaults to 128):
            The number of output channels for the first convolutional layer.
        norm_num_groups (`int`, *optional*, defaults to 32):
            The number of groups for normalization.
        patch_size (`int`, *optional*, defaults to 1):
            The patch size to use. Should be a power of 2.
        norm_layer (`str`, *optional*, defaults to `group_norm`):
            The normalization layer to use. Can be either `group_norm` or `pixel_norm`.
        causal (`bool`, *optional*, defaults to `True`):
            Whether to use causal convolutions or not.
    """

    def __init__(
        self,
        dims,
        in_channels: int = 3,
        out_channels: int = 3,
        blocks: List[Tuple[str, int | dict]] = [("res_x", 1)],
        base_channels: int = 128,
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
        patch_size: int = 1,
        norm_layer: str = "group_norm",
        causal: bool = True,
        timestep_conditioning: bool = False,
        spatial_padding_mode: str = "zeros",
    ):
        super().__init__()
        self.patch_size = patch_size
        self.layers_per_block = layers_per_block
        out_channels = out_channels * patch_size**2
        self.causal = causal
        self.blocks_desc = blocks

        # Compute output channel to be product of all channel-multiplier blocks
        output_channel = base_channels
        for block_name, block_params in list(reversed(blocks)):
            block_params = block_params if isinstance(block_params, dict) else {}
            if block_name == "res_x_y":
                output_channel = output_channel * block_params.get("multiplier", 2)
            if block_name == "compress_all":
                output_channel = output_channel * block_params.get("multiplier", 1)

        self.conv_in = make_conv_nd(
            dims,
            in_channels,
            output_channel,
            kernel_size=3,
            stride=1,
            padding=1,
            causal=True,
            spatial_padding_mode=spatial_padding_mode,
        )

        self.up_blocks = nn.ModuleList([])

        for block_name, block_params in list(reversed(blocks)):
            input_channel = output_channel
            if isinstance(block_params, int):
                block_params = {"num_layers": block_params}

            if block_name == "res_x":
                block = UNetMidBlock3D(
                    dims=dims,
                    in_channels=input_channel,
                    num_layers=block_params["num_layers"],
                    resnet_eps=1e-6,
                    resnet_groups=norm_num_groups,
                    norm_layer=norm_layer,
                    inject_noise=block_params.get("inject_noise", False),
                    timestep_conditioning=timestep_conditioning,
                    spatial_padding_mode=spatial_padding_mode,
                )
            elif block_name == "attn_res_x":
                block = UNetMidBlock3D(
                    dims=dims,
                    in_channels=input_channel,
                    num_layers=block_params["num_layers"],
                    resnet_groups=norm_num_groups,
                    norm_layer=norm_layer,
                    inject_noise=block_params.get("inject_noise", False),
                    timestep_conditioning=timestep_conditioning,
                    attention_head_dim=block_params["attention_head_dim"],
                    spatial_padding_mode=spatial_padding_mode,
                )
            elif block_name == "res_x_y":
                output_channel = output_channel // block_params.get("multiplier", 2)
                block = ResnetBlock3D(
                    dims=dims,
                    in_channels=input_channel,
                    out_channels=output_channel,
                    eps=1e-6,
                    groups=norm_num_groups,
                    norm_layer=norm_layer,
                    inject_noise=block_params.get("inject_noise", False),
                    timestep_conditioning=False,
                    spatial_padding_mode=spatial_padding_mode,
                )
            elif block_name == "compress_time":
                block = DepthToSpaceUpsample(
                    dims=dims,
                    in_channels=input_channel,
                    stride=(2, 1, 1),
                    spatial_padding_mode=spatial_padding_mode,
                )
            elif block_name == "compress_space":
                block = DepthToSpaceUpsample(
                    dims=dims,
                    in_channels=input_channel,
                    stride=(1, 2, 2),
                    spatial_padding_mode=spatial_padding_mode,
                )
            elif block_name == "compress_all":
                output_channel = output_channel // block_params.get("multiplier", 1)
                block = DepthToSpaceUpsample(
                    dims=dims,
                    in_channels=input_channel,
                    stride=(2, 2, 2),
                    residual=block_params.get("residual", False),
                    out_channels_reduction_factor=block_params.get("multiplier", 1),
                    spatial_padding_mode=spatial_padding_mode,
                )
            else:
                raise ValueError(f"unknown layer: {block_name}")

            self.up_blocks.append(block)

        if norm_layer == "group_norm":
            self.conv_norm_out = nn.GroupNorm(num_channels=output_channel, num_groups=norm_num_groups, eps=1e-6)
        elif norm_layer == "pixel_norm":
            self.conv_norm_out = PixelNorm()
        elif norm_layer == "layer_norm":
            self.conv_norm_out = LayerNorm(output_channel, eps=1e-6)

        self.conv_act = nn.SiLU()
        self.conv_out = make_conv_nd(
            dims,
            output_channel,
            out_channels,
            3,
            padding=1,
            causal=True,
            spatial_padding_mode=spatial_padding_mode,
        )

        self.gradient_checkpointing = False

        self.timestep_conditioning = timestep_conditioning

        if timestep_conditioning:
            self.timestep_scale_multiplier = nn.Parameter(torch.tensor(1000.0, dtype=torch.float32))
            self.last_time_embedder = PixArtAlphaCombinedTimestepSizeEmbeddings(
                output_channel * 2,
                0,
            )
            self.last_scale_shift_table = nn.Parameter(torch.empty(2, output_channel))

    # def forward(self, sample: torch.FloatTensor, target_shape) -> torch.FloatTensor:
    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        r"""The forward method of the `Decoder` class."""
        batch_size = sample.shape[0]

        sample = self.conv_in(sample, causal=self.causal)

        checkpoint_fn = (
            partial(torch.utils.checkpoint.checkpoint, use_reentrant=False)
            if self.gradient_checkpointing and self.training
            else lambda x: x
        )

        scaled_timestep = None
        if self.timestep_conditioning:
            assert timestep is not None, "should pass timestep with timestep_conditioning=True"
            scaled_timestep = timestep * self.timestep_scale_multiplier.to(dtype=sample.dtype, device=sample.device)

        for up_block in self.up_blocks:
            if self.timestep_conditioning and isinstance(up_block, UNetMidBlock3D):
                sample = checkpoint_fn(up_block)(sample, causal=self.causal, timestep=scaled_timestep)
            else:
                sample = checkpoint_fn(up_block)(sample, causal=self.causal)

        sample = self.conv_norm_out(sample)

        if self.timestep_conditioning:
            embedded_timestep = self.last_time_embedder(
                timestep=scaled_timestep.flatten(),
                resolution=None,
                aspect_ratio=None,
                batch_size=sample.shape[0],
                hidden_dtype=sample.dtype,
            )
            embedded_timestep = embedded_timestep.view(batch_size, embedded_timestep.shape[-1], 1, 1, 1)
            ada_values = self.last_scale_shift_table[None, ..., None, None, None].to(
                device=sample.device, dtype=sample.dtype
            ) + embedded_timestep.reshape(
                batch_size,
                2,
                -1,
                embedded_timestep.shape[-3],
                embedded_timestep.shape[-2],
                embedded_timestep.shape[-1],
            )
            shift, scale = ada_values.unbind(dim=1)
            sample = sample * (1 + scale) + shift

        sample = self.conv_act(sample)
        sample = self.conv_out(sample, causal=self.causal)

        sample = unpatchify(sample, patch_size_hw=self.patch_size, patch_size_t=1)

        return sample


class UNetMidBlock3D(nn.Module):
    """
    A 3D UNet mid-block [`UNetMidBlock3D`] with multiple residual blocks.

    Args:
        in_channels (`int`): The number of input channels.
        dropout (`float`, *optional*, defaults to 0.0): The dropout rate.
        num_layers (`int`, *optional*, defaults to 1): The number of residual blocks.
        resnet_eps (`float`, *optional*, 1e-6 ): The epsilon value for the resnet blocks.
        resnet_groups (`int`, *optional*, defaults to 32):
            The number of groups to use in the group normalization layers of the resnet blocks.
        norm_layer (`str`, *optional*, defaults to `group_norm`):
            The normalization layer to use. Can be either `group_norm` or `pixel_norm`.
        inject_noise (`bool`, *optional*, defaults to `False`):
            Whether to inject noise into the hidden states.
        timestep_conditioning (`bool`, *optional*, defaults to `False`):
            Whether to condition the hidden states on the timestep.

    Returns:
        `torch.FloatTensor`: The output of the last residual block, which is a tensor of shape `(batch_size,
        in_channels, height, width)`.

    """

    def __init__(
        self,
        dims: Union[int, Tuple[int, int]],
        in_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_groups: int = 32,
        norm_layer: str = "group_norm",
        inject_noise: bool = False,
        timestep_conditioning: bool = False,
        spatial_padding_mode: str = "zeros",
    ):
        super().__init__()
        resnet_groups = resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)

        self.timestep_conditioning = timestep_conditioning

        if timestep_conditioning:
            self.time_embedder = PixArtAlphaCombinedTimestepSizeEmbeddings(
                in_channels * 4,
                0,
            )

        self.res_blocks = nn.ModuleList(
            [
                ResnetBlock3D(
                    dims=dims,
                    in_channels=in_channels,
                    out_channels=in_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    norm_layer=norm_layer,
                    inject_noise=inject_noise,
                    timestep_conditioning=timestep_conditioning,
                    spatial_padding_mode=spatial_padding_mode,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        causal: bool = True,
        timestep: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        timestep_embed = None
        if self.timestep_conditioning:
            assert timestep is not None, "should pass timestep with timestep_conditioning=True"
            batch_size = hidden_states.shape[0]
            timestep_embed = self.time_embedder(
                timestep=timestep.flatten(),
                resolution=None,
                aspect_ratio=None,
                batch_size=batch_size,
                hidden_dtype=hidden_states.dtype,
            )
            timestep_embed = timestep_embed.view(batch_size, timestep_embed.shape[-1], 1, 1, 1)

        for resnet in self.res_blocks:
            hidden_states = resnet(hidden_states, causal=causal, timestep=timestep_embed)

        return hidden_states


class SpaceToDepthDownsample(nn.Module):
    def __init__(
        self,
        dims: int | tuple[int, int],
        in_channels: int,
        out_channels: int,
        stride: tuple[int, int, int],
        spatial_padding_mode: str,
    ):
        super().__init__()
        self.stride = stride
        self.group_size = in_channels * math.prod(stride) // out_channels
        self.conv = make_conv_nd(
            dims=dims,
            in_channels=in_channels,
            out_channels=out_channels // math.prod(stride),
            kernel_size=3,
            stride=1,
            causal=True,
            spatial_padding_mode=spatial_padding_mode,
        )

    def forward(self, x: torch.Tensor, causal: bool = True) -> torch.Tensor:
        if self.stride[0] == 2:
            x = torch.cat([x[:, :, :1, :, :], x], dim=2)  # duplicate first frames for padding

        # skip connection
        x_in = rearrange(
            x,
            "b c (d p1) (h p2) (w p3) -> b (c p1 p2 p3) d h w",
            p1=self.stride[0],
            p2=self.stride[1],
            p3=self.stride[2],
        )
        x_in = rearrange(x_in, "b (c g) d h w -> b c g d h w", g=self.group_size)
        x_in = x_in.mean(dim=2)

        # conv
        x = self.conv(x, causal=causal)
        x = rearrange(
            x,
            "b c (d p1) (h p2) (w p3) -> b (c p1 p2 p3) d h w",
            p1=self.stride[0],
            p2=self.stride[1],
            p3=self.stride[2],
        )

        x = x + x_in

        return x


class DepthToSpaceUpsample(nn.Module):
    def __init__(
        self,
        dims: int | tuple[int, int],
        in_channels: int,
        stride: tuple[int, int, int],
        residual: bool = False,
        out_channels_reduction_factor: int = 1,
        spatial_padding_mode: str = "zeros",
    ):
        super().__init__()
        self.stride = stride
        self.out_channels = math.prod(stride) * in_channels // out_channels_reduction_factor
        self.conv = make_conv_nd(
            dims=dims,
            in_channels=in_channels,
            out_channels=self.out_channels,
            kernel_size=3,
            stride=1,
            causal=True,
            spatial_padding_mode=spatial_padding_mode,
        )
        self.residual = residual
        self.out_channels_reduction_factor = out_channels_reduction_factor

    def forward(self, x: torch.Tensor, causal: bool = True, _timestep: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.residual:
            # Reshape and duplicate the input to match the output shape
            x_in = rearrange(
                x,
                "b (c p1 p2 p3) d h w -> b c (d p1) (h p2) (w p3)",
                p1=self.stride[0],
                p2=self.stride[1],
                p3=self.stride[2],
            )
            num_repeat = math.prod(self.stride) // self.out_channels_reduction_factor
            x_in = x_in.repeat(1, num_repeat, 1, 1, 1)
            if self.stride[0] == 2:
                x_in = x_in[:, :, 1:, :, :]
        x = self.conv(x, causal=causal)
        x = rearrange(
            x,
            "b (c p1 p2 p3) d h w -> b c (d p1) (h p2) (w p3)",
            p1=self.stride[0],
            p2=self.stride[1],
            p3=self.stride[2],
        )
        if self.stride[0] == 2:
            x = x[:, :, 1:, :, :]
        if self.residual:
            x = x + x_in
        return x


class LayerNorm(nn.Module):
    def __init__(self, dim: int, eps: float, elementwise_affine: bool = True) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=elementwise_affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = rearrange(x, "b c d h w -> b d h w c")
        x = self.norm(x)
        x = rearrange(x, "b d h w c -> b c d h w")
        return x


class ResnetBlock3D(nn.Module):
    r"""
    A Resnet block.

    Parameters:
        in_channels (`int`): The number of channels in the input.
        out_channels (`int`, *optional*, default to be `None`):
            The number of output channels for the first conv layer. If None, same as `in_channels`.
        dropout (`float`, *optional*, defaults to `0.0`): The dropout probability to use.
        groups (`int`, *optional*, default to `32`): The number of groups to use for the first normalization layer.
        eps (`float`, *optional*, defaults to `1e-6`): The epsilon to use for the normalization.
    """

    def __init__(
        self,
        dims: Union[int, Tuple[int, int]],
        in_channels: int,
        out_channels: Optional[int] = None,
        dropout: float = 0.0,
        groups: int = 32,
        eps: float = 1e-6,
        norm_layer: str = "group_norm",
        inject_noise: bool = False,
        timestep_conditioning: bool = False,
        spatial_padding_mode: str = "zeros",
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.inject_noise = inject_noise

        if norm_layer == "group_norm":
            self.norm1 = nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)
        elif norm_layer == "pixel_norm":
            self.norm1 = PixelNorm()
        elif norm_layer == "layer_norm":
            self.norm1 = LayerNorm(in_channels, eps=eps, elementwise_affine=True)

        self.non_linearity = nn.SiLU()

        self.conv1 = make_conv_nd(
            dims,
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            causal=True,
            spatial_padding_mode=spatial_padding_mode,
        )

        if inject_noise:
            self.per_channel_scale1 = nn.Parameter(torch.zeros((in_channels, 1, 1)))

        if norm_layer == "group_norm":
            self.norm2 = nn.GroupNorm(num_groups=groups, num_channels=out_channels, eps=eps, affine=True)
        elif norm_layer == "pixel_norm":
            self.norm2 = PixelNorm()
        elif norm_layer == "layer_norm":
            self.norm2 = LayerNorm(out_channels, eps=eps, elementwise_affine=True)

        self.dropout = torch.nn.Dropout(dropout)

        self.conv2 = make_conv_nd(
            dims,
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            causal=True,
            spatial_padding_mode=spatial_padding_mode,
        )

        if inject_noise:
            self.per_channel_scale2 = nn.Parameter(torch.zeros((in_channels, 1, 1)))

        self.conv_shortcut = (
            make_linear_nd(dims=dims, in_channels=in_channels, out_channels=out_channels)
            if in_channels != out_channels
            else nn.Identity()
        )

        self.norm3 = (
            LayerNorm(in_channels, eps=eps, elementwise_affine=True) if in_channels != out_channels else nn.Identity()
        )

        self.timestep_conditioning = timestep_conditioning

        if timestep_conditioning:
            self.scale_shift_table = nn.Parameter(torch.randn(4, in_channels) / in_channels**0.5)

    def _feed_spatial_noise(
        self, hidden_states: torch.FloatTensor, per_channel_scale: torch.FloatTensor
    ) -> torch.FloatTensor:
        spatial_shape = hidden_states.shape[-2:]
        device = hidden_states.device
        dtype = hidden_states.dtype

        # similar to the "explicit noise inputs" method in style-gan
        spatial_noise = torch.randn(spatial_shape, device=device, dtype=dtype)[None]
        scaled_noise = (spatial_noise * per_channel_scale)[None, :, None, ...]
        hidden_states = hidden_states + scaled_noise

        return hidden_states

    def forward(
        self,
        input_tensor: torch.FloatTensor,
        causal: bool = True,
        timestep: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        hidden_states = input_tensor
        batch_size = hidden_states.shape[0]

        hidden_states = self.norm1(hidden_states)
        if self.timestep_conditioning:
            assert timestep is not None, "should pass timestep with timestep_conditioning=True"
            ada_values = self.scale_shift_table[None, ..., None, None, None].to(
                device=hidden_states.device, dtype=hidden_states.dtype
            ) + timestep.reshape(
                batch_size,
                4,
                -1,
                timestep.shape[-3],
                timestep.shape[-2],
                timestep.shape[-1],
            )
            shift1, scale1, shift2, scale2 = ada_values.unbind(dim=1)

            hidden_states = hidden_states * (1 + scale1) + shift1

        hidden_states = self.non_linearity(hidden_states)

        hidden_states = self.conv1(hidden_states, causal=causal)

        if self.inject_noise:
            hidden_states = self._feed_spatial_noise(
                hidden_states,
                self.per_channel_scale1.to(device=hidden_states.device, dtype=hidden_states.dtype),
            )

        hidden_states = self.norm2(hidden_states)

        if self.timestep_conditioning:
            hidden_states = hidden_states * (1 + scale2) + shift2

        hidden_states = self.non_linearity(hidden_states)

        hidden_states = self.dropout(hidden_states)

        hidden_states = self.conv2(hidden_states, causal=causal)

        if self.inject_noise:
            hidden_states = self._feed_spatial_noise(
                hidden_states,
                self.per_channel_scale2.to(device=hidden_states.device, dtype=hidden_states.dtype),
            )

        input_tensor = self.norm3(input_tensor)

        batch_size = input_tensor.shape[0]

        input_tensor = self.conv_shortcut(input_tensor)

        output_tensor = input_tensor + hidden_states

        return output_tensor


def patchify(x: torch.Tensor, patch_size_hw: int, patch_size_t: int = 1) -> torch.Tensor:
    if patch_size_hw == 1 and patch_size_t == 1:
        return x
    if x.dim() == 4:
        x = rearrange(x, "b c (h q) (w r) -> b (c r q) h w", q=patch_size_hw, r=patch_size_hw)
    elif x.dim() == 5:
        x = rearrange(
            x,
            "b c (f p) (h q) (w r) -> b (c p r q) f h w",
            p=patch_size_t,
            q=patch_size_hw,
            r=patch_size_hw,
        )
    else:
        raise ValueError(f"Invalid input shape: {x.shape}")

    return x


def unpatchify(x: torch.Tensor, patch_size_hw: int, patch_size_t: int = 1) -> torch.Tensor:
    if patch_size_hw == 1 and patch_size_t == 1:
        return x

    if x.dim() == 4:
        x = rearrange(x, "b (c r q) h w -> b c (h q) (w r)", q=patch_size_hw, r=patch_size_hw)
    elif x.dim() == 5:
        x = rearrange(
            x,
            "b (c p r q) f h w -> b c (f p) (h q) (w r)",
            p=patch_size_t,
            q=patch_size_hw,
            r=patch_size_hw,
        )

    return x


class Processor(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("std-of-means", torch.empty(128))
        self.register_buffer("mean-of-means", torch.empty(128))
        self.register_buffer("mean-of-stds", torch.empty(128))
        self.register_buffer("mean-of-stds_over_std-of-means", torch.empty(128))
        self.register_buffer("channel", torch.empty(128))

    def un_normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x * self.get_buffer("std-of-means").view(1, -1, 1, 1, 1).to(x)) + self.get_buffer("mean-of-means").view(
            1, -1, 1, 1, 1
        ).to(x)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.get_buffer("mean-of-means").view(1, -1, 1, 1, 1).to(x)) / self.get_buffer("std-of-means").view(
            1, -1, 1, 1, 1
        ).to(x)


class VideoVAE(nn.Module):
    def __init__(self, version: int = 0, config: dict | None = None):
        super().__init__()

        if config is None:
            config = self.guess_config(version)

        self.timestep_conditioning = config.get("timestep_conditioning", False)
        self.decode_noise_scale = config.get("decode_noise_scale", 0.025)
        self.decode_timestep = config.get("decode_timestep", 0.05)
        double_z = config.get("double_z", True)
        latent_log_var = config.get("latent_log_var", "per_channel" if double_z else "none")

        self.encoder = Encoder(
            dims=config["dims"],
            in_channels=config.get("in_channels", 3),
            out_channels=config["latent_channels"],
            blocks=config.get("encoder_blocks", config.get("encoder_blocks", config.get("blocks"))),
            patch_size=config.get("patch_size", 1),
            latent_log_var=latent_log_var,
            norm_layer=config.get("norm_layer", "group_norm"),
            spatial_padding_mode=config.get("spatial_padding_mode", "zeros"),
            base_channels=config.get("encoder_base_channels", 128),
        )

        self.decoder = Decoder(
            dims=config["dims"],
            in_channels=config["latent_channels"],
            out_channels=config.get("out_channels", 3),
            blocks=config.get("decoder_blocks", config.get("decoder_blocks", config.get("blocks"))),
            base_channels=config.get("decoder_base_channels", 128),
            patch_size=config.get("patch_size", 1),
            norm_layer=config.get("norm_layer", "group_norm"),
            causal=config.get("causal_decoder", False),
            timestep_conditioning=self.timestep_conditioning,
            spatial_padding_mode=config.get("spatial_padding_mode", "reflect"),
        )

        self.per_channel_statistics = Processor()

    def guess_config(self, version: int) -> dict:
        if version == 0:
            config = {
                "_class_name": "CausalVideoAutoencoder",
                "dims": 3,
                "in_channels": 3,
                "out_channels": 3,
                "latent_channels": 128,
                "blocks": [
                    ["res_x", 4],
                    ["compress_all", 1],
                    ["res_x_y", 1],
                    ["res_x", 3],
                    ["compress_all", 1],
                    ["res_x_y", 1],
                    ["res_x", 3],
                    ["compress_all", 1],
                    ["res_x", 3],
                    ["res_x", 4],
                ],
                "scaling_factor": 1.0,
                "norm_layer": "pixel_norm",
                "patch_size": 4,
                "latent_log_var": "uniform",
                "use_quant_conv": False,
                "causal_decoder": False,
            }
        elif version == 1:
            config = {
                "_class_name": "CausalVideoAutoencoder",
                "dims": 3,
                "in_channels": 3,
                "out_channels": 3,
                "latent_channels": 128,
                "decoder_blocks": [
                    ["res_x", {"num_layers": 5, "inject_noise": True}],
                    ["compress_all", {"residual": True, "multiplier": 2}],
                    ["res_x", {"num_layers": 6, "inject_noise": True}],
                    ["compress_all", {"residual": True, "multiplier": 2}],
                    ["res_x", {"num_layers": 7, "inject_noise": True}],
                    ["compress_all", {"residual": True, "multiplier": 2}],
                    ["res_x", {"num_layers": 8, "inject_noise": False}],
                ],
                "encoder_blocks": [
                    ["res_x", {"num_layers": 4}],
                    ["compress_all", {}],
                    ["res_x_y", 1],
                    ["res_x", {"num_layers": 3}],
                    ["compress_all", {}],
                    ["res_x_y", 1],
                    ["res_x", {"num_layers": 3}],
                    ["compress_all", {}],
                    ["res_x", {"num_layers": 3}],
                    ["res_x", {"num_layers": 4}],
                ],
                "scaling_factor": 1.0,
                "norm_layer": "pixel_norm",
                "patch_size": 4,
                "latent_log_var": "uniform",
                "use_quant_conv": False,
                "causal_decoder": False,
                "timestep_conditioning": True,
            }
        else:
            config = {
                "_class_name": "CausalVideoAutoencoder",
                "dims": 3,
                "in_channels": 3,
                "out_channels": 3,
                "latent_channels": 128,
                "encoder_blocks": [
                    ["res_x", {"num_layers": 4}],
                    ["compress_space_res", {"multiplier": 2}],
                    ["res_x", {"num_layers": 6}],
                    ["compress_time_res", {"multiplier": 2}],
                    ["res_x", {"num_layers": 6}],
                    ["compress_all_res", {"multiplier": 2}],
                    ["res_x", {"num_layers": 2}],
                    ["compress_all_res", {"multiplier": 2}],
                    ["res_x", {"num_layers": 2}],
                ],
                "decoder_blocks": [
                    ["res_x", {"num_layers": 5, "inject_noise": False}],
                    ["compress_all", {"residual": True, "multiplier": 2}],
                    ["res_x", {"num_layers": 5, "inject_noise": False}],
                    ["compress_all", {"residual": True, "multiplier": 2}],
                    ["res_x", {"num_layers": 5, "inject_noise": False}],
                    ["compress_all", {"residual": True, "multiplier": 2}],
                    ["res_x", {"num_layers": 5, "inject_noise": False}],
                ],
                "scaling_factor": 1.0,
                "norm_layer": "pixel_norm",
                "patch_size": 4,
                "latent_log_var": "uniform",
                "use_quant_conv": False,
                "causal_decoder": False,
                "timestep_conditioning": True,
            }
        return config

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        frames_count = x.shape[2]
        if ((frames_count - 1) % 8) != 0:
            raise ValueError(
                "Invalid number of frames: Encode input must have 1 + 8 * x frames "
                "(e.g., 1, 9, 17, ...). Please check your input."
            )
        means, logvar = torch.chunk(self.encoder(x), 2, dim=1)
        return self.per_channel_statistics.normalize(means)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        if self.timestep_conditioning:  # TODO: seed
            x = torch.randn_like(x) * self.decode_noise_scale + (1.0 - self.decode_noise_scale) * x
        return self.decoder(self.per_channel_statistics.un_normalize(x), timestep=self.decode_timestep)

    def tiled_encode(
        self,
        x: torch.Tensor,
        tiling_threshold: int = 512 * 512 * 33,  # Default: 16M tokens
        spatial_overlap: int = 2,
        temporal_overlap: int = 2,
    ) -> torch.Tensor:
        """
        Encode videos using tiled encoding for large videos.
        
        Args:
            x: Video tensor [B, C, F, H, W] where C=3 (RGB)
            tiling_threshold: Maximum total input tokens (W*H*F) per tile
            spatial_overlap: Overlap between spatial tiles in input pixels
            temporal_overlap: Overlap between temporal tiles in input frames
            
        Returns:
            Encoded latent tensor [B, C_latent, F_latent, H_latent, W_latent]
        """
        batch, channels, frames, height, width = x.shape
        
        # Validate frame count (same as regular encode)
        if ((frames - 1) % 8) != 0:
            raise ValueError(
                "Invalid number of frames: Encode input must have 1 + 8 * x frames "
                "(e.g., 1, 9, 17, ...). Please check your input."
            )
        
        # Calculate total input tokens
        total_tokens = width * height * frames
        
        # Fast path: if below threshold, use regular encode
        if total_tokens <= tiling_threshold:
            return self.encode(x)
        
        # Calculate tokens per spatial tile with 2x2 split
        # After 2x2 spatial split, each tile has (W/2) * (H/2) * F tokens
        return self._encode_spatial_tiled(x, spatial_overlap, tiling_threshold)

    def _encode_spatial_tiled(
        self,
        x: torch.Tensor,
        spatial_overlap: int = 1,
        tiling_threshold: int = 512*512*33,
    ) -> torch.Tensor:
        """
        Encode videos using spatial tiling, similar to ComfyUI's tiled_scale_multidim.
        
        Args:
            x: Video tensor [B, C, F, H, W] where C=3 (RGB)
            spatial_overlap: Overlap between spatial tiles in input pixels
            
        Returns:
            Encoded latent tensor [B, C_latent, F_latent, H_latent, W_latent]
        """
        import itertools
        
        batch, channels, frames, height, width = x.shape
        
        # VAE compression ratios (downscaling)
        spatial_scale = 32
        temporal_scale = 8
        
        # Calculate output dimensions
        output_frames = 1 + (frames - 1) // temporal_scale
        output_height = height // spatial_scale
        output_width = width // spatial_scale
        
        # Encoder outputs latent channels (128)
        output_channels = self.encoder.latent_channels
        
        # Check if input dimensions are valid for tiling
        if height < spatial_scale or width < spatial_scale:
            raise ValueError(
                f"Input dimensions too small for tiling: {height}x{width}. "
                f"Minimum size: {spatial_scale}x{spatial_scale}"
            )
        
        # Get encoder patch size (required for patchify operation)
        patch_size = self.encoder.patch_size
        
        # Calculate tile sizes based on tiling_threshold
        # Each tile should have at most tiling_threshold tokens: W_tile * H_tile * F <= threshold
        # So: W_tile * H_tile <= threshold / F
        max_spatial_tokens = tiling_threshold // frames
        # Calculate tile dimensions: try to make roughly square
        # W_tile * H_tile <= max_spatial_tokens
        # For square tiles: tile_size^2 <= max_spatial_tokens
        tile_size = int(math.sqrt(max_spatial_tokens))
        # Ensure divisible by spatial_scale
        tile_h = (tile_size // spatial_scale) * spatial_scale
        tile_w = (tile_size // spatial_scale) * spatial_scale
        # Ensure at least spatial_scale
        tile_h = max(tile_h, spatial_scale)
        tile_w = max(tile_w, spatial_scale)

        
        # Ensure overlap is at least 1 and divisible by spatial_scale for proper blending
        if spatial_overlap < spatial_scale:
            spatial_overlap = spatial_scale
        
        # Initialize output tensors
        device = x.device
        dtype = x.dtype
        output = torch.zeros(
            (batch, output_channels, output_frames, output_height, output_width),
            device=device,
            dtype=dtype,
        )
        weights = torch.zeros(
            (batch, 1, output_frames, output_height, output_width),
            device=device,
            dtype=dtype,
        )
        
        # Process each batch item
        for b in range(batch):
            s = x[b:b+1]  # [1, C, F, H, W]
            
            # If entire input fits in a single tile, encode directly
            if height <= tile_h and width <= tile_w:
                encoded = self.encoder(s)
                means, logvar = torch.chunk(encoded, 2, dim=1)
                encoded_tile = self.per_channel_statistics.normalize(means)
                output[b:b+1] = encoded_tile
                continue
            
            # Calculate tile positions with overlap
            # Positions: range(0, dim - overlap, tile - overlap)
            h_positions = (
                list(range(0, height - spatial_overlap, tile_h - spatial_overlap))
                if height > tile_h
                else [0]
            )
            w_positions = (
                list(range(0, width - spatial_overlap, tile_w - spatial_overlap))
                if width > tile_w
                else [0]
            )
            
            # Ensure last tile covers the end
            if h_positions[-1] + tile_h < height:
                h_positions.append(height - tile_h)
            if w_positions[-1] + tile_w < width:
                w_positions.append(width - tile_w)
            
            # Process each tile
            for h_pos, w_pos in itertools.product(h_positions, w_positions):
                # Calculate tile boundaries
                h_start = max(0, min(height - spatial_overlap, h_pos))
                w_start = max(0, min(width - spatial_overlap, w_pos))
                h_end = min(h_start + tile_h, height)
                w_end = min(w_start + tile_w, width)
                
                # Extract tile
                tile = s[:, :, :, h_start:h_end, w_start:w_end]
                
                # Ensure tile dimensions are divisible by patch_size
                tile_h_actual = h_end - h_start
                tile_w_actual = w_end - w_start
                tile_h_actual = (tile_h_actual // patch_size) * patch_size
                tile_w_actual = (tile_w_actual // patch_size) * patch_size
                
                # Also ensure divisible by spatial_scale
                tile_h_actual = (tile_h_actual // spatial_scale) * spatial_scale
                tile_w_actual = (tile_w_actual // spatial_scale) * spatial_scale
                
                # Adjust if needed
                if tile_h_actual < tile_h:
                    h_end = h_start + tile_h_actual
                    tile = tile[:, :, :, :tile_h_actual, :]
                if tile_w_actual < tile_w:
                    w_end = w_start + tile_w_actual
                    tile = tile[:, :, :, :, :tile_w_actual]
                
                # Skip if too small
                if tile.shape[3] < patch_size or tile.shape[4] < patch_size:
                    continue
                
                # Encode the tile
                encoder_output = self.encoder(tile)
                means, logvar = torch.chunk(encoder_output, 2, dim=1)
                encoded_tile = self.per_channel_statistics.normalize(means)
                
                # Calculate output positions (downscaled)
                out_h_start = h_start // spatial_scale
                out_w_start = w_start // spatial_scale
                
                # Get actual encoded tile dimensions
                _, _, tile_out_frames, tile_out_height, tile_out_width = encoded_tile.shape
                
                # Calculate output end positions
                out_h_end = min(out_h_start + tile_out_height, output_height)
                out_w_end = min(out_w_start + tile_out_width, output_width)
                
                # Adjust tile if needed
                if out_h_end - out_h_start < tile_out_height:
                    encoded_tile = encoded_tile[:, :, :, :(out_h_end - out_h_start), :]
                    tile_out_height = out_h_end - out_h_start
                if out_w_end - out_w_start < tile_out_width:
                    encoded_tile = encoded_tile[:, :, :, :, :(out_w_end - out_w_start)]
                    tile_out_width = out_w_end - out_w_start
                
                # Create mask for feathering (blending at edges)
                mask = torch.ones(
                    (1, 1, output_frames, tile_out_height, tile_out_width),
                    device=encoded_tile.device,
                    dtype=encoded_tile.dtype,
                )
                
                # Calculate overlap in output space
                overlap_out_h = max(1, spatial_overlap // spatial_scale)
                overlap_out_w = max(1, spatial_overlap // spatial_scale)
                
                # Apply feathering at edges (similar to ComfyUI)
                # Left edge
                if w_start > 0 and overlap_out_w > 0 and overlap_out_w < tile_out_width:
                    for t in range(overlap_out_w):
                        a = (t + 1) / overlap_out_w
                        mask[:, :, :, :, t] *= a
                
                # Right edge
                if w_end < width and overlap_out_w > 0 and overlap_out_w < tile_out_width:
                    for t in range(overlap_out_w):
                        a = (t + 1) / overlap_out_w
                        mask[:, :, :, :, tile_out_width - 1 - t] *= a
                
                # Top edge
                if h_start > 0 and overlap_out_h > 0 and overlap_out_h < tile_out_height:
                    for t in range(overlap_out_h):
                        a = (t + 1) / overlap_out_h
                        mask[:, :, :, t, :] *= a
                
                # Bottom edge
                if h_end < height and overlap_out_h > 0 and overlap_out_h < tile_out_height:
                    for t in range(overlap_out_h):
                        a = (t + 1) / overlap_out_h
                        mask[:, :, :, tile_out_height - 1 - t, :] *= a
                
                # Accumulate weighted tile
                output[b:b+1, :, :, out_h_start:out_h_end, out_w_start:out_w_end] += (
                    encoded_tile * mask
                )
                weights[b:b+1, :, :, out_h_start:out_h_end, out_w_start:out_w_end] += mask
        
        # Normalize by weights
        output = output / (weights + 1e-8)
        
        return output

    


    def tiled_decode(
        self,
        x: torch.Tensor,
        tiling_threshold: int =10000,  # Default: 10K tokens
        spatial_overlap: int = 2,
        temporal_overlap: int = 1,
        vertical_tiles: int = 2,
        horizontal_tiles: int = 2,
    ) -> torch.Tensor:
        """
        Decode latents using tiled decoding for large videos.
        
        Args:
            x: Latent tensor [B, C, F, H, W]
            tiling_threshold: Maximum total latent tokens (W*H*F) per tile
            spatial_overlap: Overlap between spatial tiles in latent pixels
            temporal_overlap: Overlap between temporal tiles in latent frames
            
        Returns:
            Decoded video tensor [B, C, F_out, H_out, W_out]
        """
        batch, channels, frames, height, width = x.shape
        total_tokens = width * height * frames
        
        # Fast path: if below threshold, use regular decode
        if total_tokens <= tiling_threshold:
            return self.decode(x)
        
        # Calculate tokens per spatial tile with 2x2 split
        spatial_tile_tokens = (width // horizontal_tiles) * (height // vertical_tiles) * frames
        
        # If spatial tiles are still too large, use temporal tiling too
        if spatial_tile_tokens > tiling_threshold:
            return self._decode_spatial_temporal_tiled(
                x, tiling_threshold, spatial_overlap, temporal_overlap, horizontal_tiles, vertical_tiles
            )
        else:
            # Use spatial-only tiling (2x2)
            return self._decode_spatial_tiled(x, spatial_overlap)

    def _decode_spatial_tiled(
        self,
        x: torch.Tensor,
        spatial_overlap: int = 1,
        vertical_tiles: int = 2,
        horizontal_tiles: int = 2,
    ) -> torch.Tensor:
        """
        Decode latents using 2x2 spatial tiling only.
        
        Args:
            x: Latent tensor [B, C, F, H, W]
            spatial_overlap: Overlap between spatial tiles in latent pixels
            
        Returns:
            Decoded video tensor [B, C, F_out, H_out, W_out]
        """
        batch, channels, frames, height, width = x.shape
        
        # VAE compression ratios
        spatial_scale = 32
        temporal_scale = 8
        
        # Calculate output dimensions
        output_frames = 1 + (frames - 1) * temporal_scale
        output_height = height * spatial_scale
        output_width = width * spatial_scale
        
        # Decoder outputs RGB (3 channels), not latent channels
        output_channels = 3
        
        

        
        # Calculate tile sizes with overlap (2x2 grid)
  

        base_tile_height = (height + (vertical_tiles - 1) * spatial_overlap) // vertical_tiles
        base_tile_width = (width + (horizontal_tiles - 1) * spatial_overlap) // horizontal_tiles
        
        # Initialize output tensor and weight tensor
        # Note: output has 3 channels (RGB), not latent channels
        device = x.device
        dtype = x.dtype
        output = torch.zeros(
            (batch, output_channels, output_frames, output_height, output_width),
            device=device,
            dtype=dtype,
        )
        weights = torch.zeros(
            (batch, 1, output_frames, output_height, output_width),
            device=device,
            dtype=dtype,
        )
        
        # Process each spatial tile (2x2 grid)
        for v in range(vertical_tiles):
            for h in range(horizontal_tiles):
                # Calculate tile boundaries
                h_start = h * (base_tile_width - spatial_overlap)
                v_start = v * (base_tile_height - spatial_overlap)
                
                # Adjust end positions for edge tiles
                h_end = (
                    min(h_start + base_tile_width, width)
                    if h < horizontal_tiles - 1
                    else width
                )
                v_end = (
                    min(v_start + base_tile_height, height)
                    if v < vertical_tiles - 1
                    else height
                )
                
                # Extract tile
                tile = x[:, :, :, v_start:v_end, h_start:h_end]
                
                # Decode the tile (apply timestep conditioning and un_normalize)
                if self.timestep_conditioning:
                    tile = torch.randn_like(tile) * self.decode_noise_scale + (1.0 - self.decode_noise_scale) * tile
                tile = self.per_channel_statistics.un_normalize(tile)
                decoded_tile = self.decoder(tile, timestep=self.decode_timestep)
                
                # Calculate output tile boundaries (use int64 to avoid 32-bit overflow)
                out_h_start = int(v_start * spatial_scale)
                out_h_end = int(v_end * spatial_scale)
                out_w_start = int(h_start * spatial_scale)
                out_w_end = int(h_end * spatial_scale)
                
                # Create weight mask for this tile
                tile_out_height = out_h_end - out_h_start
                tile_out_width = out_w_end - out_w_start
                
                
                tile_weights = torch.ones(
                    (batch, 1, output_frames, tile_out_height, tile_out_width),
                    device=decoded_tile.device,
                    dtype=decoded_tile.dtype,
                )
                
                # Calculate overlap regions in output space
                overlap_out_h = int(spatial_overlap * spatial_scale)
                overlap_out_w = int(spatial_overlap * spatial_scale)
                
                # Apply horizontal blending weights
                if h > 0 and overlap_out_w > 0:  # Left overlap
                    h_blend = torch.linspace(
                        0, 1, overlap_out_w, device=decoded_tile.device, dtype=decoded_tile.dtype
                    )
                    tile_weights[:, :, :, :, :overlap_out_w] *= h_blend.view(1, 1, 1, 1, -1)
                if h < horizontal_tiles - 1 and overlap_out_w > 0:  # Right overlap
                    h_blend = torch.linspace(
                        1, 0, overlap_out_w, device=decoded_tile.device, dtype=decoded_tile.dtype
                    )
                    tile_weights[:, :, :, :, -overlap_out_w:] *= h_blend.view(1, 1, 1, 1, -1)
                
                # Apply vertical blending weights
                if v > 0 and overlap_out_h > 0:  # Top overlap
                    v_blend = torch.linspace(
                        0, 1, overlap_out_h, device=decoded_tile.device, dtype=decoded_tile.dtype
                    )
                    tile_weights[:, :, :, :overlap_out_h, :] *= v_blend.view(1, 1, 1, -1, 1)
                if v < vertical_tiles - 1 and overlap_out_h > 0:  # Bottom overlap
                    v_blend = torch.linspace(
                        1, 0, overlap_out_h, device=decoded_tile.device, dtype=decoded_tile.dtype
                    )
                    tile_weights[:, :, :, -overlap_out_h:, :] *= v_blend.view(1, 1, 1, -1, 1)
                
                # Add weighted tile to output (use explicit int conversion for slicing)
                output[:, :, :, out_h_start:out_h_end, out_w_start:out_w_end] += (
                    decoded_tile * tile_weights
                )
                
                # Add weights to weight tensor
                weights[:, :, :, out_h_start:out_h_end, out_w_start:out_w_end] += tile_weights
        
        # Normalize by weights
        output = output / (weights + 1e-8)
        
        return output

    def _decode_spatial_temporal_tiled(
        self,
        x: torch.Tensor,
        tiling_threshold: int,
        spatial_overlap: int = 1,
        temporal_overlap: int = 1,
        horizontal_tiles: int = 2,
        vertical_tiles: int = 2,
        height_tiles: int = 2,
    ) -> torch.Tensor:
        """
        Decode latents using both 2x2 spatial and temporal tiling.
        
        Args:
            x: Latent tensor [B, C, F, H, W]
            tiling_threshold: Maximum total latent tokens per tile
            spatial_overlap: Overlap between spatial tiles in latent pixels
            temporal_overlap: Overlap between temporal tiles in latent frames
            
        Returns:
            Decoded video tensor [B, C, F_out, H_out, W_out]
        """
        batch, channels, frames, height, width = x.shape
        
        # VAE compression ratios
        spatial_scale = 32
        temporal_scale = 8
        
        # Calculate output dimensions
        output_frames = 1 + (frames - 1) * temporal_scale
        output_height = height * spatial_scale
        output_width = width * spatial_scale
        
        # Decoder outputs RGB (3 channels), not latent channels
        output_channels = 3
        
        # Check if output dimensions exceed 32-bit limits

        
        # Calculate temporal tile length based on threshold
        # After 2x2 spatial split, each spatial tile has (W/2) * (H/2) spatial tokens
        spatial_tile_tokens = (width // horizontal_tiles) * (height // vertical_tiles)
        temporal_tile_length = max(1, tiling_threshold // spatial_tile_tokens)
        
        # Initialize output tensor
        # Note: output has 3 channels (RGB), not latent channels
        device = x.device
        dtype = x.dtype
        output = torch.zeros(
            (batch, output_channels, output_frames, output_height, output_width),
            device=device,
            dtype=dtype,
        )
        
        # Process temporal chunks
        chunk_start = 0
        total_latent_frames = frames
        
        while chunk_start < total_latent_frames:
            # Calculate chunk boundaries with overlap
            if chunk_start == 0:
                # First chunk: no overlap needed
                chunk_end = min(chunk_start + temporal_tile_length, total_latent_frames)
                overlap_start = chunk_start
            else:
                # Subsequent chunks: include overlap from previous chunk
                overlap_start = max(1, chunk_start - temporal_overlap - 1)
                extra_frames = chunk_start - overlap_start
                chunk_end = min(
                    chunk_start + temporal_tile_length - extra_frames,
                    total_latent_frames,
                )
            
            # Extract temporal chunk
            temporal_chunk = x[:, :, overlap_start:chunk_end, :, :]
            
            # Decode this temporal chunk with spatial tiling
            decoded_chunk = self._decode_spatial_tiled(temporal_chunk, spatial_overlap)
            
            # Calculate temporal output boundaries
            out_t_start = 1 + overlap_start * temporal_scale
            
            if chunk_start == 0:
                # First chunk: write directly
                out_t_end = decoded_chunk.shape[2]
                output[:, :, :out_t_end] = decoded_chunk
            else:
                # Subsequent chunks: handle overlap blending
                if decoded_chunk.shape[2] == 1:
                    raise ValueError("Dropping first frame but tile has only 1 frame")
                
                # Drop first frame (overlap frame)
                decoded_chunk = decoded_chunk[:, :, 1:]
                
                # Recalculate output end after dropping first frame
                out_t_end = out_t_start + decoded_chunk.shape[2]
                
                # Calculate overlap frames in output space
                overlap_frames = temporal_overlap * temporal_scale
                
                # Ensure we don't exceed available frames
                overlap_frames = min(overlap_frames, decoded_chunk.shape[2])
                
                if overlap_frames > 0 and overlap_frames < decoded_chunk.shape[2]:
                    # Create blending weights for overlap region
                    frame_weights = torch.linspace(
                        0,
                        1,
                        overlap_frames + 2,
                        device=decoded_chunk.device,
                        dtype=decoded_chunk.dtype,
                    )[1:-1]  # Remove endpoints
                    tile_weights = frame_weights.view(1, 1, -1, 1, 1)
                    
                    after_overlap_frames_start = out_t_start + overlap_frames
                    
                    # Blend overlap region
                    overlap_output = decoded_chunk[:, :, :overlap_frames]
                    output[:, :, out_t_start:after_overlap_frames_start] = (
                        output[:, :, out_t_start:after_overlap_frames_start] * (1 - tile_weights)
                        + tile_weights * overlap_output
                    )
                    
                    # Write non-overlapping portion
                    remaining_frames = decoded_chunk.shape[2] - overlap_frames
                    if remaining_frames > 0:
                        # Ensure we don't exceed output tensor bounds
                        actual_out_t_end = min(out_t_end, after_overlap_frames_start + remaining_frames)
                        actual_remaining = actual_out_t_end - after_overlap_frames_start
                        if actual_remaining > 0:
                            output[:, :, after_overlap_frames_start:actual_out_t_end] = decoded_chunk[:, :, overlap_frames:overlap_frames + actual_remaining]
                else:
                    # No overlap to blend or all frames are overlap, write directly
                    output[:, :, out_t_start:out_t_end] = decoded_chunk
            
            # Move to next chunk
            chunk_start = chunk_end
        
        return output
