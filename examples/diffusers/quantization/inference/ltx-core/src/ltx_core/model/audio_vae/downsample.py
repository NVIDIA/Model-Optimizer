import torch

from ltx_core.model.audio_vae.causality_axis import CausalityAxis


class Downsample(torch.nn.Module):
    """
    A downsampling layer that can use either a strided convolution
    or average pooling. Supports standard and causal padding for the
    convolutional mode.
    """

    def __init__(
        self,
        in_channels: int,
        with_conv: bool,
        causality_axis: CausalityAxis = CausalityAxis.WIDTH,
    ) -> None:
        super().__init__()
        self.with_conv = with_conv
        self.causality_axis = causality_axis

        if self.causality_axis != CausalityAxis.NONE and not self.with_conv:
            raise ValueError("causality is only supported when `with_conv=True`.")

        if self.with_conv:
            # Do time downsampling here
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.with_conv:
            # Padding tuple is in the order: (left, right, top, bottom).
            match self.causality_axis:
                case CausalityAxis.NONE:
                    pad = (0, 1, 0, 1)
                case CausalityAxis.WIDTH:
                    pad = (2, 0, 0, 1)
                case CausalityAxis.HEIGHT:
                    pad = (0, 1, 2, 0)
                case CausalityAxis.WIDTH_COMPATIBILITY:
                    pad = (1, 0, 0, 1)
                case _:
                    raise ValueError(f"Invalid causality_axis: {self.causality_axis}")

            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            # This branch is only taken if with_conv=False, which implies causality_axis is NONE.
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)

        return x
