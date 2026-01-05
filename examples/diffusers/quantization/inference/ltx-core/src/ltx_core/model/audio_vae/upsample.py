import torch

from ltx_core.model.audio_vae.causal_conv_2d import make_conv2d
from ltx_core.model.audio_vae.causality_axis import CausalityAxis


class Upsample(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        with_conv: bool,
        causality_axis: CausalityAxis = CausalityAxis.HEIGHT,
    ) -> None:
        super().__init__()
        self.with_conv = with_conv
        self.causality_axis = causality_axis
        if self.with_conv:
            self.conv = make_conv2d(in_channels, in_channels, kernel_size=3, stride=1, causality_axis=causality_axis)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
            # Drop FIRST element in the causal axis to undo encoder's padding, while keeping the length 1 + 2 * n.
            # For example, if the input is [0, 1, 2], after interpolation, the output is [0, 0, 1, 1, 2, 2].
            # The causal convolution will pad the first element as [-, -, 0, 0, 1, 1, 2, 2],
            # So the output elements rely on the following windows:
            # 0: [-,-,0]
            # 1: [-,0,0]
            # 2: [0,0,1]
            # 3: [0,1,1]
            # 4: [1,1,2]
            # 5: [1,2,2]
            # Notice that the first and second elements in the output rely only on the first element in the input,
            # while all other elements rely on two elements in the input.
            # So we can drop the first element to undo the padding (rather than the last element).
            # This is a no-op for non-causal convolutions.
            match self.causality_axis:
                case CausalityAxis.NONE:
                    pass  # x remains unchanged
                case CausalityAxis.HEIGHT:
                    x = x[:, :, 1:, :]
                case CausalityAxis.WIDTH:
                    x = x[:, :, :, 1:]
                case CausalityAxis.WIDTH_COMPATIBILITY:
                    pass  # x remains unchanged
                case _:
                    raise ValueError(f"Invalid causality_axis: {self.causality_axis}")

        return x
