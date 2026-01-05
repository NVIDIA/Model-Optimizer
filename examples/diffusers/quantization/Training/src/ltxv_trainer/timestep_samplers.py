import torch


class TimestepSampler:
    """Base class for timestep samplers.

    Timestep samplers are used to sample timesteps for diffusion models.
    They should implement both sample() and sample_for() methods.
    """

    def sample(self, batch_size: int, seq_length: int | None = None, device: torch.device = None) -> torch.Tensor:
        """Sample timesteps for a batch.

        Args:
            batch_size: Number of timesteps to sample
            seq_length: (optional) Length of the sequence being processed
            device: Device to place the samples on

        Returns:
            Tensor of shape (batch_size,) containing timesteps
        """
        raise NotImplementedError

    def sample_for(self, batch: torch.Tensor) -> torch.Tensor:
        """Sample timesteps for a specific batch tensor.

        Args:
            batch: Input tensor of shape (batch_size, seq_length, ...)

        Returns:
            Tensor of shape (batch_size,) containing timesteps
        """
        raise NotImplementedError


class UniformTimestepSampler(TimestepSampler):
    """Samples timesteps uniformly between min_value and max_value (default 0 and 1)."""

    def __init__(self, min_value: float = 0.0, max_value: float = 1.0):
        self.min_value = min_value
        self.max_value = max_value

    def sample(self, batch_size: int, seq_length: int | None = None, device: torch.device = None) -> torch.Tensor:  # noqa: ARG002
        return torch.rand(batch_size, device=device) * (self.max_value - self.min_value) + self.min_value

    def sample_for(self, batch: torch.Tensor) -> torch.Tensor:
        if batch.ndim != 3:
            raise ValueError(f"Batch should have 3 dimensions, got {batch.ndim}")

        return self.sample(batch.shape[0], device=batch.device)


class ShiftedLogitNormalTimestepSampler:
    """
    Samples timesteps from a shifted logit-normal distribution,
    where the shift is determined by the sequence length.
    """

    def __init__(self, std: float = 1.0):
        self.std = std

    def sample(self, batch_size: int, seq_length: int, device: torch.device = None) -> torch.Tensor:
        """Sample timesteps for a batch from a shifted logit-normal distribution.

        Args:
            batch_size: Number of timesteps to sample
            seq_length: Length of the sequence being processed, used to determine the shift
            device: Device to place the samples on

        Returns:
            Tensor of shape (batch_size,) containing timesteps sampled from a shifted
            logit-normal distribution, where the shift is determined by seq_length
        """
        shift = self._get_shift_for_sequence_length(seq_length)
        normal_samples = torch.randn((batch_size,), device=device) * self.std + shift
        timesteps = torch.sigmoid(normal_samples)
        return timesteps

    def sample_for(self, batch: torch.Tensor) -> torch.Tensor:
        """Sample timesteps for a specific batch tensor.

        Args:
            batch: Input tensor of shape (batch_size, seq_length, ...)

        Returns:
            Tensor of shape (batch_size,) containing timesteps sampled from a shifted
            logit-normal distribution, where the shift is determined by the sequence length
            of the input batch

        Raises:
            ValueError: If the input batch does not have 3 dimensions
        """
        if batch.ndim != 3:
            raise ValueError(f"Batch should have 3 dimensions, got {batch.ndim}")

        batch_size, seq_length, _ = batch.shape
        return self.sample(batch_size, seq_length, device=batch.device)

    @staticmethod
    def _get_shift_for_sequence_length(
        seq_length: int,
        min_tokens: int = 1024,
        max_tokens: int = 4096,
        min_shift: float = 0.95,
        max_shift: float = 2.05,
    ) -> float:
        # Calculate the shift value for a given sequence length using linear interpolation
        # between min_shift and max_shift based on sequence length.
        m = (max_shift - min_shift) / (max_tokens - min_tokens)  # Calculate slope
        b = min_shift - m * min_tokens  # Calculate y-intercept
        shift = m * seq_length + b  # Apply linear equation y = mx + b
        return shift


SAMPLERS = {
    "uniform": UniformTimestepSampler,
    "shifted_logit_normal": ShiftedLogitNormalTimestepSampler,
}


def example() -> None:
    # noinspection PyUnresolvedReferences
    import matplotlib.pyplot as plt  # noqa: PLC0415

    sampler = ShiftedLogitNormalTimestepSampler()
    for seq_length in [1024, 2048, 4096, 8192]:
        samples = sampler.sample(batch_size=1_000_000, seq_length=seq_length)

        # plot the histogram of the samples
        plt.hist(samples.numpy(), bins=100, density=True)
        plt.title(f"Timestep Samples for Sequence Length {seq_length}")
        plt.xlabel("Timestep")
        plt.ylabel("Density")
        plt.show()


if __name__ == "__main__":
    example()
