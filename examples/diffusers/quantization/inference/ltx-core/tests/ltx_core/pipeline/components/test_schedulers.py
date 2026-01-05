import torch

from ltx_core.pipeline.components.schedulers import LTX2Scheduler


def test_ltx2_scheduler_basic_properties() -> None:
    scheduler = LTX2Scheduler()

    steps = 4
    latent = torch.zeros(1, 4, 8, 8)  # non-None latent to exercise token-based shift

    sigmas = scheduler.execute(steps=steps, latent=latent)

    # We expect `steps + 1` sigma values.
    assert isinstance(sigmas, torch.Tensor)
    assert sigmas.shape == (steps + 1,)

    # All sigmas should be in [0, 1] and non-negative.
    assert torch.all(sigmas >= 0.0)
    assert torch.all(sigmas <= 1.0)
