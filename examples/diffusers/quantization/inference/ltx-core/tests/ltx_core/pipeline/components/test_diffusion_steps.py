import torch

from ltx_core.pipeline.components.diffusion_steps import EulerDiffusionStep


def test_euler_diffusion_step_simple_update() -> None:
    step = EulerDiffusionStep()
    sample = torch.tensor([1.0, 2.0])
    velocity = torch.tensor([0.5, 0.5])
    sigmas = torch.tensor([1.0, 0.5])

    out = step.step(sample=sample, velocity=velocity, sigmas=sigmas, step_index=0)

    # sigma = 1.0, sigma_next = 0.5, dt = -0.5
    # x + v * dt = [1.0, 2.0] + [0.5, 0.5] * -0.5 = [0.75, 1.75]
    expected = [0.75, 1.75]
    assert out.tolist() == expected
