import torch

from ltx_core.pipeline.components.guiders import CFGGuider


def test_cfg_guider_delta_scales_difference() -> None:
    guider = CFGGuider(scale=2.0)
    cond = torch.tensor([2.0, 4.0])
    uncond = torch.tensor([1.0, 1.0])

    # (scale - 1) * (cond - uncond) = 1.0 * [1.0, 3.0]
    delta = guider.delta(cond=cond, uncond=uncond)
    expected = [1.0, 3.0]

    assert delta.tolist() == expected
