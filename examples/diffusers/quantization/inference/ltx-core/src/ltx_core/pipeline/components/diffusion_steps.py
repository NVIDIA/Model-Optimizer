# Copyright (c) 2025 Lightricks. All rights reserved.
# Created by Amit Pintz.

import torch

from ltx_core.pipeline.components.protocols import DiffusionStepProtocol


class EulerDiffusionStep(DiffusionStepProtocol):
    def step(self, sample: torch.Tensor, velocity: torch.Tensor, sigmas: torch.Tensor, step_index: int) -> torch.Tensor:
        sigma = sigmas[step_index]
        sigma_next = sigmas[step_index + 1]
        dt = sigma_next - sigma

        return (sample.to(torch.float32) + velocity.to(torch.float32) * dt).to(sample.dtype)
