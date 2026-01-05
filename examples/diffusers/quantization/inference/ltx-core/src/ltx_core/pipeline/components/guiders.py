# Copyright (c) 2025 Lightricks. All rights reserved.
# Created by Amit Pintz.

from dataclasses import dataclass

import torch

from ltx_core.pipeline.components.protocols import GuiderProtocol


@dataclass(frozen=True)
class CFGGuider(GuiderProtocol):
    scale: float

    def delta(self, cond: torch.Tensor, uncond: torch.Tensor, *_extra_conds: torch.Tensor) -> torch.Tensor:
        return (self.scale - 1) * (cond - uncond)
