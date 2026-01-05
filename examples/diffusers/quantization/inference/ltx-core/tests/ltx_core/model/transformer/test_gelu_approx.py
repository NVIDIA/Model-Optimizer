# Copyright (c) 2025 Lightricks. All rights reserved.
# Created by Andrew Kvochko

import torch

from ltx_core.model.transformer.gelu_approx import GELUApprox


def test_gelu_approx() -> None:
    gelu_approx = GELUApprox(1, 1)
    gelu_approx.load_state_dict({"proj.weight": torch.ones(1, 1), "proj.bias": torch.zeros(1)})
    x = torch.tensor([[2.0]])
    output = gelu_approx(x)
    assert output.allclose(torch.tensor([[2.0]]), atol=0.05)
