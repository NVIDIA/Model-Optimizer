# Copyright (c) 2025 Lightricks. All rights reserved.
# Created by Andrew Kvochko

import torch

from ltx_core.model.transformer.attention import AttentionFunction


def test_attention_function() -> None:
    attention_function = AttentionFunction.DEFAULT
    assert attention_function(
        torch.tensor([[[1, 2, 3]]], dtype=torch.float32),
        torch.tensor([[[4, 5, 6]]], dtype=torch.float32),
        torch.tensor([[[7, 8, 9]]], dtype=torch.float32),
        1,
    ).tolist() == [[[7, 8, 9]]]
