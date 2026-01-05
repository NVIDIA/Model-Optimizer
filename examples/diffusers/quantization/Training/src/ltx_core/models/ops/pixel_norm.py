import torch
from torch import nn


class PixelNorm(nn.Module):
    def __init__(self, dim=1, eps=1e-8):
        super(PixelNorm, self).__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x):
        return x / torch.sqrt(torch.mean(x * x, dim=self.dim, keepdim=True) + self.eps)
