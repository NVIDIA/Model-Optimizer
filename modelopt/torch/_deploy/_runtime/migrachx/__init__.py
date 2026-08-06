# SPDX-License-Identifier: Apache-2.0
# AMD ROCm MIGraphX deploy backend for ROCm Model Optimizer.
"""MIGraphX runtime backend for AMD ROCm (AMD equivalent of the TRT backend)."""

from .migrachx_client import MIGraphXLocalClient  # noqa: F401  (triggers registry)

__all__ = ["MIGraphXLocalClient"]
