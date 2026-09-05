"""Test distillation loss balancers."""
# pyrefly: ignore [missing-import]
import pytest
from modelopt.torch.distill.loss_balancers import StaticLossBalancer

def test_static_loss_balancer_weight_validation():
    """Test that StaticLossBalancer correctly validates scalar and negative weights."""
    # 1. Verify int is accepted (and cast to list of float)
    balancer = StaticLossBalancer(1)
    assert balancer._kd_loss_weight == [1.0]

    # 2. Verify negative individual weights are rejected even if sum is valid
    with pytest.raises(ValueError, match="non-negative"):
        StaticLossBalancer([0.5, -0.3])
