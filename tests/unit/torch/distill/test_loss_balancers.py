"""Test distillation loss balancers."""
# pyrefly: ignore [missing-import]
import pytest
from modelopt.torch.distill.loss_balancers import StaticLossBalancer

@pytest.mark.parametrize(
    ("weight", "expected"),
    [(1, [1.0]), (0.5, [0.5])],
)
def test_static_loss_balancer_weight_validation(weight, expected):
    """Test that StaticLossBalancer correctly validates scalar and negative weights."""
    # 1. Verify scalar weights are accepted (and cast to list of float)
    balancer = StaticLossBalancer(weight)
    assert balancer._kd_loss_weight == expected

    # 2. Verify negative individual weights are rejected even if sum is valid
    with pytest.raises(ValueError, match="non-negative"):
        StaticLossBalancer([0.5, -0.3])
