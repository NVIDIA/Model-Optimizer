"""Tests for layer-wise distillation: FeatureExtractor, loss computation, and integration."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch import Tensor

from src.config import DistillationConfig
from src.feature_extractor import FeatureExtractor
from src.trainer import _resolve_layer_pairs


# ---------------------------------------------------------------------------
# Fixtures: tiny transformer-like model for testing hooks
# ---------------------------------------------------------------------------


class TinyBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x)


class TinyBlockTupleOutput(nn.Module):
    """Block that returns a tuple (like LTX-2's BasicAVTransformerBlock)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, x: Tensor) -> tuple[Tensor, None]:
        return self.linear(x), None


class TinyTransformer(nn.Module):
    """Minimal transformer with named blocks for hook testing."""

    def __init__(self, dim: int = 32, n_blocks: int = 4) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([TinyBlock(dim) for _ in range(n_blocks)])

    def forward(self, x: Tensor) -> Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class TinyTransformerTuple(nn.Module):
    """Transformer whose blocks return tuples."""

    def __init__(self, dim: int = 32, n_blocks: int = 4) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([TinyBlockTupleOutput(dim) for _ in range(n_blocks)])

    def forward(self, x: Tensor) -> Tensor:
        for block in self.blocks:
            x, _ = block(x)
        return x


# ---------------------------------------------------------------------------
# Tests: _resolve_layer_pairs
# ---------------------------------------------------------------------------


class TestResolveLayerPairs:
    def test_plain_strings(self):
        pairs = _resolve_layer_pairs(["blocks.0", "blocks.2"])
        assert pairs == [("blocks.0", "blocks.0"), ("blocks.2", "blocks.2")]

    def test_explicit_pairs(self):
        pairs = _resolve_layer_pairs([["teacher.0", "student.0"], ["teacher.2", "student.1"]])
        assert pairs == [("teacher.0", "student.0"), ("teacher.2", "student.1")]

    def test_mixed(self):
        pairs = _resolve_layer_pairs(["blocks.0", ["t.2", "s.1"]])
        assert pairs == [("blocks.0", "blocks.0"), ("t.2", "s.1")]

    def test_empty(self):
        assert _resolve_layer_pairs([]) == []


# ---------------------------------------------------------------------------
# Tests: FeatureExtractor
# ---------------------------------------------------------------------------


class TestFeatureExtractor:
    def test_captures_tensor_outputs(self):
        model = TinyTransformer(dim=16, n_blocks=4)
        paths = ["blocks.0", "blocks.2"]
        extractor = FeatureExtractor(model, paths)

        x = torch.randn(2, 5, 16)
        _ = model(x)

        feats = extractor.get_features()
        assert set(feats.keys()) == {"blocks.0", "blocks.2"}
        assert feats["blocks.0"].shape == (2, 5, 16)
        assert feats["blocks.2"].shape == (2, 5, 16)

    def test_captures_tuple_outputs_first_element(self):
        """Without transforms, tuples should extract first element."""
        model = TinyTransformerTuple(dim=16, n_blocks=4)
        extractor = FeatureExtractor(model, ["blocks.1"])

        x = torch.randn(2, 5, 16)
        _ = model(x)

        feats = extractor.get_features()
        assert isinstance(feats["blocks.1"], Tensor)
        assert feats["blocks.1"].shape == (2, 5, 16)

    def test_custom_output_transform(self):
        model = TinyTransformerTuple(dim=16, n_blocks=4)
        # Custom transform that negates the output
        transforms = {"blocks.1": lambda out: -out[0]}
        extractor = FeatureExtractor(model, ["blocks.1"], output_transforms=transforms)

        x = torch.randn(2, 5, 16)
        _ = model(x)

        feats = extractor.get_features()
        # Re-run without transform to compare
        extractor2 = FeatureExtractor(model, ["blocks.1"])
        _ = model(x)
        feats2 = extractor2.get_features()

        # Our transform negated the output
        torch.testing.assert_close(feats["blocks.1"], -feats2["blocks.1"])
        extractor2.remove()
        extractor.remove()

    def test_clear(self):
        model = TinyTransformer(dim=16, n_blocks=2)
        extractor = FeatureExtractor(model, ["blocks.0"])

        _ = model(torch.randn(1, 3, 16))
        assert len(extractor.get_features()) == 1

        extractor.clear()
        assert len(extractor.get_features()) == 0

    def test_remove_stops_capturing(self):
        model = TinyTransformer(dim=16, n_blocks=2)
        extractor = FeatureExtractor(model, ["blocks.0"])

        _ = model(torch.randn(1, 3, 16))
        assert len(extractor.get_features()) == 1

        extractor.remove()
        extractor.clear()

        _ = model(torch.randn(1, 3, 16))
        assert len(extractor.get_features()) == 0

    def test_invalid_module_path_raises(self):
        model = TinyTransformer(dim=16, n_blocks=2)
        with pytest.raises(ValueError, match="not found"):
            FeatureExtractor(model, ["blocks.99"])

    def test_features_update_on_each_forward(self):
        model = TinyTransformer(dim=16, n_blocks=2)
        extractor = FeatureExtractor(model, ["blocks.0"])

        x1 = torch.randn(1, 3, 16)
        _ = model(x1)
        feat1 = extractor.get_features()["blocks.0"].clone()

        x2 = torch.randn(1, 3, 16) + 10.0  # different input
        _ = model(x2)
        feat2 = extractor.get_features()["blocks.0"]

        assert not torch.allclose(feat1, feat2)
        extractor.remove()

    def test_default_transform(self):
        model = TinyTransformerTuple(dim=16, n_blocks=2)
        # Default transform applied to all paths without specific transforms
        default_fn = lambda out: out[0] * 2.0
        extractor = FeatureExtractor(
            model, ["blocks.0"], default_transform=default_fn
        )

        x = torch.randn(1, 3, 16)
        _ = model(x)

        # Compare with raw hook
        extractor2 = FeatureExtractor(model, ["blocks.0"])
        _ = model(x)
        raw = extractor2.get_features()["blocks.0"]
        scaled = extractor.get_features()["blocks.0"]

        torch.testing.assert_close(scaled, raw * 2.0)
        extractor.remove()
        extractor2.remove()


# ---------------------------------------------------------------------------
# Tests: layer distillation loss computation (unit-level)
# ---------------------------------------------------------------------------


class TestLayerDistillationLoss:
    """Test the loss computation logic extracted from DistillationTrainer."""

    @staticmethod
    def _compute_layer_loss(
        student_feats: dict[str, Tensor],
        teacher_feats: dict[str, Tensor],
        layer_pairs: list[tuple[str, str]],
        loss_type: str = "mse",
        normalize: bool = True,
    ) -> Tensor:
        """Standalone reimplementation matching trainer._compute_layer_distillation_loss."""
        losses = []
        for t_path, s_path in layer_pairs:
            s = student_feats[s_path]
            t = teacher_feats[t_path]
            if normalize:
                s = torch.nn.functional.normalize(s.float(), dim=-1)
                t = torch.nn.functional.normalize(t.float(), dim=-1)
            if loss_type == "mse":
                losses.append(torch.nn.functional.mse_loss(s, t))
            elif loss_type == "cosine":
                cos_sim = torch.nn.functional.cosine_similarity(
                    s.flatten(start_dim=1), t.flatten(start_dim=1), dim=-1
                )
                losses.append(1.0 - cos_sim.mean())
        return torch.stack(losses).mean()

    def test_mse_identical_features_zero_loss(self):
        feats = {"blocks.0": torch.randn(2, 10, 32)}
        loss = self._compute_layer_loss(
            feats, feats, [("blocks.0", "blocks.0")], "mse", normalize=False
        )
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_mse_different_features_positive_loss(self):
        s = {"b.0": torch.randn(2, 10, 32)}
        t = {"b.0": torch.randn(2, 10, 32)}
        loss = self._compute_layer_loss(s, t, [("b.0", "b.0")], "mse", normalize=False)
        assert loss.item() > 0.0

    def test_cosine_identical_features_zero_loss(self):
        feats = {"b.0": torch.randn(2, 10, 32)}
        loss = self._compute_layer_loss(
            feats, feats, [("b.0", "b.0")], "cosine", normalize=False
        )
        assert loss.item() == pytest.approx(0.0, abs=1e-5)

    def test_cosine_orthogonal_features(self):
        # Orthogonal vectors should give cosine similarity ~0, loss ~1
        s = {"b.0": torch.tensor([[[1.0, 0.0]]])}
        t = {"b.0": torch.tensor([[[0.0, 1.0]]])}
        loss = self._compute_layer_loss(
            s, t, [("b.0", "b.0")], "cosine", normalize=False
        )
        assert loss.item() == pytest.approx(1.0, abs=1e-5)

    def test_normalize_flag(self):
        s = {"b.0": torch.randn(2, 10, 32) * 100}
        t = {"b.0": torch.randn(2, 10, 32) * 0.01}
        loss_norm = self._compute_layer_loss(s, t, [("b.0", "b.0")], "mse", normalize=True)
        loss_raw = self._compute_layer_loss(s, t, [("b.0", "b.0")], "mse", normalize=False)
        # Normalized loss should be bounded (features on unit sphere), raw should be large
        assert loss_norm.item() < loss_raw.item()

    def test_multiple_pairs_averaged(self):
        # Two pairs: one identical (loss=0), one different
        s = {"b.0": torch.randn(2, 10, 32), "b.1": torch.randn(2, 10, 32)}
        t = {"b.0": s["b.0"].clone(), "b.1": torch.randn(2, 10, 32)}
        loss = self._compute_layer_loss(
            s, t, [("b.0", "b.0"), ("b.1", "b.1")], "mse", normalize=False
        )
        # Should be half of the non-zero pair's loss
        single_loss = self._compute_layer_loss(
            s, t, [("b.1", "b.1")], "mse", normalize=False
        )
        assert loss.item() == pytest.approx(single_loss.item() / 2.0, rel=1e-5)


# ---------------------------------------------------------------------------
# Tests: integration -- end-to-end with FeatureExtractor on two models
# ---------------------------------------------------------------------------


class TestLayerDistillationIntegration:
    """Integration test: student + teacher models with hooks and loss computation."""

    def test_end_to_end_self_distillation(self):
        """Same architecture for teacher/student, same module paths."""
        dim, n_blocks = 32, 4
        teacher = TinyTransformer(dim=dim, n_blocks=n_blocks)
        student = TinyTransformer(dim=dim, n_blocks=n_blocks)
        teacher.eval()

        paths = ["blocks.0", "blocks.2"]
        t_extractor = FeatureExtractor(teacher, paths)
        s_extractor = FeatureExtractor(student, paths)

        x = torch.randn(2, 8, dim)
        with torch.no_grad():
            _ = teacher(x)
        _ = student(x)

        t_feats = t_extractor.get_features()
        s_feats = s_extractor.get_features()

        assert set(t_feats.keys()) == {"blocks.0", "blocks.2"}
        assert set(s_feats.keys()) == {"blocks.0", "blocks.2"}

        # Compute loss
        losses = []
        for path in paths:
            s = torch.nn.functional.normalize(s_feats[path].float(), dim=-1)
            t = torch.nn.functional.normalize(t_feats[path].float(), dim=-1)
            losses.append(torch.nn.functional.mse_loss(s, t))
        layer_loss = torch.stack(losses).mean()

        assert layer_loss.item() > 0.0
        assert layer_loss.requires_grad  # student side has grad

        # Verify backward works
        layer_loss.backward()
        for p in student.parameters():
            if p.grad is not None:
                assert p.grad.abs().sum() > 0
                break

        t_extractor.remove()
        s_extractor.remove()

    def test_cross_architecture_distillation(self):
        """Different module paths for teacher and student."""
        teacher = TinyTransformer(dim=32, n_blocks=6)
        student = TinyTransformer(dim=32, n_blocks=3)
        teacher.eval()

        # Map teacher block 1,3,5 to student block 0,1,2
        t_paths = ["blocks.1", "blocks.3", "blocks.5"]
        s_paths = ["blocks.0", "blocks.1", "blocks.2"]
        pairs = list(zip(t_paths, s_paths))

        t_extractor = FeatureExtractor(teacher, t_paths)
        s_extractor = FeatureExtractor(student, s_paths)

        x = torch.randn(1, 4, 32)
        with torch.no_grad():
            _ = teacher(x)
        _ = student(x)

        t_feats = t_extractor.get_features()
        s_feats = s_extractor.get_features()

        # Verify all pairs captured
        for t_path, s_path in pairs:
            assert t_path in t_feats
            assert s_path in s_feats
            assert t_feats[t_path].shape == s_feats[s_path].shape

        t_extractor.remove()
        s_extractor.remove()

    def test_tuple_output_blocks_with_transforms(self):
        """Test with blocks that return tuples (LTX-2 style)."""
        teacher = TinyTransformerTuple(dim=16, n_blocks=4)
        student = TinyTransformerTuple(dim=16, n_blocks=4)
        teacher.eval()

        # Transform: extract first element from tuple
        transform = lambda out: out[0]
        transforms = {f"blocks.{i}": transform for i in [0, 2]}

        paths = ["blocks.0", "blocks.2"]
        t_ext = FeatureExtractor(teacher, paths, output_transforms=transforms)
        s_ext = FeatureExtractor(student, paths, output_transforms=transforms)

        x = torch.randn(1, 3, 16)
        with torch.no_grad():
            _ = teacher(x)
        _ = student(x)

        for path in paths:
            assert isinstance(t_ext.get_features()[path], Tensor)
            assert isinstance(s_ext.get_features()[path], Tensor)

        t_ext.remove()
        s_ext.remove()

    def test_loss_formula_integration(self):
        """Verify the full loss formula: L = alpha*task + (1-alpha)*[(1-gamma)*out_distill + gamma*layer_distill]."""
        alpha = 0.3
        gamma = 0.5

        task_loss = torch.tensor(1.0)
        output_distill_loss = torch.tensor(0.5)
        layer_distill_loss = torch.tensor(0.8)

        distill_loss = (1.0 - gamma) * output_distill_loss + gamma * layer_distill_loss
        total = alpha * task_loss + (1.0 - alpha) * distill_loss

        expected_distill = 0.5 * 0.5 + 0.5 * 0.8  # 0.65
        expected_total = 0.3 * 1.0 + 0.7 * 0.65    # 0.755

        assert distill_loss.item() == pytest.approx(expected_distill, abs=1e-6)
        assert total.item() == pytest.approx(expected_total, abs=1e-6)

    def test_gamma_zero_recovers_output_only(self):
        """gamma=0 should make layer loss irrelevant."""
        gamma = 0.0
        output_distill = torch.tensor(0.5)
        layer_distill = torch.tensor(999.0)  # should be ignored

        distill_loss = (1.0 - gamma) * output_distill + gamma * layer_distill
        assert distill_loss.item() == pytest.approx(0.5, abs=1e-6)


# ---------------------------------------------------------------------------
# Tests: config validation
# ---------------------------------------------------------------------------


class TestLayerDistillationConfig:
    def test_string_modules(self):
        cfg = DistillationConfig(layer_distillation_modules=["blocks.0", "blocks.2"])
        assert cfg.layer_distillation_modules == ["blocks.0", "blocks.2"]

    def test_pair_modules(self):
        cfg = DistillationConfig(
            layer_distillation_modules=[["t.0", "s.0"], ["t.2", "s.1"]]
        )
        assert cfg.layer_distillation_modules == [["t.0", "s.0"], ["t.2", "s.1"]]

    def test_mixed_modules(self):
        cfg = DistillationConfig(
            layer_distillation_modules=["blocks.0", ["t.2", "s.1"]]
        )
        assert cfg.layer_distillation_modules[0] == "blocks.0"
        assert cfg.layer_distillation_modules[1] == ["t.2", "s.1"]

    def test_none_disabled(self):
        cfg = DistillationConfig(layer_distillation_modules=None)
        assert cfg.layer_distillation_modules is None

    def test_invalid_pair_length_raises(self):
        with pytest.raises(Exception):
            DistillationConfig(layer_distillation_modules=[["a", "b", "c"]])

    def test_invalid_type_raises(self):
        with pytest.raises(Exception):
            DistillationConfig(layer_distillation_modules=[123])

    def test_weight_default_zero(self):
        cfg = DistillationConfig()
        assert cfg.layer_distillation_weight == 0.0

    def test_weight_bounds(self):
        cfg = DistillationConfig(layer_distillation_weight=0.0)
        assert cfg.layer_distillation_weight == 0.0
        cfg = DistillationConfig(layer_distillation_weight=1.0)
        assert cfg.layer_distillation_weight == 1.0

        with pytest.raises(Exception):
            DistillationConfig(layer_distillation_weight=1.5)
        with pytest.raises(Exception):
            DistillationConfig(layer_distillation_weight=-0.1)

    def test_loss_types(self):
        for lt in ("mse", "cosine"):
            cfg = DistillationConfig(layer_distillation_loss_type=lt)
            assert cfg.layer_distillation_loss_type == lt
