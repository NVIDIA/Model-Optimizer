from __future__ import annotations

import torch
from _test_utils.torch.distributed.utils import spawn_multiprocess_job
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard, distribute_tensor

from modelopt.torch.puzzletron.anymodel.capabilities import (
    AxisCapabilities,
    MagnitudeFallbackSpec,
    resolve_score_method,
)
from modelopt.torch.puzzletron.plugins.automodel.hooks import ActivationMagnitudeScorer
from modelopt.torch.puzzletron.plugins.automodel.output import write_scores
from modelopt.torch.puzzletron.plugins.automodel.reduction import MeshGroups
from modelopt.torch.puzzletron.plugins.automodel.target_resolver import build_magnitude_scorers


def _scorer(**overrides) -> ActivationMagnitudeScorer:
    kwargs = {
        "tensor_selector": "output",
        "scored_dim": -1,
        "output_field": "score",
        "expected_size": 2,
        "target_type": "fixture",
        "name": "layer.fixture",
    }
    kwargs.update(overrides)
    return ActivationMagnitudeScorer(nn.Identity(), MeshGroups(), **kwargs)


def test_fixed_and_packed_batches_have_identical_per_sample_magnitude() -> None:
    fixed = _scorer()
    fixed.set_batch_metadata(
        sequence_ids=torch.tensor([[0, 0], [1, 1]]),
        num_samples=2,
    )
    fixed(None, (), torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]))

    packed = _scorer()
    packed.set_batch_metadata(
        sequence_ids=torch.tensor([[0, 0, 1, 1]]),
        num_samples=2,
    )
    packed(None, (), torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]]))

    assert torch.equal(fixed.finalize()["score"], torch.tensor([8.0, 10.0]))
    assert torch.equal(packed.finalize()["score"], fixed.finalize()["score"])


def test_thd_packed_activation_matches_batched_activation() -> None:
    batched = _scorer()
    batched.set_batch_metadata(sequence_ids=torch.tensor([[0, 0], [1, 1]]), num_samples=2)
    batched(None, (), torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]))

    packed = _scorer()
    packed.set_batch_metadata(sequence_ids=torch.tensor([[0, 0, 1, 1]]), num_samples=2)
    packed(None, (), torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]))

    assert torch.equal(packed.finalize()["score"], batched.finalize()["score"])


def test_padding_and_cross_example_positions_are_excluded() -> None:
    scorer = _scorer()
    scorer.set_batch_metadata(
        sequence_ids=torch.tensor([[0, -1, 1]]),
        num_samples=2,
    )
    scorer(None, (), torch.tensor([[[2.0, 4.0], [100.0, 100.0], [6.0, 8.0]]]))

    result = scorer.finalize()
    metadata = result["magnitude_metadata"]["score"]

    assert torch.equal(result["score"], torch.tensor([8.0, 12.0]))
    assert metadata["sample_count"] == 2
    assert metadata["token_count"] == 2


def test_nonlast_scored_axis_averages_all_other_feature_dimensions() -> None:
    scorer = _scorer(scored_dim=2, expected_size=2)
    scorer.set_batch_metadata(sequence_ids=torch.tensor([[0]]), num_samples=1)
    # [batch, sequence, scored_heads, head_dim]
    scorer(None, (), torch.tensor([[[[1.0, 3.0], [4.0, 8.0]]]]))

    assert torch.equal(scorer.finalize()["score"], torch.tensor([2.0, 6.0]))


def test_tuple_output_requires_an_explicit_selector() -> None:
    scorer = _scorer()
    scorer.set_batch_metadata(sequence_ids=torch.tensor([[0]]), num_samples=1)

    try:
        scorer(None, (), (torch.ones(1, 1, 2), torch.zeros(1, 1, 2)))
    except ValueError as error:
        assert "ambiguous" in str(error)
    else:
        raise AssertionError("tuple output without an index must be rejected")

    indexed = _scorer(tensor_selector="output.1")
    indexed.set_batch_metadata(sequence_ids=torch.tensor([[0]]), num_samples=1)
    indexed(None, (), (torch.zeros(1, 1, 2), torch.tensor([[[3.0, 5.0]]])))
    assert torch.equal(indexed.finalize()["score"], torch.tensor([3.0, 5.0]))


def test_exact_resume_restores_additive_state_and_layout() -> None:
    first = _scorer()
    first.set_batch_metadata(sequence_ids=torch.tensor([[0]]), num_samples=1)
    first(None, (), torch.tensor([[[1.0, 2.0]]]))
    state = first.checkpoint_state()

    resumed = _scorer()
    resumed.load_checkpoint_state(state)
    resumed.set_batch_metadata(sequence_ids=torch.tensor([[0]]), num_samples=1)
    resumed(None, (), torch.tensor([[[3.0, 4.0]]]))

    assert torch.equal(resumed.finalize()["score"], torch.tensor([4.0, 6.0]))


def test_pp_microbatch_cursor_consumes_each_canonical_row_once() -> None:
    scorer = _scorer()
    scorer.set_batch_metadata(
        sequence_ids=torch.tensor([[0], [1]]),
        num_samples=2,
    )
    scorer(None, (), torch.tensor([[[1.0, 2.0]]]))
    scorer(None, (), torch.tensor([[[3.0, 4.0]]]))

    assert torch.equal(scorer.finalize()["score"], torch.tensor([4.0, 6.0]))


def test_finalize_emits_auditable_fallback_metadata() -> None:
    scorer = _scorer(tensor_selector="output", scored_dim=-1, output_field="head_score")
    scorer.set_batch_metadata(sequence_ids=torch.tensor([[0]]), num_samples=1)
    scorer(None, (), torch.ones(1, 1, 2))

    result = scorer.finalize()
    metadata = result["magnitude_metadata"]["head_score"]

    assert metadata["metric_kind"] == "magnitude_fallback"
    assert metadata["tensor_selector"] == "output"
    assert metadata["scored_dim"] == -1
    assert torch.equal(result["head_score"], torch.ones(2))


def test_specialized_metric_wins_and_missing_metric_fails_closed() -> None:
    fallback = MagnitudeFallbackSpec(
        observation_module="layer.self_attn",
        tensor_selector="output.0",
        scored_dim=2,
        output_field="head_score",
        expected_size=8,
    )
    specialized = AxisCapabilities(
        axis_id="heads",
        subblock_kind="attention",
        field="num_heads",
        score_hooks=("specialized",),
        magnitude_fallback=fallback,
    )
    fallback_only = AxisCapabilities(
        axis_id="heads",
        subblock_kind="attention",
        field="num_heads",
        magnitude_fallback=fallback,
    )
    missing = AxisCapabilities(axis_id="heads", subblock_kind="attention", field="num_heads")

    assert resolve_score_method(specialized) == "specialized"
    assert resolve_score_method(fallback_only) == "magnitude_fallback"
    try:
        resolve_score_method(missing)
    except ValueError as error:
        assert "no activation scorer" in str(error)
    else:
        raise AssertionError("an unscored prunable axis must fail preflight")


def test_output_writer_merges_nonoverlapping_axes_for_one_module(tmp_path) -> None:
    first = _scorer(output_field="head_score")
    second = _scorer(output_field="dim_score")
    for scorer in (first, second):
        scorer.set_batch_metadata(sequence_ids=torch.tensor([[0]]), num_samples=1)
        scorer(None, (), torch.ones(1, 1, 2))

    result = write_scores([first, second], str(tmp_path), MeshGroups())

    assert {"head_score", "dim_score"} <= set(result["layer.fixture"])


def test_target_builder_resolves_descriptor_module_templates() -> None:
    model = nn.Module()
    model.layers = nn.ModuleList([nn.Module(), nn.Module()])
    for layer in model.layers:
        layer.proj = nn.Identity()
    target = MagnitudeFallbackSpec(
        observation_module="layers.{layer_idx}.proj",
        tensor_selector="output",
        scored_dim=-1,
        output_field="channel_score",
        expected_size=2,
    )

    scorers = build_magnitude_scorers(model, MeshGroups(), [target], register=False)

    assert [scorer.name for scorer in scorers] == ["layers.0.proj", "layers.1.proj"]
    assert [scorer.block_idx for scorer in scorers] == [0, 1]


def _distributed_magnitude_job(rank: int, size: int) -> None:
    assert size == 4
    mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("cp", "tp"))
    groups = MeshGroups(
        token_group=mesh["cp"].get_group(),
        cp_group=mesh["cp"].get_group(),
        tp_group=mesh["tp"].get_group(),
    )
    full = torch.tensor(
        [[
            [1.0, 2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0, 6.0],
            [5.0, 6.0, 7.0, 8.0],
            [7.0, 8.0, 9.0, 10.0],
        ]]
    )
    sharded = distribute_tensor(full, mesh, [Shard(1), Shard(2)])
    scorer = _scorer(expected_size=4)
    scorer.groups = groups
    scorer.set_batch_metadata(sequence_ids=torch.zeros(1, 2, dtype=torch.long), num_samples=1)
    scorer(None, (), sharded)

    result = scorer.finalize()
    assert torch.equal(result["score"], full.abs().mean(dim=(0, 1)))
    metadata = result["magnitude_metadata"]["score"]
    assert metadata["sample_count"] == 1
    assert metadata["token_count"] == 4


def test_cp_tp_distributed_magnitude_has_no_duplicate_samples() -> None:
    spawn_multiprocess_job(size=4, job=_distributed_magnitude_job, backend="gloo")
