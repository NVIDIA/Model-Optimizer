# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import contextlib
import io
import re

import pytest
import torch
from _test_utils.import_helper import skip_if_no_mamba

skip_if_no_mamba()

from _test_utils.torch.megatron.models import get_mcore_mamba_hybrid_model
from _test_utils.torch.megatron.utils import (
    run_mcore_inference,
    run_mcore_inference_with_dummy_input,
)
from _test_utils.torch.misc import compare_outputs, set_seed
from _test_utils.torch.nas_prune.minitron_common import prune_minitron
from megatron.core.ssm.mamba_layer import MambaLayer
from megatron.core.transformer.identity_op import IdentityOp

import modelopt.torch.nas as mtn
from modelopt.torch.nas.plugins.megatron_model_stats import (
    mcore_memory_footprint_mb,
    mcore_param_count,
)
from modelopt.torch.prune.plugins.mcore_minitron import (
    ImportanceEstimatorRegistry,
    _convert_model_to_dynamic_space,
    get_mcore_minitron_config,
)

SEED = 1234


def _test_mcore_mamba_parameter_sorting(rank, size):
    set_seed(SEED)
    # Use relatively bigger model here for more accurate test for sorting
    channel_divisor = 64

    num_layers = size
    hybrid_override_pattern = "M" * size
    hidden_size = channel_divisor * 4
    mamba_state_dim = channel_divisor
    mamba_head_dim = 16
    mamba_num_groups = 2
    max_sequence_length = 32
    vocab_size = 64
    batch_size = 2

    model = get_mcore_mamba_hybrid_model(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=size,
        initialize_megatron=True,
        num_layers=num_layers,
        hybrid_override_pattern=hybrid_override_pattern,
        hidden_size=hidden_size,
        mamba_state_dim=mamba_state_dim,
        mamba_head_dim=mamba_head_dim,
        mamba_num_groups=mamba_num_groups,
        max_sequence_length=max_sequence_length,
        vocab_size=vocab_size,
        transformer_impl="transformer_engine",
        bf16=False,
    ).cuda()

    # Randomize norm weights instead of all zeros or ones
    for n, m in model.named_modules():
        if "norm" in n and not isinstance(m, IdentityOp):
            m.weight.data = torch.randn_like(m.weight)

    model.eval()
    dynamic_space = _convert_model_to_dynamic_space(
        model,
        get_mcore_minitron_config(
            hidden_size_divisor=channel_divisor,
            ffn_hidden_size_divisor=channel_divisor,
            mamba_head_dim_divisor=4,
        ),
    )
    registry = ImportanceEstimatorRegistry(model)  # register imp estimators and forward hooks

    # Compute activations for sorting
    for _ in range(5):
        run_mcore_inference_with_dummy_input(model, batch_size)

    # Get the output of the original model
    prompt_tokens = torch.randint(0, vocab_size, (batch_size, max_sequence_length)).cuda()
    y1 = run_mcore_inference(model, prompt_tokens)

    mtn.utils.sort_parameters(model)
    registry.cleanup()

    # check if all mamba_num_heads, mamba_head_dim, hidden_size have been sorted
    sortable_per_pp = [
        n for n, hp in dynamic_space.named_hparams(configurable=True) if hp.importance is not None
    ]
    # 2 mamba hps per layer + 1 for hidden_size (num_layers is not sorted!)
    assert len(sortable_per_pp) == 2 * num_layers // size + 1

    # sanity check if the model functionality is preserved after sorting
    y2 = run_mcore_inference(model, prompt_tokens)

    # check if the inference results after sorting is the same
    compare_outputs(y1, y2, rtol=1e-5, atol=1e-3)


def test_mcore_mamba_parameter_sorting(dist_workers):
    dist_workers.run(_test_mcore_mamba_parameter_sorting)


def _test_mcore_mamba_hybrid_pruning(rank, size, ckpt_dir):
    channel_divisor = 4

    num_layers = min(size * 2, 8)
    hidden_size = channel_divisor * 8
    ffn_hidden_size = channel_divisor * 2
    num_attention_heads = 8
    num_query_groups = 4
    mamba_state_dim = channel_divisor * 2
    mamba_head_dim = channel_divisor * 2
    mamba_num_groups = 2
    num_moe_experts = 8
    vocab_size = 32
    batch_size = 2

    def _get_model(initialize_megatron=True):
        model = get_mcore_mamba_hybrid_model(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=size,
            initialize_megatron=initialize_megatron,
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_query_groups=num_query_groups,
            ffn_hidden_size=ffn_hidden_size,
            mamba_state_dim=mamba_state_dim,
            mamba_head_dim=mamba_head_dim,
            mamba_num_groups=mamba_num_groups,
            moe_ffn_hidden_size=ffn_hidden_size,
            moe_shared_expert_intermediate_size=ffn_hidden_size,
            num_moe_experts=num_moe_experts,
            vocab_size=vocab_size,
            transformer_impl="transformer_engine",
            bf16=False,
        ).cuda()
        return model

    model = _get_model()

    mamba_layer = None
    for layer in model.decoder.layers:
        if isinstance(layer, MambaLayer):
            mamba_layer = layer
            break
    assert mamba_layer is not None, f"No MambaLayer found in the model PP rank {rank}!"
    mamba_num_heads = mamba_layer.mixer.nheads

    def forward_loop(m):
        for _ in range(2):
            run_mcore_inference_with_dummy_input(m, batch_size, hidden_size)

    # Traditional GPT pruning parameters
    pruned_ffn_hidden_size = ffn_hidden_size // 2
    pruned_num_attention_heads = num_attention_heads // 2
    pruned_hidden_size = hidden_size // 2
    pruned_num_moe_experts = num_moe_experts // 2

    # Mamba-specific pruning parameters
    pruned_mamba_num_heads = mamba_num_heads // 2
    pruned_mamba_head_dim = mamba_head_dim // 2

    # Base export config with GPT/Attention parameters
    export_config = {
        "ffn_hidden_size": pruned_ffn_hidden_size,
        "num_attention_heads": pruned_num_attention_heads,
        "hidden_size": pruned_hidden_size,
        "mamba_num_heads": pruned_mamba_num_heads,
        "mamba_head_dim": pruned_mamba_head_dim,
        "moe_ffn_hidden_size": pruned_ffn_hidden_size,
        "moe_shared_expert_intermediate_size": pruned_ffn_hidden_size,
        "num_moe_experts": pruned_num_moe_experts,
    }
    constraints = {"export_config": export_config}
    prune_minitron(
        model,
        constraints,
        {"forward_loop": forward_loop, "checkpoint": ckpt_dir},
        channel_divisor,
    )

    # Assert weights are pruned correctly
    mixer = mamba_layer.mixer
    bc = 2 * mixer.ngroups * mixer.d_state
    assert mixer.nheads == pruned_mamba_num_heads
    assert mixer.headdim == pruned_mamba_head_dim
    assert mixer.d_inner == pruned_mamba_num_heads * pruned_mamba_head_dim
    assert mixer.out_proj.out_features == pruned_hidden_size
    assert mixer.conv1d.in_channels == mixer.conv1d.out_channels == mixer.d_inner + bc

    # Assert model.config is updated for correct save/restoring
    assert model.config.ffn_hidden_size == pruned_ffn_hidden_size
    assert model.config.num_attention_heads == pruned_num_attention_heads
    assert model.config.hidden_size == pruned_hidden_size
    assert model.config.mamba_num_heads == pruned_mamba_num_heads
    assert model.config.mamba_head_dim == pruned_mamba_head_dim
    assert model.config.moe_ffn_hidden_size == pruned_ffn_hidden_size
    assert model.config.moe_shared_expert_intermediate_size == pruned_ffn_hidden_size
    assert model.config.num_moe_experts == pruned_num_moe_experts

    # Assert forward pass works on the pruned model
    run_mcore_inference_with_dummy_input(model, batch_size, pruned_hidden_size)

    # Assert re-pruning from checkpoint works without running the forward loop again
    model = _get_model(initialize_megatron=False)
    prune_minitron(model, constraints, {"checkpoint": ckpt_dir}, channel_divisor)


def test_mcore_mamba_hybrid_pruning(dist_workers, tmp_path):
    dist_workers.run(_test_mcore_mamba_hybrid_pruning, tmp_path / "minitron_scores")


# Shared parameters for the "ME*-" hybrid NAS tests
_NAS_CHANNEL_DIVISOR = 4
_NAS_BATCH_SIZE = 2
_NAS_MODEL_KWARGS = {
    "num_layers": 4,
    "hybrid_override_pattern": "ME*-",
    "hidden_size": 16,
    "ffn_hidden_size": 32,
    "num_attention_heads": 16,
    "num_query_groups": 4,
    "mamba_state_dim": 4,
    "mamba_num_heads": 8,
    "mamba_head_dim": 16,
    "mamba_num_groups": 2,
    "moe_ffn_hidden_size": 16,
    "moe_shared_expert_intermediate_size": 16,
    "num_moe_experts": 8,
    "vocab_size": 32,
}


def _make_nas_hybrid_model(size):
    return get_mcore_mamba_hybrid_model(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=size,
        initialize_megatron=True,
        transformer_impl="transformer_engine",
        bf16=False,
        **_NAS_MODEL_KWARGS,
    ).cuda()


def _nas_forward_loop(m):
    for _ in range(2):
        run_mcore_inference_with_dummy_input(m, _NAS_BATCH_SIZE, _NAS_MODEL_KWARGS["hidden_size"])


def _nas_score_func(m):
    c = m.config
    return (
        c.num_layers
        + c.hidden_size
        + c.ffn_hidden_size
        + c.mamba_num_heads
        + c.mamba_head_dim
        + c.num_attention_heads
        + c.num_moe_experts
        + c.moe_ffn_hidden_size
        + c.moe_shared_expert_intermediate_size
    )


def _base_nas_config(ckpt_dir):
    return {
        "forward_loop": _nas_forward_loop,
        "checkpoint": ckpt_dir,
        "score_func": _nas_score_func,
        "max_width_pruning": 0.5,
        "max_depth_pruning": 0.5,
        "hparams_to_skip": ["num_attention_heads", "moe_shared_expert_intermediate_size"],
        "top_k": 10,
    }


def _get_hybrid_layer_pattern(model):
    key = (
        "hybrid_override_pattern"
        if hasattr(model, "hybrid_override_pattern")
        else "hybrid_layer_pattern"
    )
    return getattr(model, key)


def _get_sorted_layers(searcher_state):
    return [
        layer
        for layer, _ in sorted(
            searcher_state["layer_scores"].items(), key=lambda x: x[1], reverse=True
        )
    ]


def _assert_top_k_candidates(searcher_state, constraint_key, expected_top_k, k=10):
    top_k = searcher_state["all_candidates_per_constraint"][constraint_key][:k]
    assert len(top_k) == k
    for actual, (ss_config, metrics, score) in zip(top_k, expected_top_k):
        assert actual.ss_config == ss_config, (actual.ss_config, ss_config)
        for metric_name, expected_value in metrics.items():
            actual_value = actual.metrics[metric_name]
            if isinstance(expected_value, float):
                assert actual_value == pytest.approx(expected_value), (actual.metrics, metrics)
            else:
                assert actual_value == expected_value, (actual.metrics, metrics)
        assert actual.score == score, (actual.score, score)


def _test_mcore_mamba_hybrid_pruning_nas_params(rank, size, ckpt_dir):
    set_seed(SEED)
    model = _make_nas_hybrid_model(size)

    baseline_params, baseline_active = mcore_param_count(
        model.config,
        model.vocab_size,
        model.share_embeddings_and_output_weights,
        hybrid_layer_pattern=_get_hybrid_layer_pattern(model),
    )
    assert baseline_params == 14984, baseline_params
    constraints = {
        "params": int(baseline_params * 0.5),
        "active_params": int(baseline_active * 0.55),
    }

    # Capture stdout to assert search space output
    stdout_capture = io.StringIO()
    with contextlib.redirect_stdout(stdout_capture):
        model, searcher_state = prune_minitron(
            model, constraints, _base_nas_config(ckpt_dir), _NAS_CHANNEL_DIVISOR
        )

    # Assert expected search space output is present (rich table format, strip ANSI codes first)
    captured_output = stdout_capture.getvalue()
    print(captured_output)
    clean_output = re.sub(r"\x1b\[[0-9;]*[mGKH]", "", captured_output)
    if rank == 0:
        lines = clean_output.splitlines()

        def assert_row(key: str, value: str) -> None:
            assert any(key in line and value in line for line in lines), (
                f"Expected row with {key!r} and {value!r} not found in search space table"
            )

        assert_row("num_layers", "[3, 4]")
        assert_row("hidden_size", "[12, 16]")
        assert_row("mamba_num_heads", "[6, 8]")
        assert_row("num_moe_experts", "[5, 6, 7, 8]")
        assert_row("moe_ffn_hidden_size", "[12, 16]")
        assert_row("Search space size", "512")

    pruned_params, pruned_active_params = mcore_param_count(
        model.config,
        model.vocab_size,
        model.share_embeddings_and_output_weights,
        hybrid_layer_pattern=_get_hybrid_layer_pattern(model),
    )
    assert pruned_params == 6536, pruned_params
    assert pruned_active_params == 6536, pruned_active_params

    # NOTE: Slight variation in layer ordering for MoE / Attention / MLP depending on PP configuration
    # This affects param counts when num_layers is pruned
    sorted_layers = _get_sorted_layers(searcher_state)
    # fmt: off
    if sorted_layers == [1, 4, 3, 2]:  # PP 1/2
        # Winner is 3-layer: keeps layers [1,4,3] from "ME*-" → drops 'E' (layer 2) → "M*-"
        assert _get_hybrid_layer_pattern(model) == "M*-", _get_hybrid_layer_pattern(model)
        expected_top_k = [
            # position 1: the one qualifying 4-layer model (active=6542 > 3-layer H=12 active),
            # demonstrating that active_params-first ranking can elevate 4-layer above 3-layer models
            [{"num_layers": 4, "hidden_size": 12, "mamba_num_heads": 6, "mamba_head_dim": 12, "num_moe_experts": 5, "moe_ffn_hidden_size": 12, "ffn_hidden_size": 32}, {"params": 7406, "active_params": 6542}, 115],  # noqa: E501
            # positions 2-9: 3-layer H=12 MNH=8 MHD=12 ffn=32 (active==params=6536, no MoE layer)
            [{"num_layers": 3, "hidden_size": 12, "mamba_num_heads": 8, "mamba_head_dim": 12, "num_moe_experts": 5, "moe_ffn_hidden_size": 12, "ffn_hidden_size": 32}, {"params": 6536, "active_params": 6536}, 116],  # noqa: E501
            [{"num_layers": 3, "hidden_size": 12, "mamba_num_heads": 8, "mamba_head_dim": 12, "num_moe_experts": 5, "moe_ffn_hidden_size": 16, "ffn_hidden_size": 32}, {"params": 6536, "active_params": 6536}, 120],  # noqa: E501
            [{"num_layers": 3, "hidden_size": 12, "mamba_num_heads": 8, "mamba_head_dim": 12, "num_moe_experts": 6, "moe_ffn_hidden_size": 12, "ffn_hidden_size": 32}, {"params": 6536, "active_params": 6536}, 117],  # noqa: E501
            [{"num_layers": 3, "hidden_size": 12, "mamba_num_heads": 8, "mamba_head_dim": 12, "num_moe_experts": 6, "moe_ffn_hidden_size": 16, "ffn_hidden_size": 32}, {"params": 6536, "active_params": 6536}, 121],  # noqa: E501
            [{"num_layers": 3, "hidden_size": 12, "mamba_num_heads": 8, "mamba_head_dim": 12, "num_moe_experts": 7, "moe_ffn_hidden_size": 12, "ffn_hidden_size": 32}, {"params": 6536, "active_params": 6536}, 118],  # noqa: E501
            [{"num_layers": 3, "hidden_size": 12, "mamba_num_heads": 8, "mamba_head_dim": 12, "num_moe_experts": 7, "moe_ffn_hidden_size": 16, "ffn_hidden_size": 32}, {"params": 6536, "active_params": 6536}, 122],  # noqa: E501
            [{"num_layers": 3, "hidden_size": 12, "mamba_num_heads": 8, "mamba_head_dim": 12, "num_moe_experts": 8, "moe_ffn_hidden_size": 12, "ffn_hidden_size": 32}, {"params": 6536, "active_params": 6536}, 119],  # noqa: E501
            [{"num_layers": 3, "hidden_size": 12, "mamba_num_heads": 8, "mamba_head_dim": 12, "num_moe_experts": 8, "moe_ffn_hidden_size": 16, "ffn_hidden_size": 32}, {"params": 6536, "active_params": 6536}, 123],  # noqa: E501
            # position 10: first 3-layer H=12 MNH=6 MHD=16 ffn=32 candidate (active=6506)
            [{"num_layers": 3, "hidden_size": 12, "mamba_num_heads": 6, "mamba_head_dim": 16, "num_moe_experts": 5, "moe_ffn_hidden_size": 12, "ffn_hidden_size": 32}, {"params": 6506, "active_params": 6506}, 118],  # noqa: E501
        ]
    else:
        raise RuntimeError(f"FIXME: Non deterministic test, assertions may fail: {sorted_layers=}")
    # fmt: on

    _assert_top_k_candidates(
        searcher_state,
        (("active_params", constraints["active_params"]), ("params", constraints["params"])),
        expected_top_k,
    )
    run_mcore_inference_with_dummy_input(model, _NAS_BATCH_SIZE, model.config.hidden_size)


@pytest.mark.skipif(
    torch.cuda.device_count() > 2, reason="Assertions not configured for more than 2 GPUs"
)
def test_mcore_mamba_hybrid_pruning_nas_params(dist_workers, tmp_path):
    dist_workers.run(_test_mcore_mamba_hybrid_pruning_nas_params, tmp_path / "minitron_scores")


def _test_mcore_mamba_hybrid_pruning_nas_memory_mb(rank, size, ckpt_dir):
    set_seed(SEED)
    dtype_bytes = 2
    sequence_length = 128
    model = _make_nas_hybrid_model(size)

    _, _, _, baseline_memory_mb = mcore_memory_footprint_mb(
        model.config,
        model.vocab_size,
        model.share_embeddings_and_output_weights,
        hybrid_layer_pattern=_get_hybrid_layer_pattern(model),
        dtype_bytes=dtype_bytes,
        sequence_length=sequence_length,
        batch_size=1,
    )
    memory_threshold = baseline_memory_mb * 0.7

    constraints = {"memory_mb": memory_threshold}
    config = {
        **_base_nas_config(ckpt_dir),
        "seq_length": sequence_length,
        "batch_size": 1,
    }
    model, searcher_state = prune_minitron(model, constraints, config, _NAS_CHANNEL_DIVISOR)

    pruned_params, _ = mcore_param_count(
        model.config,
        model.vocab_size,
        model.share_embeddings_and_output_weights,
        hybrid_layer_pattern=_get_hybrid_layer_pattern(model),
    )
    assert pruned_params == 10082, pruned_params

    sorted_layers = _get_sorted_layers(searcher_state)
    # fmt: off
    if sorted_layers == [1, 4, 3, 2]:
        expected_top_k = [
            [{"num_layers": 4, "hidden_size": 12, "mamba_num_heads": 8, "mamba_head_dim": 16, "num_moe_experts": 6, "moe_ffn_hidden_size": 12, "ffn_hidden_size": 24}, {"memory_mb": 0.0226287841796875},    114],  # noqa: E501
            [{"num_layers": 4, "hidden_size": 12, "mamba_num_heads": 8, "mamba_head_dim": 12, "num_moe_experts": 8, "moe_ffn_hidden_size": 16, "ffn_hidden_size": 32}, {"memory_mb": 0.022613525390625},     124],  # noqa: E501
            [{"num_layers": 4, "hidden_size": 12, "mamba_num_heads": 6, "mamba_head_dim": 16, "num_moe_experts": 8, "moe_ffn_hidden_size": 16, "ffn_hidden_size": 32}, {"memory_mb": 0.022556304931640625},  126],  # noqa: E501
            [{"num_layers": 4, "hidden_size": 16, "mamba_num_heads": 6, "mamba_head_dim": 12, "num_moe_experts": 7, "moe_ffn_hidden_size": 12, "ffn_hidden_size": 24}, {"memory_mb": 0.022541046142578125},  113],  # noqa: E501
            [{"num_layers": 3, "hidden_size": 16, "mamba_num_heads": 8, "mamba_head_dim": 16, "num_moe_experts": 5, "moe_ffn_hidden_size": 12, "ffn_hidden_size": 20}, {"memory_mb": 0.0225067138671875},    112],  # noqa: E501
            [{"num_layers": 3, "hidden_size": 16, "mamba_num_heads": 8, "mamba_head_dim": 16, "num_moe_experts": 5, "moe_ffn_hidden_size": 16, "ffn_hidden_size": 20}, {"memory_mb": 0.0225067138671875},    116],  # noqa: E501
            [{"num_layers": 3, "hidden_size": 16, "mamba_num_heads": 8, "mamba_head_dim": 16, "num_moe_experts": 6, "moe_ffn_hidden_size": 12, "ffn_hidden_size": 20}, {"memory_mb": 0.0225067138671875},    113],  # noqa: E501
            [{"num_layers": 3, "hidden_size": 16, "mamba_num_heads": 8, "mamba_head_dim": 16, "num_moe_experts": 6, "moe_ffn_hidden_size": 16, "ffn_hidden_size": 20}, {"memory_mb": 0.0225067138671875},    117],  # noqa: E501
            [{"num_layers": 3, "hidden_size": 16, "mamba_num_heads": 8, "mamba_head_dim": 16, "num_moe_experts": 7, "moe_ffn_hidden_size": 12, "ffn_hidden_size": 20}, {"memory_mb": 0.0225067138671875},    114],  # noqa: E501
            [{"num_layers": 3, "hidden_size": 16, "mamba_num_heads": 8, "mamba_head_dim": 16, "num_moe_experts": 7, "moe_ffn_hidden_size": 16, "ffn_hidden_size": 20}, {"memory_mb": 0.0225067138671875},    118],  # noqa: E501
        ]
    else:
        raise RuntimeError(f"FIXME: Non deterministic test, assertions may fail: {sorted_layers=}")
    # fmt: on

    _assert_top_k_candidates(searcher_state, (("memory_mb", memory_threshold),), expected_top_k)
    run_mcore_inference_with_dummy_input(model, _NAS_BATCH_SIZE, model.config.hidden_size)


@pytest.mark.skipif(
    torch.cuda.device_count() > 2, reason="Assertions not configured for more than 2 GPUs"
)
def test_mcore_mamba_hybrid_pruning_nas_memory_mb(dist_workers, tmp_path):
    dist_workers.run(_test_mcore_mamba_hybrid_pruning_nas_memory_mb, tmp_path / "minitron_scores")
