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

"""Test of quantization with FSDP2."""

import copy
from contextlib import contextmanager
from functools import partial

import pytest
import torch
import torch.nn as nn
from _test_utils.torch.distributed.utils import synchronize_state_dict
from torch.distributed._composable.fsdp.fully_shard import fully_shard
from torch.distributed.fsdp import CPUOffloadPolicy
from torch.distributed.tensor import DTensor

import modelopt.torch.quantization as mtq
import modelopt.torch.quantization.model_calib as model_calib
from modelopt.torch.opt.dynamic import _pytorch_managed
from modelopt.torch.quantization.nn import StaticBlockScaleQuantizer, TensorQuantizer
from modelopt.torch.quantization.utils import (
    enable_weight_access_and_writeback,
    persistent_materialization,
)
from modelopt.torch.quantization.utils.layerwise_calib import LayerActivationCollector
from modelopt.torch.utils.dataset_utils import _forward_loop


def _test_fsdp2_simple_linear(rank, size):
    dim = 32
    model = nn.Linear(dim, dim).cuda(rank)
    inputs = torch.randn(2, 2, dim).cuda(rank)

    synchronize_state_dict(model)
    fsdp_model_after = copy.deepcopy(model)
    model = mtq.quantize(model, mtq.INT8_DEFAULT_CFG, lambda model: model(inputs))

    manager = model._get_dm_attribute_manager()
    assert "weight" in manager.da_keys()
    assert model._get_dm_attribute_manager().get_da_value("weight") is _pytorch_managed

    out_ref = model(inputs)

    fsdp_model = fully_shard(model)
    assert "weight" in manager.da_keys()
    out_test = fsdp_model(inputs)

    assert torch.allclose(out_ref, out_test)

    # quantize after fsdp2
    fsdp_model_after = fully_shard(fsdp_model_after)
    fsdp_model_after = mtq.quantize(
        fsdp_model_after, mtq.INT8_DEFAULT_CFG, lambda model: model(inputs)
    )
    out_fsdp_model_after = fsdp_model_after(inputs)
    assert torch.allclose(out_ref, out_fsdp_model_after)


def _test_nested_fsdp2_backward(rank, size, quant_cfg):
    dim = 32
    torch.manual_seed(1)
    model = nn.Sequential(
        nn.Sequential(nn.Linear(dim, dim), nn.Linear(dim, dim)),
        nn.Sequential(nn.Linear(dim, dim), nn.Linear(dim, dim)),
        nn.Linear(dim, dim),
    ).cuda()
    inputs = torch.randn(2, 2, dim).cuda()
    inputss = inputs.detach().clone()
    synchronize_state_dict(model)
    # test for quantization after fsdp2
    fsdp_model_quant_after = copy.deepcopy(model)

    def forward_loop(model):
        model(inputs)

    forward_loop = forward_loop if quant_cfg != mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG else None

    model = mtq.quantize(model, quant_cfg, forward_loop)
    fsdp_model = copy.deepcopy(model)

    optimizer_ref = torch.optim.SGD(model.parameters(), lr=0.1)
    out_ref = model(inputs)
    out_ref.sum().backward()

    fully_shard(fsdp_model[0])
    fully_shard(fsdp_model[1])
    fsdp_model = fully_shard(fsdp_model)

    optimizer_test = torch.optim.SGD(fsdp_model.parameters(), lr=0.1)
    out_test = fsdp_model(inputs)
    out_test.sum().backward()

    fully_shard(fsdp_model_quant_after[0])
    fully_shard(fsdp_model_quant_after[1])
    fsdp_model_quant_after = fully_shard(fsdp_model_quant_after)
    fsdp_model_quant_after = mtq.quantize(fsdp_model_quant_after, quant_cfg, forward_loop)
    optimizer_quant_after = torch.optim.SGD(fsdp_model_quant_after.parameters(), lr=0.1)
    out_quant_after = fsdp_model_quant_after(inputs)
    out_quant_after.sum().backward()

    assert torch.allclose(out_ref, out_test)
    assert torch.allclose(out_ref, out_quant_after)

    optimizer_ref.step()
    optimizer_ref.zero_grad()

    optimizer_test.step()
    optimizer_test.zero_grad()

    optimizer_quant_after.step()
    optimizer_quant_after.zero_grad()

    out_ref_1 = model(inputss)
    out_test_1 = fsdp_model(inputss)
    out_quant_after_1 = fsdp_model_quant_after(inputss)
    assert torch.allclose(out_ref_1, out_test_1, rtol=1e-4)
    assert torch.allclose(out_ref_1, out_quant_after_1, rtol=1e-4)


def test_fsdp_simple_linear(dist_workers):
    dist_workers.run(_test_fsdp2_simple_linear)


@pytest.mark.parametrize(
    "quant_cfg", [mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG, mtq.INT8_SMOOTHQUANT_CFG, mtq.INT4_AWQ_CFG]
)
def test_nested_fsdp2_backward(quant_cfg, dist_workers):
    dist_workers.run(partial(_test_nested_fsdp2_backward, quant_cfg=quant_cfg))


class _LSQBf16Linear(nn.Module):
    """Minimal bf16 module with LSQ learnable amax parameters."""

    def __init__(self, dim=16):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(dim, dim, dtype=torch.bfloat16))

        tq = TensorQuantizer()
        tq._num_bits = 4
        tq._unsigned = False
        tq._narrow_range = True
        tq._disabled = False
        tq._block_sizes = {-1: dim}
        tq._pass_through_bwd = True
        tq.register_buffer("_amax", torch.ones(dim, dtype=torch.bfloat16))
        self.weight_quantizer = StaticBlockScaleQuantizer.from_tensor_quantizer(tq)
        self.weight_quantizer.enable_lsq(
            quantize_scales=False,
            learnable_amax=["pre", "post"],
            dtype=torch.bfloat16,
        )

    def forward(self, inputs):
        weight = self.weight_quantizer._fake_quantize(self.weight)
        return torch.nn.functional.linear(inputs, weight)


def _test_lsq_bf16_learnable_amax_fsdp2(rank, size):
    torch.manual_seed(1)
    model = _LSQBf16Linear().cuda(rank)
    inputs = torch.randn(2, 16, device=rank, dtype=torch.bfloat16)
    synchronize_state_dict(model)

    assert {p.dtype for p in model.parameters()} == {torch.bfloat16}

    model = fully_shard(model)
    output = model(inputs)
    output.float().sum().backward()


def test_lsq_bf16_learnable_amax_fsdp2(dist_workers):
    dist_workers.run(_test_lsq_bf16_learnable_amax_fsdp2)


class _DecoderBlock(nn.Module):
    """Minimal decoder block for FSDP2 sequential tests."""

    def __init__(self, dim=32):
        super().__init__()
        self.attn = nn.Linear(dim, dim, bias=False)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim, bias=False), nn.ReLU(), nn.Linear(dim, dim, bias=False)
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = x + self.attn(self.norm(x))
        x = x + self.ffn(x)
        return x


class _SimpleTransformerModel(nn.Module):
    """Model with ``model.layers`` for layerwise calibration discovery."""

    def __init__(self, n_layers=3, dim=32):
        super().__init__()
        self.layers = nn.ModuleList([_DecoderBlock(dim) for _ in range(n_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def _test_layerwise_calibrate_fsdp2(rank, size):
    """Layerwise calibration on FSDP2-wrapped model matches non-FSDP reference."""
    dim = 32
    torch.manual_seed(1)
    model = _SimpleTransformerModel(n_layers=3, dim=dim).cuda()
    inputs = torch.randn(2, 2, dim).cuda()
    synchronize_state_dict(model)

    # Register discoverer for our simple model
    old_support = LayerActivationCollector._decoder_layer_support[:]
    LayerActivationCollector._decoder_layer_support = [
        (
            lambda m: hasattr(m, "layers") and isinstance(m.layers, nn.ModuleList),
            lambda m: m.layers,
        ),
        *old_support,
    ]
    original_persistent_materialization = model_calib.persistent_materialization
    materialized_fsdp_layers = 0

    @contextmanager
    def tracked_persistent_materialization(layer, writeback=True):
        nonlocal materialized_fsdp_layers
        with original_persistent_materialization(layer, writeback=writeback):
            if isinstance(layer, torch.distributed.fsdp.FSDPModule) and not writeback:
                assert all(not isinstance(param, DTensor) for param in layer.parameters())
                materialized_fsdp_layers += 1
            yield

    try:
        model_calib.persistent_materialization = tracked_persistent_materialization

        # Reference: non-FSDP layerwise calibration
        ref_model = copy.deepcopy(model)
        seq_cfg = copy.deepcopy(mtq.INT8_DEFAULT_CFG)
        seq_cfg["algorithm"] = {
            "method": "max",
            "layerwise": {"enable": True, "calib_mutates_weights": False},
        }
        mtq.quantize(ref_model, seq_cfg, lambda m: m(inputs))
        output_ref = ref_model(inputs)

        # Test: FSDP2-wrapped layerwise calibration
        for layer in model.layers:
            fully_shard(layer)
        model = fully_shard(model)
        mtq.quantize(model, seq_cfg, lambda m: m(inputs))
        output_test = model(inputs)

        assert torch.allclose(output_ref, output_test)
        assert materialized_fsdp_layers == len(model.layers)
    finally:
        model_calib.persistent_materialization = original_persistent_materialization
        LayerActivationCollector._decoder_layer_support = old_support


def test_layerwise_calibrate_fsdp2(dist_workers):
    dist_workers.run(_test_layerwise_calibrate_fsdp2)


def _test_persistent_materialization(rank, size):
    """persistent_materialization keeps weights accessible and writes back modifications."""
    dim = 32
    torch.manual_seed(1)
    model = nn.Sequential(
        nn.Sequential(nn.Linear(dim, dim), nn.Linear(dim, dim)),
        nn.Sequential(nn.Linear(dim, dim), nn.Linear(dim, dim)),
    ).cuda(rank)
    synchronize_state_dict(model)

    fully_shard(model[0])
    fully_shard(model[1])
    model = fully_shard(model)

    layer = model[0]
    inputs = torch.randn(2, dim).cuda(rank)

    # Warmup forward to trigger FSDP2's lazy_init (mirrors real usage where
    # layerwise_calibrate always runs get_first_layer_inputs first).
    model(inputs)

    # Save reference weight (gathered)
    with enable_weight_access_and_writeback(layer[0], model):
        ref_weight = layer[0].weight.clone()

    # Verify sharded before context
    assert isinstance(next(iter(layer.parameters())), DTensor)

    with persistent_materialization(layer):
        # Params are local tensors (not DTensors)
        assert not isinstance(layer[0].weight, DTensor)
        assert layer[0].weight.device.type == "cuda"

        # Run multiple forward passes (FSDP hooks fire, unshard/reshard are no-ops)
        for _ in range(3):
            layer(inputs)

        # Modify a weight
        layer[0].weight.data.add_(1.0)

    # After context: params restored to DTensors (sharded)
    assert isinstance(next(iter(layer.parameters())), DTensor)

    # Verify modification persisted
    with enable_weight_access_and_writeback(layer[0], model):
        assert torch.allclose(layer[0].weight, ref_weight + 1.0)

    with persistent_materialization(layer, writeback=False):
        assert not isinstance(layer[0].weight, DTensor)
        assert layer[0].weight.device.type == "cuda"
        layer(inputs)

    assert isinstance(next(iter(layer.parameters())), DTensor)


def test_persistent_materialization(dist_workers):
    dist_workers.run(_test_persistent_materialization)


def _test_writeback_root_unwrapped(rank, size):
    """Writeback works when only the decoder layers are FSDP2-wrapped and the root is unsharded.

    The root is only the search boundary: ``enable_weight_access_and_writeback(layer, model)``
    walks ``layer[0]``'s ancestors to the sharded decoder layer and gathers/writes back its
    DTensor, so the root needs no FSDP state. Covers the ``shard_root=False`` / nested-FSDP case
    (``fsdp2_wrap`` now defaults to ``shard_root=True``, wrapping the root too). Regression guard
    for the old ``isinstance(root_model, FSDPModule)`` assert that wrongly required a wrapped root.
    """
    dim = 32
    torch.manual_seed(1)
    # Root is a plain container; model[0] stands in for a decoder layer.
    model = nn.Sequential(nn.Sequential(nn.Linear(dim, dim), nn.Linear(dim, dim))).cuda(rank)
    synchronize_state_dict(model)

    # Wrap ONLY the "decoder layer" -- intentionally NO ``fully_shard(model)`` on the root,
    # mirroring fsdp2_wrap. ``root_model`` (model) is therefore not an FSDPModule.
    fully_shard(model[0])
    layer = model[0]
    inputs = torch.randn(2, dim).cuda(rank)

    # Warmup forward to trigger FSDP2's lazy_init (mirrors layerwise calibration).
    model(inputs)

    # This is the exact call save()/full_restore() make. Before the fix it tripped the
    # ``assert isinstance(root_model, FSDPModule)`` because the root is unwrapped — that's
    # the regression we guard. The DTensor-shape checks are not portable across torch
    # versions when the root is not FSDP-wrapped, so we just verify the writeback path
    # runs and mutations persist.
    with enable_weight_access_and_writeback(layer[0], model):
        ref_weight = layer[0].weight.clone()
        layer[0].weight.data.add_(1.0)  # mutate -> exercises the writeback path

    # Modification was written back into the shards.
    with enable_weight_access_and_writeback(layer[0], model):
        assert torch.allclose(layer[0].weight, ref_weight + 1.0)


def test_writeback_root_unwrapped(dist_workers):
    dist_workers.run(_test_writeback_root_unwrapped)


def _test_writeback_cpu_offload(rank, size):
    """Writeback round-trip when the FSDP2 shard is CPU-resident (``CPUOffloadPolicy``).

    Regression guard for the CPU↔GPU mirror added to
    ``fsdp2_weight_access_and_writeback_context``: the gathered shard is on CPU,
    so the helper mirrors it to GPU for in-context mutation and must copy
    modifications back to the CPU shard on exit.
    """
    dim = 32
    torch.manual_seed(1)
    model = nn.Sequential(nn.Sequential(nn.Linear(dim, dim), nn.Linear(dim, dim))).cuda(rank)
    synchronize_state_dict(model)

    # Wrap the "decoder layer" with cpu_offload; root stays unwrapped.
    fully_shard(model[0], offload_policy=CPUOffloadPolicy())
    layer = model[0]

    # Warmup forward triggers FSDP2's lazy_init.
    model(torch.randn(2, dim).cuda(rank))

    # Regression guard for the CPU→GPU mirror in fsdp2_weight_access_and_writeback_context:
    # if the helper handed back a CPU tensor under cpu_offload, calibration ops would crash
    # on the in-context mutation below (GPU activations vs CPU weight). The fact that this
    # block runs and the mutation persists is the evidence the mirror trip worked.
    with enable_weight_access_and_writeback(layer[0], model):
        ref_weight = layer[0].weight.clone()
        layer[0].weight.data.add_(1.0)

    # Mutation written back to the CPU shard.
    with enable_weight_access_and_writeback(layer[0], model):
        assert torch.allclose(layer[0].weight, ref_weight + 1.0)


def test_writeback_cpu_offload(dist_workers):
    dist_workers.run(_test_writeback_cpu_offload)


class _EmbedRootModel(nn.Module):
    """Root owns embed/norm params plus a decoder block. Mirrors the sharded-root layout
    where ``model(**batch)`` must fire the root's FSDP2 hook to unshard embed for the forward."""

    def __init__(self, vocab=16, dim=32):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.block = _DecoderBlock(dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, input_ids=None, **kwargs):
        return self.norm(self.block(self.embed(input_ids)))


def _test_sharded_root_calibration(rank, size):
    """Calibration through the standard forward loop works with a *sharded* FSDP2 root.

    Regression guard for removing ``materialize_fsdp2_root``: ``_forward_loop`` now calls
    ``model(**batch)`` (not ``model.forward``), so the root's FSDP2 pre/post-forward hooks
    unshard embed/norm for the forward and reshard them after — no manual materialization.
    With the old ``model.forward`` bypass this hit ``aten.embedding: mixed Tensor and DTensor``.
    """
    dim = 32
    torch.manual_seed(1)
    model = _EmbedRootModel(dim=dim).cuda(rank)
    synchronize_state_dict(model)

    # Shard the decoder block AND the root -> the root's own params (embed/norm) are sharded DTensors.
    fully_shard(model.block)
    model = fully_shard(model)
    assert isinstance(model.embed.weight, DTensor)

    batches = [{"input_ids": torch.randint(0, 16, (2, 8), device=rank)} for _ in range(2)]
    mtq.quantize(model, mtq.INT8_DEFAULT_CFG, lambda m: _forward_loop(m, batches))

    # Root params are resharded after calibration (needed for export / get_model_state_dict),
    # and the model still runs.
    assert isinstance(model.embed.weight, DTensor)
    assert isinstance(model.norm.weight, DTensor)
    model(input_ids=batches[0]["input_ids"])


def test_sharded_root_calibration(dist_workers):
    dist_workers.run(_test_sharded_root_calibration)
