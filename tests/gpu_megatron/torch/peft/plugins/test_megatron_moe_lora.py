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

"""Smoke tests for the TE-grouped MoE LoRA plugin (OMNIML-4944).

Exercises modelopt/torch/peft/lora/plugins/megatron_moe.py end-to-end on a
small MoE GPT model built with ``moe_grouped_gemm=True`` and the TE
spec -- i.e. the production path used by Nemotron-3 Hybrid. The tests
verify registration, the zero-init forward identity, random-init
divergence from base, ``lora_dtype`` pinning, and gradient flow.
"""

import copy
from functools import partial

import pytest
import torch
import torch.nn.init as init
from _test_utils.torch.megatron.models import get_mcore_gpt_model
from _test_utils.torch.megatron.utils import (
    initialize_for_megatron,
    load_distributed_checkpoint,
    save_distributed_checkpoint,
)

import modelopt.torch.peft as mtpeft
import modelopt.torch.quantization as mtq
from modelopt.torch.opt.plugins.mcore_dist_checkpointing import (
    restore_sharded_modelopt_state,
    save_sharded_modelopt_state,
)
from modelopt.torch.peft.lora.plugins.megatron_moe import HAVE_TE_GROUPED
from modelopt.torch.utils.plugins import megatron_prefill

pytestmark = pytest.mark.skipif(
    not HAVE_TE_GROUPED,
    reason="Requires Transformer Engine >= 1.9.0.dev0 with TEGroupedLinear support",
)

if HAVE_TE_GROUPED:
    from modelopt.torch.peft.lora.plugins.megatron_moe import (
        _LoRATEGroupedColumnParallelLinear,
        _LoRATEGroupedRowParallelLinear,
    )

NUM_EXPERTS = 4

DEFAULT_LORA_CFG = {
    "adapter_type": "lora",
    "adapter_name": "default",
    "adapter_cfg": {
        "*": {"rank": 8, "scale": 1.0, "enable": True},
        "*output_layer*": {"enable": False},
    },
}

RANDOM_INIT_LORA_CFG = {
    "adapter_type": "lora",
    "adapter_name": "random",
    "adapter_cfg": {
        "*": {
            "rank": 8,
            "scale": 1.0,
            "lora_a_init": init.kaiming_uniform_,
            "lora_b_init": init.kaiming_uniform_,
            "enable": True,
        },
        "*output_layer*": {"enable": False},
    },
}

BF16_SIDECAR_LORA_CFG = {
    "adapter_type": "lora",
    "adapter_name": "bf16_sidecar",
    "adapter_cfg": {
        "*": {
            "rank": 8,
            "scale": 1.0,
            "lora_dtype": "bf16",
            "enable": True,
        },
        "*output_layer*": {"enable": False},
    },
}

SVDQUANT_LORA_CFG = {
    "adapter_type": "lora",
    "adapter_name": "svdquant",
    "adapter_cfg": {
        # SVDQuant init only on MoE expert grouped linears (matches the OMNIML-4944
        # spec: "LoRA adapters placed one per up_proj/down_proj across all MoE expert
        # layers"). Patterns are applied in order, last match wins. This also avoids
        # a pre-existing modelopt bug in the quantize -> LoRA path for TE plain linears
        # (LoRAQuantTERowParallelLinear lacks `input_size`).
        "*": {"enable": False},
        "*experts*linear_fc*": {
            "rank": 8,
            "scale": 1.0,
            "lora_init_method": "svdquant",
            "enable": True,
        },
        "*output_layer*": {"enable": False},
    },
}

# Per-tensor INT8 weight quantization. Compatible with TEGroupedLinear which restricts
# weight quantization to per-tensor (assertion at quantization/plugins/megatron.py:696
# in _process_quantizer_amax). A meaningful but small residual lets us check that
# the SVDQuant init produces non-trivial, residual-correlated LoRA factors. Matches
# the list-of-dicts shape used by NVFP4_DEFAULT_CONFIG in tests/.../test_megatron_peft.py.
INT8_PER_TENSOR_QUANT_CFG = {
    "quant_cfg": [
        {"quantizer_name": "*", "enable": False},
        {
            "quantizer_name": "*weight_quantizer",
            "cfg": {"num_bits": 8, "axis": None},
            "enable": True,
        },
        {"quantizer_name": "*output_layer*", "enable": False},
    ],
    "algorithm": "max",
}


def _moe_model_provider(tp_size: int, hidden_size: int = 256):
    """Build a tiny TE-MoE GPT with grouped GEMM (production path)."""
    model = get_mcore_gpt_model(
        tensor_model_parallel_size=tp_size,
        num_layers=2,
        ffn_hidden_size=None,
        num_attention_heads=4,
        activation_func="swiglu",
        transformer_impl="transformer_engine",
        hidden_size=hidden_size,
        vocab_size=64,
        moe_grouped_gemm=True,
        num_moe_experts=NUM_EXPERTS,
        moe_ffn_hidden_size=hidden_size * 2,
    ).cuda()
    return model.eval()


def _grouped_lora_modules(model):
    """Yield (name, module) for every TE-grouped LoRA wrapper in the model."""
    for name, module in model.named_modules():
        if isinstance(
            module, (_LoRATEGroupedColumnParallelLinear, _LoRATEGroupedRowParallelLinear)
        ):
            yield name, module


def _test_registration_and_zero_init_identity(rank, size):
    """Default (Kaiming on A / zeros on B) init must not perturb the base output."""
    initialize_for_megatron(tensor_model_parallel_size=size, pipeline_model_parallel_size=1)
    model = _moe_model_provider(tp_size=size)
    prompt_tokens = torch.randint(0, model.vocab_size, (2, model.max_sequence_length)).cuda()

    base_output = megatron_prefill(model, prompt_tokens)
    mtpeft.update_model(model, DEFAULT_LORA_CFG)
    lora_output = megatron_prefill(model, prompt_tokens)

    # The plugin must have replaced at least one expert linear per expert layer.
    grouped_lora_modules = list(_grouped_lora_modules(model))
    assert len(grouped_lora_modules) > 0, "no TE-grouped LoRA modules were registered"

    for _, module in grouped_lora_modules:
        # Per-expert stacked factor shapes
        assert "default" in module._lora_adapters
        adapter = module._lora_adapters["default"]
        a_w = adapter["lora_a"].weight
        b_w = adapter["lora_b"].weight
        assert a_w.shape == (module.num_gemms, 8, module.in_features)
        assert b_w.shape == (module.num_gemms, module.out_features, 8)

    # Zero-init on B → LoRA contribution is exactly zero → outputs match base.
    assert lora_output.shape == base_output.shape
    assert torch.allclose(lora_output, base_output, rtol=1e-5, atol=1e-5)

    # Disabling + re-enabling adapters round-trips correctly.
    mtpeft.disable_adapters(model)
    assert torch.allclose(megatron_prefill(model, prompt_tokens), base_output, rtol=1e-5, atol=1e-5)
    mtpeft.enable_adapters(model)
    assert torch.allclose(megatron_prefill(model, prompt_tokens), lora_output, rtol=1e-5, atol=1e-5)


def test_registration_and_zero_init_identity(dist_workers):
    dist_workers.run(_test_registration_and_zero_init_identity)


def _test_random_init_perturbs_output(rank, size):
    """Random init on both A and B must change the model output."""
    initialize_for_megatron(tensor_model_parallel_size=size, pipeline_model_parallel_size=1)
    model = _moe_model_provider(tp_size=size)
    prompt_tokens = torch.randint(0, model.vocab_size, (2, model.max_sequence_length)).cuda()

    base_output = megatron_prefill(model, prompt_tokens)
    mtpeft.update_model(model, RANDOM_INIT_LORA_CFG)
    lora_output = megatron_prefill(model, prompt_tokens)

    assert lora_output.shape == base_output.shape
    assert not torch.allclose(lora_output, base_output, rtol=1e-5)


def test_random_init_perturbs_output(dist_workers):
    dist_workers.run(_test_random_init_perturbs_output)


def _test_lora_dtype_pin(rank, size):
    """lora_dtype='bf16' must pin LoRA factor tensors to bfloat16."""
    initialize_for_megatron(tensor_model_parallel_size=size, pipeline_model_parallel_size=1)
    model = _moe_model_provider(tp_size=size)
    mtpeft.update_model(model, BF16_SIDECAR_LORA_CFG)

    found = False
    for _, module in _grouped_lora_modules(model):
        adapter = module._lora_adapters["bf16_sidecar"]
        assert adapter["lora_a"].weight.dtype == torch.bfloat16
        assert adapter["lora_b"].weight.dtype == torch.bfloat16
        found = True
    assert found, "no TE-grouped LoRA modules with bf16_sidecar adapter"


def test_lora_dtype_pin(dist_workers):
    dist_workers.run(_test_lora_dtype_pin)


def _test_gradient_flow(rank, size):
    """Random-init LoRA factors must receive gradients; base weights must not (default freeze)."""
    initialize_for_megatron(tensor_model_parallel_size=size, pipeline_model_parallel_size=1)
    model = _moe_model_provider(tp_size=size)
    prompt_tokens = torch.randint(0, model.vocab_size, (2, model.max_sequence_length)).cuda()

    mtpeft.update_model(model, RANDOM_INIT_LORA_CFG)
    # Intentionally keep the model in the eval mode set by _moe_model_provider.
    # Megatron's MoELayer.forward asserts at moe_layer.py:616 when
    # self.training AND attn_tp_group.size() > 1 AND not sequence_parallel --
    # the test factory hardcodes sequence_parallel=False, and dist_workers
    # runs at tp_size=4 here, so model.train() would raise before our LoRA
    # forward dispatch is even reached. PyTorch autograd is mode-independent,
    # so loss.backward() in eval mode still produces gradients on the LoRA
    # factors and base weights as required by this test.

    batch_size, seq_len = prompt_tokens.shape
    attention_mask = (
        torch.triu(
            torch.ones((batch_size, seq_len, seq_len), device=prompt_tokens.device), diagonal=1
        )
        .bool()
        .view(batch_size, 1, seq_len, seq_len)
    )
    output = model(prompt_tokens, position_ids=None, attention_mask=attention_mask)
    output.sum().backward()

    grouped_lora_param_names = []
    for mod_name, _ in _grouped_lora_modules(model):
        grouped_lora_param_names.append(mod_name)
    assert grouped_lora_param_names, "no TE-grouped LoRA modules present"

    saw_grouped_lora_grad = False
    for name, param in model.named_parameters():
        if "lora" in name and any(g in name for g in grouped_lora_param_names):
            assert param.grad is not None, f"lora param {name} has no grad"
            assert torch.any(param.grad != 0), f"lora param {name} grad is all zero"
            saw_grouped_lora_grad = True
        elif "lora" not in name:
            # default freeze_base_model=True → base params should not have grads
            assert param.grad is None, f"base param {name} unexpectedly got a grad"

    assert saw_grouped_lora_grad, "no per-expert stacked LoRA param received a gradient"


def test_gradient_flow(dist_workers):
    dist_workers.run(_test_gradient_flow)


def _test_svdquant_init_recovers_residual(rank, size):
    """After mtq.quantize -> mtpeft.update_model(svdquant), per-expert B @ A is a
    rank-r approximation of the quantization residual W_e - quant(W_e).

    Verifies (a) the LoRA factors are populated non-trivially when there's a real
    residual, and (b) the reconstruction error is bounded by the residual norm
    (i.e. SVD recovered some structure, not random/zero factors).
    """
    initialize_for_megatron(tensor_model_parallel_size=size, pipeline_model_parallel_size=1)
    model = _moe_model_provider(tp_size=size)
    prompt_tokens = torch.randint(0, model.vocab_size, (2, model.max_sequence_length)).cuda()

    def forward_func(mod):
        _ = megatron_prefill(model, prompt_tokens)

    # Quantize first so weight_quantizer exists on each TE-grouped linear.
    mtq.quantize(model, INT8_PER_TENSOR_QUANT_CFG, forward_func)
    # Then add LoRA: update_layer_lora reads weight_quantizer per module to compute residuals.
    mtpeft.update_model(model, SVDQUANT_LORA_CFG)

    found = False
    for name, module in _grouped_lora_modules(model):
        quantizer = getattr(module, "weight_quantizer", None)
        if quantizer is None or not getattr(quantizer, "is_enabled", True):
            continue
        adapter = module._lora_adapters["svdquant"]
        a = adapter["lora_a"].weight  # [E, r, in]
        b = adapter["lora_b"].weight  # [E, out, r]

        with torch.no_grad():
            for e in range(module.num_gemms):
                w_e = getattr(module, f"weight{e}")
                w_q = quantizer(w_e)
                expected_residual = (w_e - w_q).detach().float()
                reconstructed = b[e].float() @ a[e].float()  # [out, in]

                ref = expected_residual.norm()
                err = (reconstructed - expected_residual).norm()

                # When the residual is non-trivial, rank-r SVD should:
                # (1) produce non-zero factors
                # (2) reduce reconstruction error below the residual norm.
                if ref > 0:
                    assert a[e].abs().sum() > 0, (
                        f"{name} expert {e}: lora_a is all zeros despite non-zero residual"
                    )
                    assert b[e].abs().sum() > 0, (
                        f"{name} expert {e}: lora_b is all zeros despite non-zero residual"
                    )
                    assert err < ref, (
                        f"{name} expert {e}: reconstruction err {err.item():.4g} "
                        f">= residual norm {ref.item():.4g} -- SVDQuant init didn't recover "
                        f"any structure"
                    )
                found = True

    assert found, "no TE-grouped LoRA modules with svdquant adapter found"


def test_svdquant_init_recovers_residual(dist_workers):
    dist_workers.run(_test_svdquant_init_recovers_residual)


def _test_svdquant_init_no_quantizer_falls_back(rank, size):
    """Without a prior mtq.quantize call, lora_init_method='svdquant' has no residual
    to SVD; it must fall back to zero-init on both factors (with a warning).
    """
    initialize_for_megatron(tensor_model_parallel_size=size, pipeline_model_parallel_size=1)
    model = _moe_model_provider(tp_size=size)

    # Deliberately skip mtq.quantize -- no weight_quantizer attached to grouped linears.
    mtpeft.update_model(model, SVDQUANT_LORA_CFG)

    found = False
    for _, module in _grouped_lora_modules(model):
        adapter = module._lora_adapters["svdquant"]
        a = adapter["lora_a"].weight
        b = adapter["lora_b"].weight
        assert torch.all(a == 0), (
            "lora_a should be zero-init when svdquant runs without a weight_quantizer"
        )
        assert torch.all(b == 0), (
            "lora_b should be zero-init when svdquant runs without a weight_quantizer"
        )
        found = True
    assert found, "no TE-grouped LoRA modules with svdquant adapter found"


def test_svdquant_init_no_quantizer_falls_back(dist_workers):
    dist_workers.run(_test_svdquant_init_no_quantizer_falls_back)


def _test_quantize_then_lora_svdquant_state_dict_roundtrip(rank, size):
    """Disk save + load round-trips the per-expert stacked LoRA factors bitwise.

    Sanity scaffold for Phase 2 plumbing (OMNIML-4944): validates that a model
    in the quantize -> LoRA(svdquant) state can serialize its LoRA factors via
    torch.save and restore them via torch.load + load_state_dict. Doesn't use
    Megatron's distributed-checkpoint format (sharded_state_dict integration is
    deferred); each rank saves to its own temp path so dist_workers ranks don't
    collide. Confirms the LoRA-on-quant-checkpoint claim end-to-end at small
    scale before we wire into the production trainer.
    """
    import os
    import shutil
    import tempfile

    initialize_for_megatron(tensor_model_parallel_size=size, pipeline_model_parallel_size=1)
    model = _moe_model_provider(tp_size=size)
    prompt_tokens = torch.randint(0, model.vocab_size, (2, model.max_sequence_length)).cuda()

    def forward_func(mod):
        _ = megatron_prefill(model, prompt_tokens)

    mtq.quantize(model, INT8_PER_TENSOR_QUANT_CFG, forward_func)
    mtpeft.update_model(model, SVDQUANT_LORA_CFG)

    # Snapshot the original LoRA factors per grouped module.
    original_factors = {}
    for name, module in _grouped_lora_modules(model):
        adapter = module._lora_adapters["svdquant"]
        original_factors[name] = (
            adapter["lora_a"].weight.detach().clone(),
            adapter["lora_b"].weight.detach().clone(),
        )
    assert original_factors, "expected at least one TE-grouped LoRA module with svdquant adapter"

    # Save to a per-rank temp path so dist_workers ranks don't collide.
    tmpdir = tempfile.mkdtemp(prefix=f"omniml4944_lora_roundtrip_rank{rank}_")
    path = os.path.join(tmpdir, "state.pt")
    try:
        torch.save(model.state_dict(), path)

        # Mutate the LoRA factors in place so a successful load is observable.
        with torch.no_grad():
            for _, module in _grouped_lora_modules(model):
                adapter = module._lora_adapters["svdquant"]
                adapter["lora_a"].weight.zero_()
                adapter["lora_b"].weight.zero_()

        # Restore from disk and assert bitwise equality with the snapshot.
        loaded = torch.load(path, map_location="cuda", weights_only=True)
        model.load_state_dict(loaded, strict=True)

        for name, module in _grouped_lora_modules(model):
            adapter = module._lora_adapters["svdquant"]
            a_orig, b_orig = original_factors[name]
            assert torch.equal(adapter["lora_a"].weight, a_orig), (
                f"{name}: lora_a did not bitwise round-trip through torch.save/load"
            )
            assert torch.equal(adapter["lora_b"].weight, b_orig), (
                f"{name}: lora_b did not bitwise round-trip through torch.save/load"
            )
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_quantize_then_lora_svdquant_state_dict_roundtrip(dist_workers):
    dist_workers.run(_test_quantize_then_lora_svdquant_state_dict_roundtrip)


def _test_quantize_then_lora_svdquant_dist_checkpoint_roundtrip(tmp_path, rank, size):
    """Megatron dist-checkpoint round-trip for the per-expert stacked LoRA factors.

    Mirrors test_megatron_peft.py::test_mcore_quantize_then_lora_save_restore but for
    the TE-grouped MoE LoRA path: build a quantized+LoRA model, save with
    ``save_distributed_checkpoint`` + ``save_sharded_modelopt_state``, restore into a
    fresh model with ``restore_sharded_modelopt_state`` + ``load_distributed_checkpoint``,
    and assert the post-LoRA forward output matches the source within tolerance.

    Exercises the new ``_LoRATEGroupedBase.sharded_state_dict`` and
    ``_load_from_state_dict`` hooks end-to-end on Megatron's sharded format (the format
    that real Nemotron-3 checkpoints use).
    """
    initialize_for_megatron(tensor_model_parallel_size=size, pipeline_model_parallel_size=1)
    model_ref = _moe_model_provider(tp_size=size)
    model_test = _moe_model_provider(tp_size=size)
    prompt_tokens = torch.randint(
        0, model_ref.vocab_size, (2, model_ref.max_sequence_length)
    ).cuda()

    # Capture the pre-LoRA output of model_test for the "LoRA actually changed something"
    # sanity assertion at the bottom.
    pre_lora_output_test = megatron_prefill(model_test, prompt_tokens)

    def forward_func(mod):
        _ = megatron_prefill(model_ref, prompt_tokens)

    mtq.quantize(model_ref, INT8_PER_TENSOR_QUANT_CFG, forward_func)
    mtpeft.update_model(model_ref, SVDQUANT_LORA_CFG)
    ref_output = megatron_prefill(model_ref, prompt_tokens)

    # Save: ref's distributed checkpoint + sharded modelopt state (quantizers).
    save_distributed_checkpoint(tmp_path, model_ref)
    save_sharded_modelopt_state([model_ref], tmp_path)

    # Restore: sharded modelopt state first (attaches quantizers + LoRA module shape via
    # the saved modelopt_state), then the model parameters.
    restore_sharded_modelopt_state([model_test], tmp_path)
    model_test = load_distributed_checkpoint(tmp_path, model_test)
    test_output = megatron_prefill(model_test, prompt_tokens)

    assert torch.allclose(ref_output, test_output, rtol=1e-5), (
        "dist-checkpoint round-trip changed the model output -- LoRA factors did not fully restore"
    )
    assert not torch.allclose(ref_output, pre_lora_output_test, rtol=1e-5), (
        "ref output equals the pre-LoRA baseline; LoRA may not be perturbing the model"
    )


def test_quantize_then_lora_svdquant_dist_checkpoint_roundtrip(dist_workers, tmp_path):
    dist_workers.run(
        partial(_test_quantize_then_lora_svdquant_dist_checkpoint_roundtrip, str(tmp_path))
    )


# OMNIML-4998 AC-3: per-expert axis=0 weight quantization composed with the PR #3
# TE-grouped MoE LoRA plugin.
PER_EXPERT_QUANT_CFG = {
    "quant_cfg": [
        {"quantizer_name": "*", "enable": False},
        {
            "quantizer_name": "*experts.linear_fc*.weight_quantizer",
            "cfg": {"num_bits": 8, "axis": 0},
            "enable": True,
        },
        {"quantizer_name": "*output_layer*", "enable": False},
    ],
    "algorithm": "max",
}


def _test_per_expert_quant_then_lora_dist_checkpoint_roundtrip(
    tmp_path, lora_init_method, rank, size
):
    """Per-expert weight amax + PR #3 LoRA composed through Megatron dist-checkpoint.

    Builds a TE-grouped MoE GPT with ep=2 so the per-expert path actually exercises
    the OMNIML-4998 EP all-gather / narrow on save+restore. Quantizes with axis=0
    on the expert weight quantizers, then attaches LoRA via mtpeft.update_model.
    Saves to dist-checkpoint, reloads into a fresh model, and asserts the post-LoRA
    forward output matches the source within tolerance for both lora_init_method
    variants: ``zeros`` (no quantizer call in init -- pure composition) and
    ``svdquant`` (exercises the per-expert stack-then-quantize branch in
    _init_factors_svdquant).
    """
    initialize_for_megatron(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=2,
    )

    def _make_model():
        return (
            get_mcore_gpt_model(
                tensor_model_parallel_size=1,
                expert_model_parallel_size=2,
                num_layers=2,
                hidden_size=256,
                num_attention_heads=4,
                activation_func="swiglu",
                transformer_impl="transformer_engine",
                vocab_size=64,
                moe_grouped_gemm=True,
                num_moe_experts=NUM_EXPERTS,
                moe_ffn_hidden_size=512,
            )
            .cuda()
            .eval()
        )

    model_ref = _make_model()
    model_test = _make_model()
    prompt_tokens = torch.randint(
        0, model_ref.vocab_size, (2, model_ref.max_sequence_length)
    ).cuda()
    pre_lora_output = megatron_prefill(model_test, prompt_tokens)

    def forward_func(_mod):
        _ = megatron_prefill(model_ref, prompt_tokens)

    mtq.quantize(model_ref, copy.deepcopy(PER_EXPERT_QUANT_CFG), forward_func)
    lora_cfg = {
        "adapter_type": "lora",
        "adapter_name": f"compose_{lora_init_method}",
        "adapter_cfg": {
            "*": {"enable": False},
            "*experts*linear_fc*": {
                "rank": 8,
                "scale": 1.0,
                "lora_init_method": lora_init_method,
                "enable": True,
            },
            "*output_layer*": {"enable": False},
        },
    }
    mtpeft.update_model(model_ref, lora_cfg)
    ref_output = megatron_prefill(model_ref, prompt_tokens)

    save_distributed_checkpoint(tmp_path, model_ref)
    save_sharded_modelopt_state([model_ref], tmp_path)
    restore_sharded_modelopt_state([model_test], tmp_path)
    model_test = load_distributed_checkpoint(tmp_path, model_test)
    test_output = megatron_prefill(model_test, prompt_tokens)

    assert torch.allclose(ref_output, test_output, rtol=1e-5), (
        f"per-expert quant + LoRA ({lora_init_method}) dist-checkpoint round-trip "
        "changed the model output"
    )
    if lora_init_method != "zeros":
        assert not torch.allclose(ref_output, pre_lora_output, rtol=1e-5), (
            f"{lora_init_method} init failed to perturb the model output"
        )


@pytest.mark.parametrize("lora_init_method", ["zeros", "svdquant"])
def test_per_expert_quant_then_lora_dist_checkpoint_roundtrip(
    dist_workers_size_4, tmp_path, lora_init_method
):
    """OMNIML-4998 AC-3: per-expert quant + PR #3 LoRA + dist-checkpoint round-trip."""
    dist_workers_size_4.run(
        partial(
            _test_per_expert_quant_then_lora_dist_checkpoint_roundtrip,
            str(tmp_path),
            lora_init_method,
        )
    )
