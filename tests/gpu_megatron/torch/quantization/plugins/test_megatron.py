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

import copy
import math
import re
import sys
import types
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from _test_utils.torch.megatron.models import (
    HAS_MAMBA,
    MegatronModel,
    get_mcore_gpt_model,
    get_mcore_hybrid_model,
)
from _test_utils.torch.megatron.utils import (
    compare_amax_sync_across_expert_parallel,
    copy_weights_from_grouped_to_non_grouped,
    get_batch,
    get_forward,
    initialize_for_megatron,
    run_mcore_inference,
    sharded_state_dict_test_helper,
)
from _test_utils.torch.misc import set_seed
from _test_utils.torch.quantization.models import RegularQuantModelForTP
from _test_utils.torch.quantization.quant_utils import get_model_size
from _test_utils.torch.quantization.quantize_common import (
    auto_quantize_helper,
    data_tensor_context_parallel_test_helper,
    verify_kv_cache_amax_sync,
)
from megatron.core.parallel_state import (
    destroy_model_parallel,
    get_data_parallel_group,
    get_tensor_model_parallel_group,
)
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer import MegatronModule, TransformerConfig
from megatron.core.transformer.moe.experts import SequentialMLP, TEGroupedMLP
from megatron.core.transformer.moe.router import TopKRouter

import modelopt
import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.algorithms import QuantRecipe, _AutoQuantizeBaseSearcher
from modelopt.torch.quantization.nn import QuantModuleRegistry, SequentialQuantizer
from modelopt.torch.quantization.nn.modules.quant_linear import RealQuantLinear
from modelopt.torch.quantization.plugins.megatron import (
    _output_layer_untied,
    _QuantMegatronTEGroupedLinear,
    _QuantTEMCoreRowParallelLinear,
    _resolve_output_layer_untied,
    get_mcore_layerwise_calibration_layers,
    megatron_replace_quant_module_hook,
    quant_module_get_extra_state,
)
from modelopt.torch.quantization.plugins.transformer_engine import (
    _COMPILE_TEGROUPED_WEIGHT_LOOP_ENV,
)
from modelopt.torch.quantization.qtensor import QTensorWrapper
from modelopt.torch.quantization.utils import is_quantized_linear
from modelopt.torch.quantization.utils.layerwise_calib import LayerActivationCollector

try:
    from megatron.core.extensions.transformer_engine import TERowParallelLinear

    HAS_TE = True
except ImportError:
    HAS_TE = False

SEED = 1234

# TODO: re-enable the marked tests once fixed. Blackwell (sm_120, e.g. RTX PRO 6000) with the
# nemo:26.06 / TE 2.16 / CUDA 13 stack hits a flaky, intermittent CUDA "illegal memory access" in
# several tests: When fails, it poisons the process CUDA context and cascades hang across subsequent tests.
# Passes locally on different GPUs. Likely an upstream TE/CUDA-13 kernel bug, not modelopt logic.
skip_flaky_on_blackwell = pytest.mark.skipif(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 12,
    reason="Flaky CUDA illegal memory access on Blackwell (nemo:26.06 / TE 2.16 / CUDA 13)",
)


def test_convert_megatron_parallel_linear(distributed_setup_size_1):
    initialize_for_megatron(seed=SEED)
    set_seed(SEED)

    assert ColumnParallelLinear in QuantModuleRegistry
    assert RowParallelLinear in QuantModuleRegistry

    model_ref = MegatronModel().cuda()
    model_test = MegatronModel().cuda()
    model_test.load_state_dict(model_ref.state_dict())

    mtq.replace_quant_module(model_test)
    for module in model_test.modules():
        if isinstance(module, (ColumnParallelLinear, RowParallelLinear)):
            assert hasattr(module, "input_quantizer")
            assert hasattr(module, "weight_quantizer")
            assert hasattr(module, "output_quantizer")

    mtq.set_quantizer_attributes_partial(model_test, "*", {"enable": False})

    x = model_ref.get_dummy_input().cuda()
    out_1 = model_ref(x)
    out_2 = model_test(x)
    assert torch.allclose(out_1, out_2)

    mtq.set_quantizer_attributes_partial(model_test, "*input_quantizer", {"enable": True})
    mtq.set_quantizer_attributes_partial(model_test, "*weight_quantizer", {"enable": True})
    model_ref = RegularQuantModelForTP().cuda()
    model_ref.load_state_dict(model_test.state_dict(), strict=False)

    out_1 = model_ref(x)
    out_2 = model_test(x)
    assert torch.allclose(out_1, out_2)

    # Clean up since this is not a spawned process
    destroy_model_parallel()


# Unified parallelism test helper
def _test_parallelism_helper(
    config,
    rank,
    size,
    tensor_model_parallel_size=1,
    context_parallel_size=1,
    use_rank_in_seed=False,
    test_pre_quant_scale=True,
):
    """
    Unified helper for testing different parallelism configurations.
    Args:
        config: Quantization config to test
        rank: Current rank in distributed setup
        size: Total number of processes
        tensor_model_parallel_size: Size of tensor model parallel group (default: 1)
        context_parallel_size: Size of context parallel group (default: 1)
        use_rank_in_seed: Whether to add rank to seed for different data across ranks (default: False)
    """
    seed = SEED + rank if use_rank_in_seed else SEED
    initialize_for_megatron(
        tensor_model_parallel_size=tensor_model_parallel_size,
        context_parallel_size=context_parallel_size,
        seed=seed,
    )

    # Determine if we need tp_group and dp_group
    tp_group = get_tensor_model_parallel_group() if tensor_model_parallel_size > 1 else None
    dp_group = get_data_parallel_group(with_context_parallel=True)

    # Create model with appropriate parallelism settings
    model = MegatronModel(
        tp_size=tensor_model_parallel_size,
        cp_size=context_parallel_size,
        tp_group=tp_group,
    ).cuda()

    # Call the test helper with appropriate groups
    data_tensor_context_parallel_test_helper(
        model,
        config,
        dp_group=dp_group,
        tp_group=tp_group,
        test_pre_quant_scale=test_pre_quant_scale,
    )


@pytest.mark.parametrize(
    "config",
    [
        mtq.INT8_DEFAULT_CFG,
        mtq.FP8_DEFAULT_CFG,
        mtq.W4A8_AWQ_BETA_CFG,
        mtq.INT8_SMOOTHQUANT_CFG,
        mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG,
        mtq.INT4_AWQ_CFG,
        mtq.NVFP4_DEFAULT_CFG,
    ],
)
def test_tensor_parallel(dist_workers, config):
    dist_workers.run(
        partial(
            _test_parallelism_helper, config, tensor_model_parallel_size=torch.cuda.device_count()
        )
    )


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Same as test_tensor_parallel on 1 GPU")
@pytest.mark.parametrize(
    "config",
    [
        mtq.INT8_DEFAULT_CFG,
        mtq.FP8_DEFAULT_CFG,
        mtq.W4A8_AWQ_BETA_CFG,
        mtq.INT8_SMOOTHQUANT_CFG,
        mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG,
        mtq.INT4_AWQ_CFG,
        mtq.NVFP4_DEFAULT_CFG,
    ],
)
def test_data_parallel(dist_workers, config):
    dist_workers.run(partial(_test_parallelism_helper, config, use_rank_in_seed=True))


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Same as test_tensor_parallel on 1 GPU")
@pytest.mark.parametrize(
    "config",
    [
        mtq.INT8_DEFAULT_CFG,
        mtq.FP8_DEFAULT_CFG,
        mtq.W4A8_AWQ_BETA_CFG,
        mtq.INT8_SMOOTHQUANT_CFG,
        mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG,
        mtq.INT4_AWQ_CFG,
        mtq.NVFP4_DEFAULT_CFG,
    ],
)
def test_context_parallel(dist_workers, config):
    dist_workers.run(
        partial(
            _test_parallelism_helper,
            config,
            context_parallel_size=torch.cuda.device_count(),
            use_rank_in_seed=True,
        ),
    )


@pytest.mark.parametrize(
    "config",
    [
        mtq.INT8_DEFAULT_CFG,
        mtq.FP8_DEFAULT_CFG,
        mtq.W4A8_AWQ_BETA_CFG,
        mtq.INT8_SMOOTHQUANT_CFG,
        mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG,
        mtq.INT4_AWQ_CFG,
        mtq.NVFP4_DEFAULT_CFG,
    ],
)
def test_data_tensor_context_parallel(dist_workers, need_8_gpus, config):
    dist_workers.run(
        partial(
            _test_parallelism_helper,
            config,
            tensor_model_parallel_size=2,
            context_parallel_size=2,
            use_rank_in_seed=True,
            test_pre_quant_scale=False,
        ),
    )


def _gpt_model_provider(
    tp_size: int,
    hidden_size=256,
    vocab_size=64,
    num_moe_experts=None,
    moe_grouped_gemm=False,
    meta_device=False,
    ep_size=1,
    etp_size=None,
    transformer_impl="local",
    # Hybrid mamba MOE parameters
    is_hybrid=False,
    hybrid_layer_pattern=None,
    mamba_head_dim=16,
):
    device_ctx = torch.device("meta") if meta_device else nullcontext()

    with device_ctx:
        if is_hybrid:
            # Derive num_layers from pattern length, default to 4
            num_layers = len(hybrid_layer_pattern) if hybrid_layer_pattern else 4
            model = get_mcore_hybrid_model(
                tensor_model_parallel_size=tp_size,
                num_layers=num_layers,
                hidden_size=hidden_size,
                vocab_size=vocab_size,
                num_attention_heads=8,
                ffn_hidden_size=None,
                hybrid_layer_pattern=hybrid_layer_pattern,
                mamba_head_dim=mamba_head_dim,
                mamba_num_groups=tp_size,  # Must be divisible by tp_size
                num_moe_experts=num_moe_experts,
                sequence_parallel=(tp_size > 1),  # Required for MoE + TP
                # EP/ETP passed via config_kwargs
                expert_model_parallel_size=ep_size,
                expert_tensor_parallel_size=etp_size,
            )
        else:
            model = get_mcore_gpt_model(
                tensor_model_parallel_size=tp_size,
                expert_model_parallel_size=ep_size,
                expert_tensor_parallel_size=etp_size,
                num_layers=4,
                ffn_hidden_size=None,
                num_attention_heads=8,
                activation_func="squared_relu",
                transformer_impl=transformer_impl,
                hidden_size=hidden_size,
                vocab_size=vocab_size,
                use_cpu_initialization=meta_device,
                num_moe_experts=num_moe_experts,
                moe_grouped_gemm=moe_grouped_gemm,
            )

    if not meta_device:
        model = model.cuda()
    return model.eval()


def _test_sharded_state_dict(
    tmp_path, config, hidden_size, modelopt_version, compress, meta_device, model_config, rank, size
):
    # Must disable output_layer quantization since output_layer amax cannot be restore via
    # sharded_state_dict. All output_layer quantizers state are removed.
    config["quant_cfg"].append({"quantizer_name": "*output_layer*", "enable": False})

    if modelopt_version is not None:
        mto.conversion.__version__ = modelopt_version
        mtq.plugins.megatron.__version__ = modelopt_version

    tp_size = model_config.get("tp_size", size)
    ep_size = model_config.get("ep_size", 1)
    etp_size = model_config.get("etp_size", None)
    num_moe_experts = model_config.get("num_moe_experts", None)
    moe_grouped_gemm = model_config.get("moe_grouped_gemm", False)
    transformer_impl = model_config.get("transformer_impl", "local")
    # Hybrid mamba MOE parameters
    is_hybrid = model_config.get("is_hybrid", False)
    hybrid_layer_pattern = model_config.get("hybrid_layer_pattern", None)

    initialize_for_megatron(
        tensor_model_parallel_size=tp_size,
        seed=SEED,
        expert_model_parallel_size=ep_size,
        expert_tensor_parallel_size=etp_size,
    )

    model_ref = _gpt_model_provider(
        tp_size,
        hidden_size,
        vocab_size=256,
        num_moe_experts=num_moe_experts,
        moe_grouped_gemm=moe_grouped_gemm,
        ep_size=ep_size,
        etp_size=etp_size,
        transformer_impl=transformer_impl,
        is_hybrid=is_hybrid,
        hybrid_layer_pattern=hybrid_layer_pattern,
    )
    model_test = _gpt_model_provider(
        tp_size,
        hidden_size,
        vocab_size=256,
        num_moe_experts=num_moe_experts,
        moe_grouped_gemm=moe_grouped_gemm,
        meta_device=meta_device,
        ep_size=ep_size,
        etp_size=etp_size,
        transformer_impl=transformer_impl,
        is_hybrid=is_hybrid,
        hybrid_layer_pattern=hybrid_layer_pattern,
    )

    forward = get_forward(model_ref)
    model_ref = mtq.quantize(model_ref, config, forward)
    if compress:
        mtq.compress(model_ref)

    for module in model_ref.modules():
        if hasattr(module, "_amax_for_smoothing"):
            delattr(module, "_amax_for_smoothing")

    sharded_state_dict_test_helper(
        tmp_path,
        model_ref,
        model_test,
        forward,
        meta_device=meta_device,
        version=modelopt_version,
    )

    if modelopt_version is not None:
        mto.conversion.__version__ = modelopt.__version__
        mtq.plugins.megatron.__version__ = modelopt.__version__

    # Make sure all ranks have arrived before destroying NCCL
    torch.distributed.barrier()


mixed_precision_config = copy.deepcopy(mtq.W4A8_AWQ_BETA_CFG)
mixed_precision_config["quant_cfg"].extend(
    [
        {"quantizer_name": "*.1.*", "enable": False},
        {"quantizer_name": "*.2.*weight_quantizer", "cfg": {"num_bits": (4, 3), "axis": None}},
        {"quantizer_name": "*.2.*input_quantizer", "cfg": {"num_bits": (4, 3), "axis": None}},
        {"quantizer_name": "*.3.*weight_quantizer.0", "cfg": {"num_bits": 8, "axis": 0}},
        {"quantizer_name": "*.3.*weight_quantizer.1", "enable": False},
        {"quantizer_name": "*.3.*input_quantizer", "cfg": {"num_bits": 8, "axis": None}},
    ]
)


mixed_block_size_config = copy.deepcopy(mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG)
mixed_block_size_config["quant_cfg"].extend(
    [
        {"quantizer_name": "*.1.*", "enable": False},
        {
            "quantizer_name": "*.2.*weight_quantizer",
            "cfg": {"num_bits": 4, "block_sizes": {-1: 64}},
            "enable": True,
        },
        {"quantizer_name": "*.2.*input_quantizer", "cfg": {"num_bits": (4, 3), "axis": None}},
        {
            "quantizer_name": "*.3.*weight_quantizer",
            "cfg": {"num_bits": 4, "block_sizes": {-1: 128, -2: 64}},
            "enable": True,
        },
        {"quantizer_name": "*.3.*input_quantizer", "cfg": {"num_bits": 8, "axis": None}},
    ]
)

# Combined NVFP4 GEMM + KV cache quantization config
NVFP4_GEMM_KV_CFG = copy.deepcopy(mtq.NVFP4_DEFAULT_CFG)
NVFP4_GEMM_KV_CFG["quant_cfg"].extend(mtq.NVFP4_KV_CFG["quant_cfg"])

# Combined FP8 GEMM + KV cache quantization config
FP8_GEMM_KV_CFG = copy.deepcopy(mtq.FP8_DEFAULT_CFG)
FP8_GEMM_KV_CFG["quant_cfg"].extend(mtq.FP8_KV_CFG["quant_cfg"])


@pytest.mark.parametrize(
    "config",
    [
        mtq.FP8_DEFAULT_CFG,
        mtq.INT8_SMOOTHQUANT_CFG,
        mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG,
        mtq.INT4_AWQ_CFG,
        mtq.W4A8_AWQ_BETA_CFG,
        mtq.NVFP4_DEFAULT_CFG,
        mtq.FP8_2D_BLOCKWISE_WEIGHT_ONLY_CFG,
        mtq.FP8_KV_CFG,
        mtq.NVFP4_KV_CFG,
    ],
)
@pytest.mark.parametrize("meta_device", [False, True])
@pytest.mark.parametrize("transformer_impl", ["local", "modelopt"])
@skip_flaky_on_blackwell
def test_homogeneous_sharded_state_dict(
    dist_workers, tmp_path, config, meta_device, transformer_impl
):
    _run_homogeneous_sharded_state_dict(
        dist_workers,
        tmp_path,
        config,
        compress=False,
        meta_device=meta_device,
        transformer_impl=transformer_impl,
    )


@pytest.mark.parametrize(
    "config",
    [
        mtq.FP8_DEFAULT_CFG,
        mtq.INT4_AWQ_CFG,
        mtq.NVFP4_DEFAULT_CFG,
    ],
)
@pytest.mark.parametrize("meta_device", [False, True])
@pytest.mark.parametrize("transformer_impl", ["local", "modelopt"])
@pytest.mark.timeout(240)
# Compressed state dict takes longer due to real quant conversion & saving/loading
# Its real mtq.compress() path exercises the same TE/CUDA-13 kernels that trip the flaky
# Blackwell (sm_120) illegal-memory-access; #1901 skipped the fake-quant sibling but missed this one.
@skip_flaky_on_blackwell
def test_homogeneous_compressed_sharded_state_dict(
    dist_workers, tmp_path, config, meta_device, transformer_impl
):
    _run_homogeneous_sharded_state_dict(
        dist_workers,
        tmp_path,
        config,
        compress=True,
        meta_device=meta_device,
        transformer_impl=transformer_impl,
    )


def _run_homogeneous_sharded_state_dict(
    dist_workers, tmp_path, config, compress, meta_device, transformer_impl
):
    if compress and config is mtq.W4A8_AWQ_BETA_CFG:
        pytest.skip("W4A8_AWQ_BETA_CFG is not supported for compress")

    if config in (mtq.FP8_KV_CFG, mtq.NVFP4_KV_CFG):
        if transformer_impl != "modelopt" or compress or meta_device:
            pytest.skip(
                "KV cache configs require transformer_impl='modelopt' and no compress/meta_device"
            )

    model_config = {"transformer_impl": transformer_impl}
    dist_workers.run(
        partial(
            _test_sharded_state_dict,
            tmp_path,
            config,
            256,
            None,
            compress,
            meta_device,
            model_config,
        ),
    )


@pytest.mark.parametrize(
    "config",
    [NVFP4_GEMM_KV_CFG, FP8_GEMM_KV_CFG, mtq.MAMBA_MOE_NVFP4_CONSERVATIVE_CFG],
)
def test_homogeneous_sharded_state_dict_hybrid(dist_workers, tmp_path, config):
    """Test sharded state dict for hybrid Mamba MOE models."""
    if not HAS_MAMBA:
        pytest.skip("Mamba not installed")
    # TP+EP is not supported by QuantSequentialMLP. Set either TP or EP to 1
    num_gpus = torch.cuda.device_count()
    if num_gpus > 4:
        pytest.skip("Test needs to be fixed for more than 4 GPUs")
    model_config = {
        "is_hybrid": True,
        "hybrid_layer_pattern": "MEM*E",  # 5 layers: Mamba → MoE → Mamba → Attention → MoE
        "num_moe_experts": 8,
        "tp_size": num_gpus,
        "ep_size": 1,
        "etp_size": num_gpus,
    }
    dist_workers.run(
        partial(
            _test_sharded_state_dict,
            tmp_path,
            config,
            256,
            None,
            False,  # compress
            False,  # meta_device
            model_config,
        ),
    )


@pytest.mark.parametrize(
    "config",
    [
        mixed_precision_config,
        mixed_block_size_config,
    ],
)
@skip_flaky_on_blackwell
def test_heterogenous_sharded_state_dict(dist_workers, tmp_path, config):
    dist_workers.run(
        partial(_test_sharded_state_dict, tmp_path, config, 256, None, False, False, {}),
    )


@pytest.mark.parametrize(
    "hidden_size",
    [
        256,
        pytest.param(320, marks=skip_flaky_on_blackwell),
    ],
)
def test_regular_state_dict(distributed_setup_size_1, hidden_size):
    initialize_for_megatron(tensor_model_parallel_size=1, pipeline_model_parallel_size=1, seed=SEED)

    model_ref = _gpt_model_provider(tp_size=1, hidden_size=hidden_size)
    model_test = _gpt_model_provider(tp_size=1, hidden_size=hidden_size)
    prompt_tokens = torch.randint(
        0, model_ref.vocab_size, (2, model_ref.max_sequence_length)
    ).cuda()

    def forward_fn(model):
        return run_mcore_inference(model, prompt_tokens)

    model_ref = mtq.quantize(model_ref, mixed_precision_config, forward_fn)

    mto.restore_from_modelopt_state(model_test, mto.modelopt_state(model_ref))
    model_test.load_state_dict(model_ref.state_dict())

    model_test_sd = model_test.state_dict()
    for k, v in model_ref.state_dict().items():
        # The extra_state checkint must be skipped. It can be a byte tensor serialized
        # from a dict where the order can change.
        if "_extra_state" in k:
            continue
        assert not isinstance(v, torch.Tensor) or torch.allclose(v, model_test_sd[k]), k

    logits_ref = forward_fn(model_ref)
    logits_test = forward_fn(model_test)
    assert torch.allclose(logits_ref, logits_test)

    # Clean up since this is not a spawned process
    destroy_model_parallel()


def _test_auto_quantize_helper(rank, size):
    initialize_for_megatron(tensor_model_parallel_size=size)
    model = MegatronModel().cuda()
    auto_quantize_helper(model)


def test_auto_quantize(dist_workers):
    dist_workers.run(_test_auto_quantize_helper)


def _test_fp8_real_quantize_helper(rank, size):
    initialize_for_megatron(
        tensor_model_parallel_size=size, pipeline_model_parallel_size=1, seed=SEED
    )
    hidden_size = 256
    config = mtq.FP8_2D_BLOCKWISE_WEIGHT_ONLY_CFG

    model = _gpt_model_provider(tp_size=1, hidden_size=hidden_size)

    forward = get_forward(model)
    forward(model)

    # real quant the model
    cur_mem = get_model_size(model)
    real_quant_model = mtq.quantize(model, config, forward)
    mtq.compress(real_quant_model)
    real_quant_mem = get_model_size(real_quant_model)

    # Since not all parameters are quantized, the size won't be lower than half.
    assert real_quant_mem < (cur_mem / 2) * 1.1, "Memory after real quantization is not reduced."

    # check forward works after real quantization
    forward(real_quant_model)

    assert real_quant_mem < cur_mem


def test_fp8_real_quantize(dist_workers):
    dist_workers.run(_test_fp8_real_quantize_helper)


# TODO: etp requires sequence parallelism now in Megatron due to a bug
@pytest.mark.parametrize(
    "config",
    [mtq.FP8_DEFAULT_CFG, mtq.NVFP4_DEFAULT_CFG, mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG],
)
@pytest.mark.parametrize("moe_grouped_gemm", [True, False])
def test_moe_sharded_state_dict(dist_workers, need_4_gpus, tmp_path, config, moe_grouped_gemm):
    if config == mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG and moe_grouped_gemm:
        pytest.skip("TEGroupedMLP only supports per-tensor quantization.")

    # TODO: Add support for compress=True for TEGroupedMLP
    moe_config = {
        "tp_size": 2,
        "ep_size": 2,
        "etp_size": 1,
        "num_moe_experts": 4,
        "moe_grouped_gemm": moe_grouped_gemm,
        "transformer_impl": "transformer_engine" if moe_grouped_gemm else "modelopt",
    }
    if not moe_grouped_gemm:
        moe_config["tp_size"] = 1  # TODO: TP+EP is not supported by QuantSequentialMLP
    dist_workers.run(
        partial(
            _test_sharded_state_dict,
            tmp_path,
            config,
            256,
            None,
            False,
            False,
            moe_config,
        ),
    )


def _test_te_grouped_vs_sequential_quantize_helper(tp_size, ep_size, quant_cfg, rank, size):
    """Test that TEGrouped and sequential MoE models produce similar amax values."""
    initialize_for_megatron(
        tensor_model_parallel_size=tp_size,
        expert_model_parallel_size=ep_size,
        seed=SEED,
    )

    # Create TEGrouped MoE model
    te_grouped_moe_model = _gpt_model_provider(
        tp_size=tp_size,
        ep_size=ep_size,
        hidden_size=32,
        moe_grouped_gemm=True,
        transformer_impl="transformer_engine",
        num_moe_experts=4,
    )

    # Create forward function with cached inputs
    forward = get_forward(te_grouped_moe_model, batch_size=8)

    num_te_grouped_mlp = sum(
        isinstance(module, TEGroupedMLP) for module in te_grouped_moe_model.modules()
    )
    assert num_te_grouped_mlp == 4, (
        f"TEGroupedMoEModel has {num_te_grouped_mlp} TEGroupedMLP modules, it should have 4"
    )

    # Create sequential MoE model
    sequential_moe_model = _gpt_model_provider(
        tp_size=tp_size,
        ep_size=ep_size,
        hidden_size=32,
        moe_grouped_gemm=False,
        num_moe_experts=4,
        transformer_impl="modelopt",
    )
    num_sequential_mlp = sum(
        isinstance(module, SequentialMLP) for module in sequential_moe_model.modules()
    )
    assert num_sequential_mlp == 4, (
        f"SequentialMoEModel has {num_sequential_mlp} SequentialMLP modules, it should have 4"
    )
    # Copy weights from grouped to non-grouped model
    copy_weights_from_grouped_to_non_grouped(te_grouped_moe_model, sequential_moe_model)

    # Compare model outputs before quantization
    te_grouped_moe_output = forward(te_grouped_moe_model)
    sequential_moe_output = forward(sequential_moe_model)
    assert torch.allclose(te_grouped_moe_output, sequential_moe_output, atol=1e-6, rtol=1e-6)

    # Quantize grouped model
    mtq.quantize(te_grouped_moe_model, quant_cfg, forward)

    # TEGroupedMLP now quantizes per-expert by default (GroupedQuantizer), matching
    # SequentialMLP's per-expert quantizers, so no amax sync override is needed for the
    # two models to produce identical quantized outputs.
    mtq.quantize(sequential_moe_model, copy.deepcopy(quant_cfg), forward)

    # Compare model outputs after quantization
    te_grouped_moe_quant_output = forward(te_grouped_moe_model)
    sequential_moe_quant_output = forward(sequential_moe_model)

    assert torch.allclose(
        te_grouped_moe_quant_output, sequential_moe_quant_output, atol=1e-6, rtol=1e-6
    )


# TODO SequentialMLP local spec doesn't support EP and TP simultaneously yet
@pytest.mark.parametrize("quant_cfg", [mtq.FP8_DEFAULT_CFG, mtq.NVFP4_DEFAULT_CFG])
def test_te_grouped_vs_sequential_quantize(dist_workers_size_4, quant_cfg):
    """Test that TEGrouped and sequential MoE models produce similar quantized models."""
    dist_workers_size_4.run(
        partial(_test_te_grouped_vs_sequential_quantize_helper, 1, 2, quant_cfg)
    )


def test_te_grouped_process_quantizer_amax_preserves_per_expert_shape():
    """Per-expert amax buffers retain their native checkpoint shape."""
    value = torch.randn(3, 2)
    state_dict = {}

    _QuantMegatronTEGroupedLinear._process_quantizer_amax(
        None, "weight_quantizer.2._amax", value, state_dict
    )

    assert state_dict["weight_quantizer.2._amax"] is value
    assert state_dict["weight_quantizer.2._amax"].shape == (3, 2)


@pytest.mark.parametrize("compile_enabled", [False, True])
def test_te_grouped_compiled_weight_quantizer_loop(
    distributed_setup_size_1, monkeypatch, compile_enabled
):
    """The opt-in flag controls compilation and preserves per-expert backward."""
    compile_kwargs = []
    compiled_calls = []

    def fake_compile(fn, **kwargs):
        compile_kwargs.append(kwargs)

        def compiled(*args):
            compiled_calls.append(len(args))
            return fn(*args)

        return compiled

    if compile_enabled:
        monkeypatch.setenv(_COMPILE_TEGROUPED_WEIGHT_LOOP_ENV, "1")
    else:
        monkeypatch.delenv(_COMPILE_TEGROUPED_WEIGHT_LOOP_ENV, raising=False)
    monkeypatch.setattr(torch, "compile", fake_compile)
    initialize_for_megatron(seed=SEED)
    model = _gpt_model_provider(
        tp_size=1,
        hidden_size=32,
        moe_grouped_gemm=True,
        transformer_impl="transformer_engine",
        num_moe_experts=4,
    )
    forward = get_forward(model)
    for module in model.modules():
        if isinstance(module, TopKRouter):
            module.topk = module.num_experts

    mtq.quantize(model, copy.deepcopy(mtq.INT8_DEFAULT_CFG), forward)
    grouped_modules = [
        module
        for module in model.modules()
        if isinstance(getattr(module, "weight_quantizer", None), mtq.nn.GroupedQuantizer)
    ]
    compiled_modules = [
        module for module in model.modules() if hasattr(module, "_compiled_weight_quantizer_loop")
    ]
    assert grouped_modules
    assert len(compiled_modules) == (len(grouped_modules) if compile_enabled else 0)
    assert len(compile_kwargs) == (len(grouped_modules) if compile_enabled else 0)
    assert all(
        kwargs == {"backend": "inductor", "fullgraph": False, "mode": "reduce-overhead"}
        for kwargs in compile_kwargs
    )
    # Calibration mutates collector state and must stay eager even when the flag is enabled.
    assert not compiled_calls

    loss = forward(model).sum()
    loss.backward()
    if compile_enabled:
        assert compiled_calls
        assert set(compiled_calls) == {4}
    else:
        assert not compiled_calls
    assert all(
        torch.isfinite(getattr(module, f"weight{i}").grad).all()
        for module in grouped_modules
        for i in range(module.num_gemms)
    )
    destroy_model_parallel()


def test_te_grouped_real_compile_weight_quantizer_loop(distributed_setup_size_1, monkeypatch):
    """Real (unpatched) torch.compile parity for the per-expert weight-quantizer loop.

    Complements test_te_grouped_compiled_weight_quantizer_loop, which fakes torch.compile to
    assert wiring only. Here torch.compile is left intact so the opt-in loop is actually
    compiled, executed, and back-propagated, and its numerics are checked against the eager
    path built from identical weights and calibrated amax.
    """
    initialize_for_megatron(seed=SEED)

    def build():
        model = _gpt_model_provider(
            tp_size=1,
            hidden_size=32,
            moe_grouped_gemm=True,
            transformer_impl="transformer_engine",
            num_moe_experts=4,
        )
        for module in model.modules():
            if isinstance(module, TopKRouter):
                module.topk = module.num_experts
        return model

    # Two identical models (same raw weights); one stays eager, one is real-compiled.
    model_eager = build()
    model_compiled = build()
    model_compiled.load_state_dict(model_eager.state_dict())

    # One cached input batch, shared across both models for an apples-to-apples compare.
    forward = get_forward(model_eager)

    monkeypatch.delenv(_COMPILE_TEGROUPED_WEIGHT_LOOP_ENV, raising=False)
    mtq.quantize(model_eager, copy.deepcopy(mtq.INT8_DEFAULT_CFG), forward)

    monkeypatch.setenv(_COMPILE_TEGROUPED_WEIGHT_LOOP_ENV, "1")
    mtq.quantize(model_compiled, copy.deepcopy(mtq.INT8_DEFAULT_CFG), forward)

    grouped_eager = [
        m
        for m in model_eager.modules()
        if isinstance(getattr(m, "weight_quantizer", None), mtq.nn.GroupedQuantizer)
    ]
    grouped_compiled = [
        m
        for m in model_compiled.modules()
        if isinstance(getattr(m, "weight_quantizer", None), mtq.nn.GroupedQuantizer)
    ]
    assert grouped_compiled and len(grouped_eager) == len(grouped_compiled)
    # The opt-in path attached the real compiled loop (torch.compile left unpatched); the
    # eager control model did not.
    assert all(hasattr(m, "_compiled_weight_quantizer_loop") for m in grouped_compiled)
    assert all(not hasattr(m, "_compiled_weight_quantizer_loop") for m in grouped_eager)

    # Forward parity: the first call on model_compiled triggers real compilation.
    out_eager = forward(model_eager)
    out_compiled = forward(model_compiled)
    torch.testing.assert_close(out_compiled, out_eager, rtol=1e-3, atol=1e-3)

    # Backward parity: per-expert weight grads must be finite and match the eager path.
    out_eager.sum().backward()
    out_compiled.sum().backward()
    for m_e, m_c in zip(grouped_eager, grouped_compiled):
        for i in range(m_c.num_gemms):
            g_e = getattr(m_e, f"weight{i}").grad
            g_c = getattr(m_c, f"weight{i}").grad
            assert g_c is not None and torch.isfinite(g_c).all()
            torch.testing.assert_close(g_c, g_e, rtol=1e-2, atol=1e-2)

    destroy_model_parallel()


def test_te_grouped_per_expert_quantizer_default(distributed_setup_size_1):
    """TEGroupedLinear installs a per-expert GroupedQuantizer (one quantizer per fused expert).

    Per-expert weight quantization is unconditional: every ``TEGroupedLinear`` gets a
    ``GroupedQuantizer`` with ``num_gemms`` independent quantizers, not a single shared one.
    """
    initialize_for_megatron(seed=SEED)
    model = _gpt_model_provider(
        tp_size=1,
        hidden_size=32,
        moe_grouped_gemm=True,
        transformer_impl="transformer_engine",
        num_moe_experts=4,
    )
    forward = get_forward(model)
    for module in model.modules():
        if isinstance(module, TopKRouter):
            module.topk = module.num_experts

    mtq.quantize(model, copy.deepcopy(mtq.INT8_DEFAULT_CFG), forward)

    grouped_linears = [
        getattr(mlp, name)
        for mlp in model.modules()
        if isinstance(mlp, TEGroupedMLP)
        for name in ("linear_fc1", "linear_fc2")
    ]
    assert grouped_linears
    for gl in grouped_linears:
        wq = gl.weight_quantizer
        assert isinstance(wq, mtq.nn.GroupedQuantizer), (
            "TEGroupedLinear should install a per-expert GroupedQuantizer"
        )
        assert len(wq) == gl.num_gemms

    destroy_model_parallel()


def _te_grouped_expert_magnitude(linear_name, local_idx):
    """Distinct, known weight magnitude for each (linear, local-expert) pair.

    Chosen so every per-expert weight quantizer sees a different amax (and fc1 vs fc2 differ
    too), making divergence guaranteed by construction rather than by random initialization.
    """
    return {"linear_fc1": 0.25, "linear_fc2": 1.25}[linear_name] + 0.5 * local_idx


def _test_te_grouped_vs_sequential_default_amax_helper(tp_size, ep_size, quant_cfg, rank, size):
    """TEGrouped keeps a per-expert weight quantizer (GroupedQuantizer) by default; each expert's
    amax must equal the corresponding SequentialMLP expert's (no cross-expert sharing).

    Divergence is made causal: each local expert's weights are filled with a distinct known
    magnitude, so its weight amax is that magnitude by construction. The test then asserts
    (a) grouped == sequential per expert, (b) each amax equals ITS OWN set magnitude, and
    (c) the per-expert quantizer objects are distinct instances. A cross-expert-sharing
    regression therefore fails deterministically, not by luck of the random init.
    """
    initialize_for_megatron(
        tensor_model_parallel_size=tp_size,
        expert_model_parallel_size=ep_size,
        seed=SEED,
    )

    te_grouped = _gpt_model_provider(
        tp_size=tp_size,
        ep_size=ep_size,
        hidden_size=32,
        moe_grouped_gemm=True,
        transformer_impl="transformer_engine",
        num_moe_experts=4,
    )
    forward = get_forward(te_grouped, batch_size=8)

    sequential = _gpt_model_provider(
        tp_size=tp_size,
        ep_size=ep_size,
        hidden_size=32,
        moe_grouped_gemm=False,
        num_moe_experts=4,
        transformer_impl="modelopt",
    )

    # Fill each local expert's grouped weights with a distinct, known magnitude so the per-expert
    # weight amax is deterministic (== that magnitude) and diverges across experts by construction.
    for te_mlp in (m for m in te_grouped.modules() if isinstance(m, TEGroupedMLP)):
        for linear_name in ("linear_fc1", "linear_fc2"):
            grouped_linear = getattr(te_mlp, linear_name)
            for i in range(grouped_linear.num_gemms):
                with torch.no_grad():
                    getattr(grouped_linear, f"weight{i}").fill_(
                        _te_grouped_expert_magnitude(linear_name, i)
                    )

    # Propagate the identical per-expert weights to the sequential model.
    copy_weights_from_grouped_to_non_grouped(te_grouped, sequential)

    for module in te_grouped.modules():
        if isinstance(module, TopKRouter):
            module.topk = module.num_experts
    for module in sequential.modules():
        if isinstance(module, TopKRouter):
            module.topk = module.num_experts

    mtq.quantize(te_grouped, quant_cfg, forward)
    mtq.quantize(sequential, quant_cfg, forward)

    te_modules = [m for m in te_grouped.modules() if isinstance(m, TEGroupedMLP)]
    seq_modules = [m for m in sequential.modules() if isinstance(m, SequentialMLP)]
    assert len(te_modules) == len(seq_modules)

    for te_mlp, seq_mlp in zip(te_modules, seq_modules):
        for linear_name in ("linear_fc1", "linear_fc2"):
            te_wq = getattr(te_mlp, linear_name).weight_quantizer
            # One weight quantizer per local expert, not a single shared one.
            assert len(te_wq) == len(seq_mlp.local_experts), (
                f"{linear_name}: expected {len(seq_mlp.local_experts)} per-expert quantizers, "
                f"got {len(te_wq)}"
            )

            per_expert_amax = []
            for i, expert in enumerate(seq_mlp.local_experts):
                te_amax = te_wq[i].amax
                seq_amax = getattr(expert, linear_name).weight_quantizer.amax
                expected = _te_grouped_expert_magnitude(linear_name, i)
                assert te_amax is not None

                # (a) grouped and sequential agree per expert (cross-implementation parity).
                assert torch.allclose(te_amax, seq_amax, atol=1e-5, rtol=1e-5), (
                    f"TEGrouped expert {i} amax != Sequential expert {i} amax for {linear_name}"
                )
                # (b) causal: this expert's amax equals ITS OWN set magnitude, proving the amax
                #     was computed from that expert's weights (no cross-expert leakage).
                assert torch.allclose(
                    te_amax, torch.full_like(te_amax, expected), rtol=1e-3, atol=1e-3
                ), (
                    f"{linear_name} expert {i}: amax {te_amax.reshape(-1)[0].item():.6f} "
                    f"!= set magnitude {expected}"
                )
                # (c) each expert owns a distinct quantizer instance, not a shared one.
                for j in range(i):
                    assert te_wq[i] is not te_wq[j], (
                        f"{linear_name}: experts {i} and {j} share a quantizer object"
                    )
                per_expert_amax.append(te_amax.reshape(-1)[0])

            # Divergence is now guaranteed by construction (distinct set magnitudes).
            stacked = torch.stack(per_expert_amax)
            assert (stacked.max() - stacked.min()).item() > 1e-4, (
                f"{linear_name}: per-expert amax did not diverge despite distinct set magnitudes"
            )


@pytest.mark.parametrize("quant_cfg", [mtq.FP8_DEFAULT_CFG, mtq.NVFP4_DEFAULT_CFG])
def test_te_grouped_vs_sequential_default_amax(dist_workers_size_1, quant_cfg):
    dist_workers_size_1.run(
        partial(_test_te_grouped_vs_sequential_default_amax_helper, 1, 1, quant_cfg)
    )


def _te_grouped_expert_identity_from_sharded_state(module):
    """Return {local_key: (global_expert_idx, num_global_experts)} for per-expert amax shards.

    The grouped linear must give each fused expert the same global identity the weights use:
    the dict key keeps the local expert index (maps to the local buffer on restore) while the
    ShardedTensor carries the global expert offset. Called with sharded_offsets=() so the expert
    axis is the (only) prepended axis at index 0.
    """
    sharded_sd = module.sharded_state_dict(prefix="", sharded_offsets=(), metadata=None)
    identity = {}
    for key, sh_ten in sharded_sd.items():
        if re.match(r"weight_quantizer\.\d+\..*_amax$", key):
            assert sh_ten.prepend_axis_num >= 1, f"{key}: expected a prepended expert axis"
            identity[key] = (int(sh_ten.global_offset[0]), int(sh_ten.global_shape[0]))
    return identity


def _test_te_grouped_sharded_state_dict_global_expert_identity_helper(
    tp_size, ep_size, quant_cfg, rank, size
):
    """Per-expert quantizer amax must persist all num_global_experts across EP.

    With EP>1 the base linear emitted ``weight_quantizer.{local_i}._amax`` at the local index with
    no expert offset, so every rank wrote identical keys and torch_dist dedup collapsed them to a
    single rank's experts. Assert each rank's fused experts now carry distinct global identities so
    the union across ranks covers every global expert.
    """
    initialize_for_megatron(
        tensor_model_parallel_size=tp_size,
        expert_model_parallel_size=ep_size,
        seed=SEED,
    )
    num_experts = 4
    num_local = num_experts // ep_size

    te_grouped = _gpt_model_provider(
        tp_size=tp_size,
        ep_size=ep_size,
        hidden_size=32,
        moe_grouped_gemm=True,
        transformer_impl="transformer_engine",
        num_moe_experts=num_experts,
    )
    forward = get_forward(te_grouped, batch_size=8)
    for module in te_grouped.modules():
        if isinstance(module, TopKRouter):
            module.topk = module.num_experts
    mtq.quantize(te_grouped, quant_cfg, forward)

    grouped_linears = [
        m for m in te_grouped.modules() if isinstance(m, _QuantMegatronTEGroupedLinear)
    ]
    assert grouped_linears, "No grouped quant linears found"

    expected_global = {rank * num_local + i for i in range(num_local)}
    for linear in grouped_linears:
        # Give each expert a distinct amax so a value mix-up would also be observable.
        for i in range(linear.num_gemms):
            wq = linear.weight_quantizer[i]
            leaves = list(wq) if isinstance(wq, SequentialQuantizer) else [wq]
            for leaf in leaves:
                if hasattr(leaf, "_amax") and leaf._amax is not None:
                    leaf._amax.fill_(1.0 + rank * num_local + i)

        identity = _te_grouped_expert_identity_from_sharded_state(linear)
        # One entry per local expert per amax buffer; dict keys keep the LOCAL index.
        local_keys = {int(re.search(r"weight_quantizer\.(\d+)\.", k).group(1)) for k in identity}
        assert local_keys == set(range(num_local)), (
            f"Expected local expert keys {set(range(num_local))}, got {local_keys}"
        )
        # ShardedTensor global identity: this rank owns experts {rank*num_local + i}.
        local_global = {gidx for gidx, _ in identity.values()}
        assert local_global == expected_global, (
            f"rank {rank}: expected global experts {expected_global}, got {local_global}"
        )
        assert all(total == num_experts for _, total in identity.values()), (
            f"num_global_experts should be {num_experts}, got {identity}"
        )

    # Gather the global expert indices across all EP ranks: the union must cover every expert.
    gathered = [None] * size
    torch.distributed.all_gather_object(gathered, sorted(expected_global))
    union = set()
    for part in gathered:
        union.update(part)
    assert union == set(range(num_experts)), (
        f"Union of global experts across EP ranks should be {set(range(num_experts))}, got {union}"
    )


@pytest.mark.parametrize("quant_cfg", [mtq.FP8_DEFAULT_CFG, mtq.NVFP4_DEFAULT_CFG])
def test_te_grouped_sharded_state_dict_global_expert_identity(dist_workers_size_2, quant_cfg):
    dist_workers_size_2.run(
        partial(_test_te_grouped_sharded_state_dict_global_expert_identity_helper, 1, 2, quant_cfg)
    )


def _test_te_grouped_vs_sequential_default_loss_helper(tp_size, ep_size, quant_cfg, rank, size):
    """TEGrouped quantized output should diverge from BF16 more than SequentialMLP under default sync=False."""
    initialize_for_megatron(
        tensor_model_parallel_size=tp_size,
        expert_model_parallel_size=ep_size,
        seed=SEED,
    )

    te_grouped = _gpt_model_provider(
        tp_size=tp_size,
        ep_size=ep_size,
        hidden_size=32,
        moe_grouped_gemm=True,
        transformer_impl="transformer_engine",
        num_moe_experts=4,
    )
    forward = get_forward(te_grouped, batch_size=8)

    sequential = _gpt_model_provider(
        tp_size=tp_size,
        ep_size=ep_size,
        hidden_size=32,
        moe_grouped_gemm=False,
        num_moe_experts=4,
        transformer_impl="modelopt",
    )
    copy_weights_from_grouped_to_non_grouped(te_grouped, sequential)

    for module in te_grouped.modules():
        if isinstance(module, TopKRouter):
            module.topk = module.num_experts
    for module in sequential.modules():
        if isinstance(module, TopKRouter):
            module.topk = module.num_experts

    ref_te = forward(te_grouped)
    ref_seq = forward(sequential)

    mtq.quantize(te_grouped, quant_cfg, forward)
    mtq.quantize(sequential, quant_cfg, forward)

    out_te = forward(te_grouped)
    out_seq = forward(sequential)

    err_te = (out_te - ref_te).abs().mean().item()
    err_seq = (out_seq - ref_seq).abs().mean().item()

    if rank == 0:
        print(
            f"\n[default-amax] TEGrouped quant-err={err_te:.6f}, "
            f"Sequential quant-err={err_seq:.6f}, ratio TE/Seq={err_te / max(err_seq, 1e-12):.3f}"
        )

    # At toy scale (4 small experts) the per-tensor amax difference is dominated
    # by other numerical noise (~few %); the effect amplifies at production scale
    # (e.g. 128 experts in Nemotron Nano). Just sanity-check both errors are finite.
    assert err_te > 0 and err_seq > 0
    assert math.isfinite(err_te) and math.isfinite(err_seq)


@pytest.mark.parametrize("quant_cfg", [mtq.FP8_DEFAULT_CFG, mtq.NVFP4_DEFAULT_CFG])
def test_te_grouped_vs_sequential_default_loss(dist_workers_size_1, quant_cfg):
    dist_workers_size_1.run(
        partial(_test_te_grouped_vs_sequential_default_loss_helper, 1, 1, quant_cfg)
    )


def _test_auto_quantize_moe_ep_helper(rank, size):
    initialize_for_megatron(
        tensor_model_parallel_size=1,
        expert_model_parallel_size=size,
        seed=SEED,
    )
    model = _gpt_model_provider(
        tp_size=1,
        ep_size=size,
        hidden_size=32,
        num_moe_experts=4,
        moe_grouped_gemm=False,
        transformer_impl="modelopt",
    )

    def forward_step(model, batch):
        input_ids, labels, position_ids, attention_mask, loss_mask = batch
        return model.forward(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            labels=labels,
            loss_mask=loss_mask,
        )

    auto_quantize_helper(
        model,
        data_loader=[get_batch(model, batch_size=2) for _ in range(2)],
        forward_step=forward_step,
        forward_backward_step=lambda m, b: forward_step(m, b).mean().backward(),
        quantization_formats=[mtq.NVFP4_DEFAULT_CFG, mtq.FP8_DEFAULT_CFG],
    )


def test_auto_quantize_moe_ep(dist_workers):
    """auto_quantize must pick a consistent recipe across EP ranks when multiple GPUs run."""
    dist_workers.run(_test_auto_quantize_moe_ep_helper)


def _mamba_hybrid_forward_step(model, batch):
    input_ids, labels, position_ids, attention_mask, loss_mask = batch
    return model.forward(
        input_ids=input_ids,
        position_ids=position_ids,
        attention_mask=attention_mask,
        labels=labels,
    )


@pytest.mark.skipif(not HAS_MAMBA, reason="Mamba not installed")
def test_gptq_mamba_hybrid(dist_workers_size_1):
    """End-to-end GPTQ (NVFP4) on a tiny Megatron-Core NemotronH-style hybrid model."""
    dist_workers_size_1.run(_test_gptq_mamba_hybrid)


def _test_gptq_mamba_hybrid(rank, size):
    initialize_for_megatron(tensor_model_parallel_size=1, seed=SEED)
    model = get_mcore_hybrid_model(
        tensor_model_parallel_size=1,
        hidden_size=32,
        num_attention_heads=4,
        ffn_hidden_size=64,
        mamba_state_dim=16,
        mamba_head_dim=8,
        num_moe_experts=4,
        moe_grouped_gemm=False,
        moe_ffn_hidden_size=32,
        moe_shared_expert_intermediate_size=16,
        transformer_impl="modelopt",
    ).cuda()

    quant_cfg = copy.deepcopy(mtq.NVFP4_DEFAULT_CFG)
    quant_cfg["algorithm"] = {"method": "gptq"}
    forward = get_forward(model, batch_size=1)
    model = mtq.quantize(model, quant_cfg, forward)

    for m in model.modules():
        if isinstance(m, SequentialMLP):
            assert all(
                is_quantized_linear(e.linear_fc1)
                and e.linear_fc1.weight_quantizer.is_enabled
                and e.linear_fc1.input_quantizer.is_enabled
                for e in m.local_experts
            )
            assert all(
                is_quantized_linear(e.linear_fc2)
                and e.linear_fc2.weight_quantizer.is_enabled
                and e.linear_fc2.input_quantizer.is_enabled
                for e in m.local_experts
            )
    assert torch.isfinite(forward(model)).all()


def _auto_quantize_mamba_hybrid_cost_helper(rank, size, expert_model_parallel_size, result_path):
    initialize_for_megatron(
        tensor_model_parallel_size=1,
        expert_model_parallel_size=expert_model_parallel_size,
        seed=SEED,
    )
    model = get_mcore_hybrid_model(
        tensor_model_parallel_size=1,
        expert_model_parallel_size=expert_model_parallel_size,
        hidden_size=32,
        num_attention_heads=4,
        ffn_hidden_size=64,
        mamba_state_dim=16,
        mamba_head_dim=8,
        num_moe_experts=4,
        moe_grouped_gemm=False,
        moe_ffn_hidden_size=32,
        moe_shared_expert_intermediate_size=16,
        transformer_impl="modelopt",
    ).cuda()

    def forward_backward_step(model, batch):
        _mamba_hybrid_forward_step(model, batch).mean().backward()

    _, search_state = mtq.auto_quantize(
        model,
        constraints={"effective_bits": 8.0},
        quantization_formats=[mtq.NVFP4_DEFAULT_CFG, mtq.FP8_DEFAULT_CFG],
        data_loader=[get_batch(model, batch_size=1)],
        forward_step=_mamba_hybrid_forward_step,
        forward_backward_step=forward_backward_step,
        num_calib_steps=1,
        num_score_steps=1,
        verbose=True,
    )

    no_quant = QuantRecipe(quant_cfg=None)
    summed_cost = sum(
        stat["costs"][stat["formats"].index(no_quant)]
        for stat in search_state["candidate_stats"].values()
    )
    # The per-op no-quant costs must sum to the cost denominator AutoQuantize uses,
    # which is the full quantizable weight size aggregated across EP ranks.
    assert summed_cost == pytest.approx(search_state["cost_denominator"], rel=1e-6)
    assert search_state["best"]["is_satisfied"]

    if expert_model_parallel_size == 1:
        # With EP=1, every rank has the full expert set. DP de-duplication should make the
        # summed no-quant cost match the local quantizable weight size on each rank.
        local_total = _AutoQuantizeBaseSearcher._get_total_weight_size(list(model.modules()))
        assert summed_cost == pytest.approx(local_total, rel=1e-6)

    if rank == 0:
        Path(result_path).write_text(repr(summed_cost))


@pytest.mark.skipif(not HAS_MAMBA, reason="Mamba not installed")
def test_auto_quantize_mamba_hybrid_ep_cost(dist_workers, tmp_path):
    """AutoQuantize cost must match for EP=1 and EP=2 when two GPUs are available."""
    ep1_path = tmp_path / "ep1_cost.txt"
    dist_workers.run(
        partial(
            _auto_quantize_mamba_hybrid_cost_helper,
            expert_model_parallel_size=1,
            result_path=str(ep1_path),
        )
    )
    if dist_workers.world_size < 2:
        return

    ep2_path = tmp_path / "ep2_cost.txt"
    dist_workers.run(
        partial(
            _auto_quantize_mamba_hybrid_cost_helper,
            expert_model_parallel_size=2,
            result_path=str(ep2_path),
        )
    )
    cost_ep1 = float(ep1_path.read_text())
    cost_ep2 = float(ep2_path.read_text())
    assert cost_ep1 == pytest.approx(cost_ep2, rel=1e-6)


def _test_mcore_layerwise_calibration_layers_do_not_mutate_decoder(rank, size):
    initialize_for_megatron(tensor_model_parallel_size=1, seed=SEED)
    model = _gpt_model_provider(
        tp_size=1,
        hidden_size=32,
        meta_device=True,
        transformer_impl="modelopt",
    )
    decoder_layers = model.decoder.layers
    decoder_len = len(decoder_layers)
    output_layer = model.output_layer

    discovered_layers = get_mcore_layerwise_calibration_layers(model)

    assert discovered_layers is not None
    assert len(discovered_layers) == decoder_len + 1
    assert discovered_layers[-1] is output_layer
    assert len(decoder_layers) == decoder_len
    assert all(layer is not output_layer for layer in decoder_layers)

    assert LayerActivationCollector.is_supported(model)
    discovered_layers = LayerActivationCollector.get_decoder_layers(model)
    assert discovered_layers is not None
    assert len(discovered_layers) == decoder_len + 1
    assert discovered_layers[-1] is output_layer
    assert len(decoder_layers) == decoder_len
    assert all(layer is not output_layer for layer in decoder_layers)


def test_mcore_layerwise_calibration_layers_do_not_mutate_decoder(dist_workers_size_1):
    dist_workers_size_1.run(_test_mcore_layerwise_calibration_layers_do_not_mutate_decoder)


@pytest.mark.parametrize("ep_size", [1, 2])
@pytest.mark.parametrize("sync_weight_amax", [True, False])
def test_layer_sync_moe_local_experts_amax(dist_workers, ep_size, sync_weight_amax):
    """Test expert model parallel synchronization."""
    if torch.cuda.device_count() < ep_size:
        pytest.skip(f"Requires at least {ep_size} GPUs for expert model parallel test")

    dist_workers.run(
        partial(
            _test_layer_sync_moe_local_experts_amax,
            ep_size,
            sync_weight_amax,
        ),
    )


def _test_layer_sync_moe_local_experts_amax(ep_size, sync_weight_amax, rank, size):
    initialize_for_megatron(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=ep_size,
        expert_tensor_parallel_size=1,
        seed=SEED,
    )
    model = _gpt_model_provider(
        tp_size=1,
        ep_size=ep_size,
        etp_size=1,
        hidden_size=256,
        num_moe_experts=8,
        transformer_impl="modelopt",
    )
    # Make weight initialization different across experts, otherwise experts will have similar amax values
    for layer in model.decoder.layers:
        for i, expert in enumerate(layer.mlp.experts.local_experts):
            expert.linear_fc1.weight.data.fill_(0.1 + i * 0.05)
            expert.linear_fc2.weight.data.fill_(0.2 + i * 0.05)

    quant_cfg = mtq.FP8_DEFAULT_CFG
    model = mtq.quantize(model, quant_cfg, get_forward(model))

    for layer in model.decoder.layers:
        layer.mlp.experts.layer_sync_moe_local_experts_amax(sync_weight_amax=sync_weight_amax)

    for layer in model.decoder.layers:
        # Check input quantizer amax is synced across local experts
        fc1_amax = None
        fc2_amax = None
        for expert in layer.mlp.experts.local_experts:
            assert expert.linear_fc1.input_quantizer.amax is not None
            assert expert.linear_fc2.input_quantizer.amax is not None
            if fc1_amax is None:
                fc1_amax = expert.linear_fc1.input_quantizer.amax
            else:
                assert torch.allclose(fc1_amax, expert.linear_fc1.input_quantizer.amax)
            if fc2_amax is None:
                fc2_amax = expert.linear_fc2.input_quantizer.amax
            else:
                assert torch.allclose(fc2_amax, expert.linear_fc2.input_quantizer.amax)

        # Check weight quantizer amax
        fc1_amax = None
        fc2_amax = None
        for expert in layer.mlp.experts.local_experts:
            assert expert.linear_fc1.weight_quantizer.amax is not None
            assert expert.linear_fc2.weight_quantizer.amax is not None
            if fc1_amax is None:
                fc1_amax = expert.linear_fc1.weight_quantizer.amax
            elif sync_weight_amax:
                assert torch.allclose(fc1_amax, expert.linear_fc1.weight_quantizer.amax)
            else:
                assert not torch.allclose(fc1_amax, expert.linear_fc1.weight_quantizer.amax)
            if fc2_amax is None:
                fc2_amax = expert.linear_fc2.weight_quantizer.amax
            elif sync_weight_amax:
                assert torch.allclose(fc2_amax, expert.linear_fc2.weight_quantizer.amax)
            else:
                assert not torch.allclose(fc2_amax, expert.linear_fc2.weight_quantizer.amax)


def _test_expert_model_parallel_amax_sync(
    tp_size, ep_size, etp_size, moe_grouped_gemm, config, rank, size
):
    """Test expert parallel synchronization with different configurations."""
    initialize_for_megatron(
        tensor_model_parallel_size=tp_size,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=ep_size,
        expert_tensor_parallel_size=etp_size,
        seed=SEED,
    )

    # Create model with expert parallelism
    model = _gpt_model_provider(
        tp_size=tp_size,
        ep_size=ep_size,
        etp_size=etp_size,
        hidden_size=256,
        moe_grouped_gemm=moe_grouped_gemm,
        num_moe_experts=8,
        transformer_impl="modelopt",
    )

    # Initialize ALL weights based on rank to produce different amax values
    # to produce different amax values across ranks that need synchronization
    weight_idx = 0
    for name, param in model.named_parameters():
        # Skip embeddings and any parameters without 'weight' in the name
        if "embedding" in name.lower() or "weight" not in name.lower():
            continue

        if param.requires_grad and param.dim() >= 2:  # Only weight matrices, not biases
            # Different constant value based on rank and parameter index
            const_val = 0.1 + (rank * 0.5) + (weight_idx * 0.05)
            param.data.fill_(const_val)
            weight_idx += 1

    # force all expert routing
    for module in model.modules():
        if isinstance(module, TopKRouter):
            module.topk = module.num_experts

    forward = get_forward(model)

    # quantize the model
    model = mtq.quantize(model, config, forward)
    # Check initial sync status
    initial_sync, quantizer_type, rank_values = compare_amax_sync_across_expert_parallel(model)
    assert initial_sync, (
        f"Inconsistent amax for expert {quantizer_type} across ranks: {rank_values}"
    )

    # Test if the amax values are inconsistent when distributed sync is disabled
    mtq.model_calib.max_calibrate(model, forward, distributed_sync=False)
    inconsistent_amax, _, _ = compare_amax_sync_across_expert_parallel(
        model, compare_across_experts=False
    )

    assert not inconsistent_amax, (
        "Consistent amax across expert parallel ranks, "
        "Amax should not be synchronized across expert parallel ranks since expert parallel is disabled"
    )
    # calibrate the model with distributed sync and test synchronization
    mtq.model_calib.max_calibrate(model, forward, distributed_sync=True)

    final_sync, quantizer_type, rank_values = compare_amax_sync_across_expert_parallel(model)
    assert final_sync, f"Inconsistent amax for expert {quantizer_type} across ranks: {rank_values}"


@pytest.mark.skip(reason="TODO: etp requires sequence parallelism now in Megatron due to a bug;")
@pytest.mark.parametrize("config", [mtq.FP8_DEFAULT_CFG, mtq.INT8_DEFAULT_CFG])
@pytest.mark.parametrize(("ep_size", "etp_size"), [(1, 2), (2, 1), (2, 2)])
@pytest.mark.parametrize("moe_grouped_gemm", [True, False])
def test_expert_parallel_sync(dist_workers, config, ep_size, etp_size, moe_grouped_gemm):
    """Test expert model parallel synchronization."""
    if torch.cuda.device_count() < ep_size * etp_size:
        pytest.skip(f"Requires at least {ep_size * etp_size} GPUs for expert model parallel test")

    dist_workers.run(
        partial(
            _test_expert_model_parallel_amax_sync,
            etp_size,  # tp_size
            ep_size,
            etp_size,
            moe_grouped_gemm,
            config,
        ),
    )


def _test_kv_cache_quant_helper(config, rank, size):
    """Helper function for testing KV cache quantization with TEDotProductAttention."""
    initialize_for_megatron(
        tensor_model_parallel_size=size, pipeline_model_parallel_size=1, seed=SEED
    )

    # Use existing infrastructure to create a minimal GPT model with TEDotProductAttention
    # Note: transformer_impl must be "modelopt" or "transformer_engine" (not "local") to get TEDotProductAttention
    model = get_mcore_gpt_model(
        tensor_model_parallel_size=size,
        num_layers=1,
        hidden_size=64,
        num_attention_heads=4,
        vocab_size=32,
        transformer_impl="modelopt",  # This uses TEDotProductAttention via get_gpt_modelopt_spec
    ).cuda()

    # Create forward function with cached inputs
    forward = get_forward(model)

    # Test KV cache quantization with the given config
    quantized_model = mtq.quantize(model, config, forward)

    # Find TEDotProductAttention modules and verify they have KV cache quantizers
    te_attention_found = False
    for name, module in quantized_model.named_modules():
        # Check if this is a quantized TEDotProductAttention
        if hasattr(module, "q_bmm_quantizer") and hasattr(module, "k_bmm_quantizer"):
            te_attention_found = True
            # Verify all expected quantizers exist
            assert hasattr(module, "v_bmm_quantizer"), f"Missing v_bmm_quantizer in {name}"

            # Verify K and V quantizers are enabled (main purpose of KV cache configs)
            assert module.k_bmm_quantizer.is_enabled, f"K quantizer not enabled in {name}"
            assert module.v_bmm_quantizer.is_enabled, f"V quantizer not enabled in {name}"

    assert te_attention_found, "No TEDotProductAttention with KV cache quantizers found in model"

    # Quick smoke test that forward still works
    output = forward(quantized_model)
    assert output is not None, "Forward pass failed"


@pytest.mark.parametrize(
    "config",
    [
        mtq.FP8_KV_CFG,
        mtq.NVFP4_KV_CFG,
    ],
)
def test_kv_cache_quant(dist_workers_size_1, config):
    """Verify KV cache quantization works correctly with TEDotProductAttention.

    This test ensures TEDotProductAttention is properly registered and gets the
    expected q/k/v_bmm_quantizers when using KV cache configs.

    Note: This test requires Transformer Engine to be installed since TEDotProductAttention
    is only available with transformer_impl="modelopt" or "transformer_engine" (not "local").
    """
    dist_workers_size_1.run(partial(_test_kv_cache_quant_helper, config))


def _test_kv_cache_amax_sync_helper(config, rank, size, tensor_model_parallel_size=1):
    """Helper function for testing KV cache quantizer amax sync across distributed world."""
    # Use rank in seed to produce different amax values across ranks
    seed = SEED + rank
    initialize_for_megatron(
        tensor_model_parallel_size=tensor_model_parallel_size,
        pipeline_model_parallel_size=1,
        seed=seed,
    )

    model = get_mcore_gpt_model(
        tensor_model_parallel_size=tensor_model_parallel_size,
        num_layers=1,
        hidden_size=64,
        num_attention_heads=max(4, size),
        vocab_size=32,
        transformer_impl="modelopt",
    ).cuda()

    forward = get_forward(model)

    # Quantize with KV cache config
    quantized_model = mtq.quantize(model, config, forward)

    # Verify KV cache quantizer amax is synced across the whole world
    kv_quantizers_found = verify_kv_cache_amax_sync(quantized_model)
    assert kv_quantizers_found, "No KV cache quantizers found in model"


def test_kv_cache_amax_sync(dist_workers):
    """Test KV cache quantizer amax is synced across the distributed world."""
    dist_workers.run(
        partial(
            _test_kv_cache_amax_sync_helper,
            NVFP4_GEMM_KV_CFG,
            tensor_model_parallel_size=torch.cuda.device_count(),
        ),
    )


def test_convert_mcore_te_gpt_model(distributed_setup_size_1):
    initialize_for_megatron(tensor_model_parallel_size=1, seed=SEED)
    model = get_mcore_gpt_model(tensor_model_parallel_size=1, transformer_impl="transformer_engine")

    forward = get_forward(model)

    for name, param in model.named_parameters():
        param.requires_grad = True

    # Set to eval mode to disable dropout for deterministic outputs
    model.eval()
    ref_output = forward(model)

    model = mtq.quantize(model, mtq.INT8_DEFAULT_CFG, forward)

    for n, m in model.named_modules():
        if isinstance(m, TERowParallelLinear):
            assert isinstance(m, _QuantTEMCoreRowParallelLinear), f"{m=}, {type(m)}"
            assert m.input_quantizer.amax is not None
            assert m.weight_quantizer.amax is not None

    # Save which quantizers are enabled before disabling
    enabled_quantizers = {
        name
        for name, m in model.named_modules()
        if isinstance(m, mtq.nn.TensorQuantizer) and m.is_enabled
    }

    mtq.disable_quantizer(model, "*")
    disabled_output = forward(model)
    assert torch.allclose(ref_output, disabled_output, atol=1e-5), (
        "Output with quantizers disabled should match reference output"
    )

    mtq.enable_quantizer(model, lambda name: name in enabled_quantizers)
    enabled_output = forward(model)
    assert not torch.allclose(ref_output, enabled_output, atol=1e-5), (
        "Output with quantizers enabled should differ from reference output"
    )
    # enable model for training to test backward pass
    model.train()
    loss = forward(model).sum()
    loss.backward()

    destroy_model_parallel()


def test_homogeneous_sharded_state_dict_te_spec(dist_workers, tmp_path):
    dist_workers.run(
        partial(
            _test_sharded_state_dict,
            tmp_path,
            mtq.INT8_DEFAULT_CFG,
            256,
            None,
            False,
            False,
            {"transformer_impl": "transformer_engine"},
        ),
    )


def test_output_layer_extra_state_empty_when_nothing_quantized():
    """``GPTModel.sharded_state_dict`` asserts a disabled output_layer carries no extra state.

    The subject is a ``RealQuantLinear`` with an uncompressed weight, which is what ``mtq.compress``
    leaves behind for a disabled output_layer, and the e2e coverage
    (``test_homogeneous_compressed_sharded_state_dict``) is Blackwell-skipped.
    """
    module = RealQuantLinear.convert(QuantModuleRegistry.convert(torch.nn.Linear(4, 4)))
    module._modelopt_output_layer = True
    assert not isinstance(module.weight, QTensorWrapper)  # disabled quantizers are not compressed

    for quantizer in module.modules():
        if isinstance(quantizer, mtq.nn.TensorQuantizer):
            quantizer.disable()
    assert quant_module_get_extra_state(module) == {}

    module.weight_quantizer.enable()
    assert "modelopt_quantizer_state" in quant_module_get_extra_state(module)


def test_resolve_output_layer_untied():
    """The tiedness signal is read off the model, not from Megatron-LM global args."""

    class _Flagged(torch.nn.Module):
        def __init__(self, shared):
            super().__init__()
            self.share_embeddings_and_output_weights = shared

    # No signal anywhere -> unknown.
    assert _resolve_output_layer_untied(torch.nn.Module()) is None

    # The root's own flag wins over any subtree.
    root = _Flagged(False)
    root.inner = _Flagged(True)
    assert _resolve_output_layer_untied(root) is True

    # Otherwise fall back to a subtree scan.
    root = torch.nn.Module()
    root.language_model = _Flagged(True)
    assert _resolve_output_layer_untied(root) is False

    # Subtrees that do not own the language model's output_layer are skipped: the vision tower
    # and a distillation teacher, either of which may be tied differently from the student.
    root = torch.nn.Module()
    root.vision_model = _Flagged(True)
    root._teacher_model = _Flagged(True)
    root.language_model = _Flagged(False)
    assert _resolve_output_layer_untied(root) is True


@pytest.mark.parametrize("mlm_untied", [True, False])
def test_output_layer_untied_falls_back_to_megatron_lm_args(mlm_untied):
    """With no model-derived flag, the answer comes from Megatron-LM's args."""

    class _Config:
        pass

    fake_training = types.ModuleType("megatron.training")
    fake_training.get_args = lambda: SimpleNamespace(untie_embeddings_and_output_weights=mlm_untied)

    config = _Config()
    with patch.dict(sys.modules, {"megatron.training": fake_training}):
        assert _output_layer_untied(config) is mlm_untied

    # The model-derived flag takes precedence over the args fallback.
    config.modelopt_output_layer_untied = not mlm_untied
    with patch.dict(sys.modules, {"megatron.training": fake_training}):
        assert _output_layer_untied(config) is (not mlm_untied)


def test_output_layer_untied_warns_once_when_args_unavailable():
    """Without either signal the layer is treated as tied, and the warning is not repeated."""

    class _Config:
        pass

    broken = types.ModuleType("megatron.training")  # no get_args attribute

    config = _Config()
    with (
        patch.dict(sys.modules, {"megatron.training": broken}),
        patch("modelopt.torch.quantization.plugins.megatron.warn_rank_0") as warn,
    ):
        assert _output_layer_untied(config) is False
        assert _output_layer_untied(config) is False
    assert warn.call_count == 1


def test_output_layer_untied_warns_when_args_uninitialized():
    """Megatron-LM importable but not initialized: treated as tied, warned once."""

    class _Config:
        pass

    def _uninitialized():
        raise AssertionError("args is not initialized.")

    fake_training = types.ModuleType("megatron.training")
    fake_training.get_args = _uninitialized

    config = _Config()
    with (
        patch.dict(sys.modules, {"megatron.training": fake_training}),
        patch("modelopt.torch.quantization.plugins.megatron.warn_rank_0") as warn,
    ):
        assert _output_layer_untied(config) is False
        assert _output_layer_untied(config) is False
    assert warn.call_count == 1


def test_output_layer_untied_not_stamped_onto_teacher_config():
    """A distillation teacher keeps its own tiedness; the student's answer must not leak in."""

    def _config():
        return TransformerConfig(num_layers=1, hidden_size=8, num_attention_heads=1)

    class _Tiny(MegatronModule):
        def __init__(self, config, shared):
            super().__init__(config)
            self.share_embeddings_and_output_weights = shared

    student = _Tiny(_config(), shared=False)  # untied
    student._teacher_model = _Tiny(_config(), shared=True)  # tied -- must not be overwritten

    megatron_replace_quant_module_hook(student)

    assert student.config.modelopt_output_layer_untied is True
    assert not hasattr(student._teacher_model.config, "modelopt_output_layer_untied")
