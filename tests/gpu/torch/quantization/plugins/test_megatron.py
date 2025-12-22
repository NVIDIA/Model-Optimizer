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
from functools import partial

import pytest
import torch
from _test_utils.import_helper import skip_if_no_megatron
from _test_utils.torch.distributed.utils import spawn_multiprocess_job
from _test_utils.torch.megatron.models import MegatronModel, get_mcore_gpt_model
from _test_utils.torch.megatron.utils import (
    compare_amax_sync_across_expert_parallel,
    copy_weights_from_grouped_to_non_grouped,
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
)

skip_if_no_megatron()

from megatron.core.parallel_state import (
    destroy_model_parallel,
    get_data_parallel_group,
    get_tensor_model_parallel_group,
)
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.moe.experts import SequentialMLP, TEGroupedMLP
from megatron.core.transformer.moe.router import TopKRouter

import modelopt
import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.nn import QuantModuleRegistry
from modelopt.torch.quantization.plugins.megatron import _QuantTEMCoreRowParallelLinear
from modelopt.torch.utils.plugins import megatron_prefill

try:
    from megatron.core.extensions.transformer_engine import TERowParallelLinear

    HAS_TE = True
except ImportError:
    HAS_TE = False

SEED = 1234


def get_batch(model, batch_size=2):
    seq_length = model.max_sequence_length
    vocab_size = model.vocab_size

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_length)).cuda()
    labels = torch.randint(0, vocab_size, (batch_size, seq_length)).cuda()
    position_ids = (
        torch.arange(seq_length, dtype=torch.int64).unsqueeze(0).repeat(batch_size, 1).cuda()
    )
    attention_mask = torch.tril(
        torch.ones((batch_size, 1, seq_length, seq_length), dtype=torch.bool)
    ).cuda()
    loss_mask = torch.ones((batch_size, seq_length), dtype=torch.float32).cuda()

    return input_ids, labels, position_ids, attention_mask, loss_mask


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

    mtq.set_quantizer_attribute(model_test, "*", {"enable": False})

    x = model_ref.get_dummy_input().cuda()
    out_1 = model_ref(x)
    out_2 = model_test(x)
    assert torch.allclose(out_1, out_2)

    mtq.set_quantizer_attribute(model_test, "*input_quantizer", {"enable": True})
    mtq.set_quantizer_attribute(model_test, "*weight_quantizer", {"enable": True})
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
def test_tensor_parallel(need_2_gpus, config):
    spawn_multiprocess_job(
        size=2,
        job=partial(_test_parallelism_helper, config, tensor_model_parallel_size=2),
        backend="nccl",
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
def test_data_parallel(need_2_gpus, config):
    spawn_multiprocess_job(
        size=2,
        job=partial(_test_parallelism_helper, config, use_rank_in_seed=True),
        backend="nccl",
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
def test_context_parallel(need_2_gpus, config):
    spawn_multiprocess_job(
        size=2,
        job=partial(
            _test_parallelism_helper, config, context_parallel_size=2, use_rank_in_seed=True
        ),
        backend="nccl",
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
def test_data_tensor_context_parallel(need_8_gpus, config):
    spawn_multiprocess_job(
        size=8,
        job=partial(
            _test_parallelism_helper,
            config,
            tensor_model_parallel_size=2,
            context_parallel_size=2,
            use_rank_in_seed=True,
            test_pre_quant_scale=False,
        ),
        backend="nccl",
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
    use_te=False,
    transformer_impl="local",
):
    """Build the model."""

    if meta_device:
        with torch.device("meta"):
            gpt_model = get_mcore_gpt_model(
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
                use_te=use_te,
            )
    else:
        gpt_model = get_mcore_gpt_model(
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
            num_moe_experts=num_moe_experts,
            moe_grouped_gemm=moe_grouped_gemm,
            use_te=use_te,
        ).cuda()
    return gpt_model.eval()


def _test_sharded_state_dict(
    tmp_path, config, hidden_size, modelopt_version, compress, meta_device, moe_config, rank, size
):
    # Must disable output_layer quantization since output_layer amax cannot be restore via
    # sharded_state_dict. All output_layer quantizers state are removed.
    config["quant_cfg"]["*output_layer*"] = {"enable": False}

    if modelopt_version is not None:
        mto.conversion.__version__ = modelopt_version
        mtq.plugins.megatron.__version__ = modelopt_version

    tp_size = moe_config.get("tp_size", size)
    ep_size = moe_config.get("ep_size", 1)
    etp_size = moe_config.get("etp_size", None)
    num_moe_experts = moe_config.get("num_moe_experts", None)
    moe_grouped_gemm = moe_config.get("moe_grouped_gemm", False)
    use_te = moe_config.get("use_te", False)
    transformer_impl = moe_config.get("transformer_impl", "local")

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
        use_te=use_te,
        ep_size=ep_size,
        etp_size=etp_size,
        transformer_impl=transformer_impl,
    )
    model_test = _gpt_model_provider(
        tp_size,
        hidden_size,
        vocab_size=256,
        num_moe_experts=num_moe_experts,
        moe_grouped_gemm=moe_grouped_gemm,
        use_te=use_te,
        meta_device=meta_device,
        ep_size=ep_size,
        etp_size=etp_size,
        transformer_impl=transformer_impl,
    )

    prompt_tokens = torch.randint(
        0, model_ref.vocab_size, (2, model_ref.max_sequence_length)
    ).cuda()

    def forward_fn(model):
        return megatron_prefill(model, prompt_tokens)

    model_ref = mtq.quantize(model_ref, config, forward_fn)
    if compress:
        mtq.compress(model_ref)

    for module in model_ref.modules():
        if hasattr(module, "_amax_for_smoothing"):
            delattr(module, "_amax_for_smoothing")

    sharded_state_dict_test_helper(
        tmp_path,
        model_ref,
        model_test,
        forward_fn,
        meta_device=meta_device,
        version=modelopt_version,
    )

    if modelopt_version is not None:
        mto.conversion.__version__ = modelopt.__version__
        mtq.plugins.megatron.__version__ = modelopt.__version__

    # Make sure all ranks have arrived before destroying NCCL
    torch.distributed.barrier()


mixed_precision_config = copy.deepcopy(mtq.W4A8_AWQ_BETA_CFG)
mixed_precision_config["quant_cfg"].update(
    {
        "*.1.*": {"enable": False},
        "*.2.*weight_quantizer": {"num_bits": (4, 3), "axis": None},
        "*.2.*input_quantizer": {"num_bits": (4, 3), "axis": None},
        "*.3.*weight_quantizer.0": {"num_bits": 8, "axis": 0},
        "*.3.*weight_quantizer.1": {"enable": False},
        "*.3.*input_quantizer": {"num_bits": 8, "axis": None},
    }
)


mixed_block_size_config = copy.deepcopy(mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG)
mixed_block_size_config["quant_cfg"].update(
    {
        "*.1.*": {"enable": False},
        "*.2.*weight_quantizer": {"num_bits": 4, "block_sizes": {-1: 64}, "enable": True},
        "*.2.*input_quantizer": {"num_bits": (4, 3), "axis": None},
        "*.3.*weight_quantizer": {"num_bits": 4, "block_sizes": {-1: 128, -2: 64}, "enable": True},
        "*.3.*input_quantizer": {"num_bits": 8, "axis": None},
    }
)


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
        # Note: KV cache configs (FP8_KV_CFG, NVFP4_KV_CFG) are tested separately in test_kv_cache_quant
        # They require TEDotProductAttention which needs transformer_impl="modelopt", not "local"
    ],
)
@pytest.mark.parametrize("compress", [False, True])
@pytest.mark.parametrize("meta_device", [False, True])
def test_homogeneous_sharded_state_dict(tmp_path, config, compress, meta_device):
    if compress and config is mtq.W4A8_AWQ_BETA_CFG:
        pytest.skip("W4A8_AWQ_BETA_CFG is not supported for compress")

    size = torch.cuda.device_count()

    spawn_multiprocess_job(
        size=size,
        job=partial(
            _test_sharded_state_dict, tmp_path, config, 256, None, compress, meta_device, {}
        ),
        backend="nccl",
    )


@pytest.mark.parametrize(
    "config",
    [
        mixed_precision_config,
        mixed_block_size_config,
    ],
)
def test_heterogenous_sharded_state_dict(need_2_gpus, tmp_path, config):
    spawn_multiprocess_job(
        size=2,
        job=partial(_test_sharded_state_dict, tmp_path, config, 256, None, False, False, {}),
        backend="nccl",
    )


@pytest.mark.parametrize(
    "config",
    [
        mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG,
        mtq.NVFP4_DEFAULT_CFG,
        mixed_precision_config,
    ],
)
@pytest.mark.parametrize("modelopt_version", ["0.25", "0.27"])
@pytest.mark.skip(
    reason="0.31 has breaking change without backward compatibility. This unittest needs to be refactorized."
)
def test_sharded_state_dict_old_checkpoints(need_2_gpus, tmp_path, config, modelopt_version):
    spawn_multiprocess_job(
        size=2,
        job=partial(
            _test_sharded_state_dict, tmp_path, config, 256, modelopt_version, False, False, {}
        ),
        backend="nccl",
    )


@pytest.mark.parametrize("hidden_size", [256, 320])
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


def test_auto_quantize(need_2_gpus):
    spawn_multiprocess_job(size=2, job=_test_auto_quantize_helper, backend="nccl")


def _test_fp8_real_quantize_helper(rank, size):
    initialize_for_megatron(
        tensor_model_parallel_size=size, pipeline_model_parallel_size=1, seed=SEED
    )
    hidden_size = 256
    config = mtq.FP8_2D_BLOCKWISE_WEIGHT_ONLY_CFG

    model = _gpt_model_provider(tp_size=1, hidden_size=hidden_size)
    prompt_tokens = torch.randint(0, model.vocab_size, (2, model.max_sequence_length)).cuda()

    def forward_fn(model):
        return megatron_prefill(model, prompt_tokens)

    forward_fn(model)

    # real quant the model
    cur_mem = get_model_size(model)
    real_quant_model = mtq.quantize(model, config, forward_fn)
    mtq.compress(real_quant_model)
    real_quant_mem = get_model_size(real_quant_model)

    # Since not all parameters are quantized, the size won't be lower than half.
    assert real_quant_mem < (cur_mem / 2) * 1.1, "Memory after real quantization is not reduced."

    # check forward works after real quantization
    forward_fn(real_quant_model)

    assert real_quant_mem < cur_mem


def test_fp8_real_quantize():
    size = torch.cuda.device_count()
    spawn_multiprocess_job(size=size, job=_test_fp8_real_quantize_helper, backend="nccl")


@pytest.mark.skip(reason="TODO: etp requires sequence parallelism now in Megatron due to a bug;")
@pytest.mark.parametrize(
    "config",
    [mtq.FP8_DEFAULT_CFG, mtq.NVFP4_DEFAULT_CFG, mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG],
)
@pytest.mark.parametrize("moe_grouped_gemm", [True, False])
def test_moe_sharded_state_dict(need_4_gpus, tmp_path, config, moe_grouped_gemm):
    if moe_grouped_gemm:
        pytest.skip("TEGroupedMLP is not enabled in Megatron-LM currently")
    size = torch.cuda.device_count()
    # TODO: Add support for compress=True for TEGroupedMLP
    moe_config = {
        "tp_size": 2,
        "ep_size": 2,
        "etp_size": 2,
        "num_moe_experts": 4,
        "moe_grouped_gemm": moe_grouped_gemm,
        "use_te": moe_grouped_gemm,
        "transformer_impl": "modelopt",
    }
    spawn_multiprocess_job(
        size=size,
        job=partial(
            _test_sharded_state_dict,
            tmp_path,
            config,
            256,
            None,
            False,
            False,
            moe_config,
        ),
        backend="nccl",
    )


def _test_te_grouped_vs_sequential_quantize_helper(tp_size, ep_size, etp_size, rank, size):
    """Test that TEGrouped and sequential MoE models produce similar amax values."""
    initialize_for_megatron(
        tensor_model_parallel_size=tp_size,
        expert_model_parallel_size=ep_size,
        expert_tensor_parallel_size=etp_size,
        seed=SEED,
    )

    # Create input
    prompt_tokens = torch.randint(0, 64, (2, 16)).cuda()

    def forward_fn(model):
        return megatron_prefill(model, prompt_tokens)

    # Create TEGrouped MoE model
    te_grouped_moe_model = _gpt_model_provider(
        tp_size=tp_size,
        ep_size=ep_size,
        etp_size=etp_size,
        hidden_size=32,
        moe_grouped_gemm=True,
        use_te=True,
        num_moe_experts=4,
    )
    num_te_grouped_mlp = sum(
        isinstance(module, TEGroupedMLP) for module in te_grouped_moe_model.modules()
    )
    assert num_te_grouped_mlp == 4, (
        f"TEGrupedMoEModel has {num_te_grouped_mlp} TEGroupedMLP modules, it should have 4"
    )

    # Create sequential MoE model
    sequential_moe_model = _gpt_model_provider(
        tp_size=tp_size,
        ep_size=ep_size,
        etp_size=etp_size,
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
    te_grouped_moe_output = forward_fn(te_grouped_moe_model)
    sequential_moe_output = forward_fn(sequential_moe_model)
    assert torch.allclose(te_grouped_moe_output, sequential_moe_output, atol=1e-6, rtol=1e-6)

    # Quantize grouped model
    mtq.quantize(te_grouped_moe_model, mtq.FP8_DEFAULT_CFG, forward_fn)

    # Quantize non-grouped model
    mtq.quantize(sequential_moe_model, mtq.FP8_DEFAULT_CFG, forward_fn)

    # Compare model outputs after quantization
    te_grouped_moe_quant_output = forward_fn(te_grouped_moe_model)
    sequential_moe_quant_output = forward_fn(sequential_moe_model)
    assert torch.allclose(
        te_grouped_moe_quant_output, sequential_moe_quant_output, atol=1e-6, rtol=1e-6
    )


def test_te_grouped_vs_sequential_quantize(need_4_gpus):
    """Test that TEGrouped and sequential MoE models produce similar quantized models."""
    pytest.skip("TEGroupedMLP is not enabled in Megatron-LM currently")
    size = torch.cuda.device_count()
    spawn_multiprocess_job(
        size=size,
        job=partial(_test_te_grouped_vs_sequential_quantize_helper, 1, 2, 2),
        backend="nccl",
    )


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
        use_te=moe_grouped_gemm,
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

    prompt_tokens = (torch.ones((2, model.max_sequence_length)) * 0.05 + rank * 0.5).cuda().long()

    # force all expert routing
    for module in model.modules():
        if isinstance(module, TopKRouter):
            module.topk = module.num_experts

    def forward_fn(model):
        return megatron_prefill(model, prompt_tokens)

    # quantize the model
    model = mtq.quantize(model, config, forward_fn)
    # Check initial sync status
    initial_sync, quantizer_type, rank_values = compare_amax_sync_across_expert_parallel(model)
    assert initial_sync, (
        f"Inconsistent amax for expert {quantizer_type} across ranks: {rank_values}"
    )

    # Test if the amax values are inconsistent when distributed sync is disabled
    mtq.model_calib.max_calibrate(model, forward_fn, distributed_sync=False)
    inconsistent_amax, _, _ = compare_amax_sync_across_expert_parallel(
        model, compare_across_experts=False
    )

    assert not inconsistent_amax, (
        "Consistent amax across expert parallel ranks, "
        "Amax should not be synchronized across expert parallel ranks since expert parallel is disabled"
    )
    # calibrate the model with distributed sync and test synchronization
    mtq.model_calib.max_calibrate(model, forward_fn, distributed_sync=True)
    for module in model.modules():
        if hasattr(module, "sync_moe_local_experts_amax"):
            module.sync_moe_local_experts_amax()

    final_sync, quantizer_type, rank_values = compare_amax_sync_across_expert_parallel(model)
    assert final_sync, f"Inconsistent amax for expert {quantizer_type} across ranks: {rank_values}"


@pytest.mark.skip(reason="TODO: etp requires sequence parallelism now in Megatron due to a bug;")
@pytest.mark.parametrize("config", [mtq.FP8_DEFAULT_CFG, mtq.INT8_DEFAULT_CFG])
@pytest.mark.parametrize(("ep_size", "etp_size"), [(1, 2), (2, 1), (2, 2)])
@pytest.mark.parametrize("moe_grouped_gemm", [True, False])
def test_expert_parallel_sync(config, ep_size, etp_size, moe_grouped_gemm):
    """Test expert model parallel synchronization."""
    size = torch.cuda.device_count()
    if size < ep_size * etp_size:
        pytest.skip(f"Requires at least {ep_size * etp_size} GPUs for expert model parallel test")

    if moe_grouped_gemm:
        pytest.skip("TEGroupedMLP is not enabled in Megatron-LM currently")

    spawn_multiprocess_job(
        size=size,
        job=partial(
            _test_expert_model_parallel_amax_sync,
            etp_size,  # tp_size
            ep_size,
            etp_size,
            moe_grouped_gemm,
            config,
        ),
        backend="nccl",
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

    # Create dummy input for calibration
    prompt_tokens = torch.randint(0, model.vocab_size, (2, model.max_sequence_length)).cuda()

    def forward_fn(model):
        return megatron_prefill(model, prompt_tokens)

    # Test KV cache quantization with the given config
    quantized_model = mtq.quantize(model, config, forward_fn)

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
    output = forward_fn(quantized_model)
    assert output is not None, "Forward pass failed"


def _test_kv_cache_sharded_state_dict_helper(tmp_path, config, rank, size):
    """Helper for testing KV cache quantization with sharded state dict save/load."""
    # Disable output_layer quantization (same as other sharded state dict tests)
    config["quant_cfg"]["*output_layer*"] = {"enable": False}

    initialize_for_megatron(
        tensor_model_parallel_size=size, pipeline_model_parallel_size=1, seed=SEED
    )

    # Create GPT models with TEDotProductAttention (transformer_impl="modelopt")
    model_ref = get_mcore_gpt_model(
        tensor_model_parallel_size=size,
        num_layers=2,  # At least 2 layers to test multiple attention modules
        hidden_size=64,
        num_attention_heads=4,
        vocab_size=64,
        transformer_impl="modelopt",  # CRITICAL: Use TEDotProductAttention
    ).cuda()

    model_test = get_mcore_gpt_model(
        tensor_model_parallel_size=size,
        num_layers=2,
        hidden_size=64,
        num_attention_heads=4,
        vocab_size=64,
        transformer_impl="modelopt",
    ).cuda()

    prompt_tokens = torch.randint(
        0, model_ref.vocab_size, (2, model_ref.max_sequence_length)
    ).cuda()

    def forward_fn(model):
        return megatron_prefill(model, prompt_tokens)

    # Quantize the reference model
    model_ref = mtq.quantize(model_ref, config, forward_fn)

    # CRITICAL: model_test must also be quantized with the same config
    # Otherwise it won't have the KV cache quantizer keys when loading state dict
    model_test = mtq.quantize(model_test, config, forward_fn)

    # Verify KV cache quantizers were created
    kv_quantizers_found = False
    for name, module in model_ref.named_modules():
        if hasattr(module, "k_bmm_quantizer") and hasattr(module, "v_bmm_quantizer"):
            kv_quantizers_found = True
            assert module.k_bmm_quantizer.is_enabled, f"K quantizer not enabled in {name}"
            assert module.v_bmm_quantizer.is_enabled, f"V quantizer not enabled in {name}"

    assert kv_quantizers_found, "No KV cache quantizers found in quantized model"

    # Test sharded state dict save/load
    sharded_state_dict_test_helper(
        tmp_path,
        model_ref,
        model_test,
        forward_fn,
        meta_device=False,
        version=None,
    )

    # Verify KV cache quantizers are restored correctly in model_test
    for (name_ref, module_ref), (name_test, module_test) in zip(
        model_ref.named_modules(), model_test.named_modules()
    ):
        if hasattr(module_ref, "k_bmm_quantizer"):
            assert hasattr(module_test, "k_bmm_quantizer"), (
                f"K quantizer missing after restore in {name_test}"
            )
            assert hasattr(module_test, "v_bmm_quantizer"), (
                f"V quantizer missing after restore in {name_test}"
            )

            # Check that quantizer states match
            if hasattr(module_ref.k_bmm_quantizer, "_amax"):
                assert hasattr(module_test.k_bmm_quantizer, "_amax"), (
                    f"K quantizer _amax missing in {name_test}"
                )
                if module_ref.k_bmm_quantizer._amax is not None:
                    assert torch.allclose(
                        module_ref.k_bmm_quantizer._amax, module_test.k_bmm_quantizer._amax
                    ), f"K quantizer _amax mismatch in {name_test}"

            if hasattr(module_ref.v_bmm_quantizer, "_amax"):
                assert hasattr(module_test.v_bmm_quantizer, "_amax"), (
                    f"V quantizer _amax missing in {name_test}"
                )
                if module_ref.v_bmm_quantizer._amax is not None:
                    assert torch.allclose(
                        module_ref.v_bmm_quantizer._amax, module_test.v_bmm_quantizer._amax
                    ), f"V quantizer _amax mismatch in {name_test}"


@pytest.mark.parametrize(
    "config",
    [
        mtq.FP8_KV_CFG,
        mtq.NVFP4_KV_CFG,
    ],
)
def test_kv_cache_quant(config):
    """Verify KV cache quantization works correctly with TEDotProductAttention.

    This test ensures TEDotProductAttention is properly registered and gets the
    expected q/k/v_bmm_quantizers when using KV cache configs.

    Note: This test requires Transformer Engine to be installed since TEDotProductAttention
    is only available with transformer_impl="modelopt" or "transformer_engine" (not "local").
    """
    spawn_multiprocess_job(size=1, job=partial(_test_kv_cache_quant_helper, config), backend="nccl")


@pytest.mark.parametrize(
    "config",
    [
        mtq.FP8_KV_CFG,
        mtq.NVFP4_KV_CFG,
    ],
)
def test_kv_cache_sharded_state_dict(tmp_path, config):
    """Test KV cache quantization with sharded state dict save/load.

    This test verifies the complete workflow of saving and loading KV cache quantized
    models with distributed checkpointing, ensuring quantizer states are properly
    preserved across the save/load cycle.
    """
    size = min(2, torch.cuda.device_count())  # Use 2 GPUs if available, else 1
    spawn_multiprocess_job(
        size=size,
        job=partial(_test_kv_cache_sharded_state_dict_helper, tmp_path, config),
        backend="nccl",
    )


def test_convert_mcore_te_gpt_model(distributed_setup_size_1):
    if not HAS_TE:
        pytest.skip("Transformer Engine is not installed")
    initialize_for_megatron(tensor_model_parallel_size=1, seed=SEED)
    model = get_mcore_gpt_model(tensor_model_parallel_size=1, transformer_impl="transformer_engine")

    input_ids, labels, position_ids, attention_mask, loss_mask = get_batch(model)

    def forward(model):
        return model.forward(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            labels=labels,
            loss_mask=loss_mask,
        )

    for name, param in model.named_parameters():
        param.requires_grad = True

    # Set to eval mode to disable dropout for deterministic outputs
    model.eval()
    ref_output = forward(model)

    model = mtq.quantize(model, mtq.INT8_DEFAULT_CFG, forward)

    for n, m in model.named_modules():
        if isinstance(m, TERowParallelLinear):
            assert isinstance(m, _QuantTEMCoreRowParallelLinear)
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


def test_homogeneous_sharded_state_dict_te_spec(tmp_path):
    pytest.skip("The test is temporarily disabled to avoid CI timeout")
    spawn_multiprocess_job(
        size=2,
        job=partial(
            _test_sharded_state_dict,
            tmp_path,
            mtq.INT8_DEFAULT_CFG,
            256,
            None,
            False,
            False,
            {"transformer_impl": "transformer_engine"},
        ),
        backend="nccl",
    )
