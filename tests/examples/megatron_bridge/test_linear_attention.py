# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import json
import runpy

import pytest
import torch
from _test_utils.examples.megatron_example_runner import reset_megatron_global_state
from _test_utils.examples.run_command import MODELOPT_ROOT
from _test_utils.torch.distributed.utils import DistributedWorkerPool
from megatron.bridge.models.hybrid.hybrid_provider import HybridModelProvider
from megatron.bridge.training.config import (
    CheckpointConfig,
    ConfigContainer,
    DistributedDataParallelConfig,
    LoggerConfig,
    MockGPTDatasetConfig,
    OptimizerConfig,
    RNGConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainingConfig,
    ValidationConfig,
)
from megatron.bridge.training.post_training.checkpointing import has_modelopt_state
from megatron.core.utils import unwrap_model

from modelopt.torch.quantization.utils import is_quantized

run_training = runpy.run_path(str(MODELOPT_ROOT / "examples/llm_qat/linear_attention/train.py"))[
    "run_training"
]


def _train(tmp_path, qad, recipe, train_steps=1, tp_size=1):
    captured = {}

    def provider(name):
        model = HybridModelProvider(
            num_layers=1,
            hidden_size=64,
            ffn_hidden_size=128,
            num_attention_heads=2,
            hybrid_layer_pattern="G",
            vocab_size=128,
            tensor_model_parallel_size=tp_size,
            sequence_parallel=tp_size > 1,
            seq_length=16,
            linear_num_key_heads=2,
            linear_num_value_heads=2,
            linear_key_head_dim=32,
            linear_value_head_dim=32,
            linear_conv_kernel_dim=4,
            linear_attention_freq=1,
            experimental_attention_variant="gated_delta_net",
            is_hybrid_model=True,
            activation_func=torch.nn.functional.silu,
            calculate_per_token_loss=True,
            gradient_accumulation_fusion=False,
            recompute_granularity="full",
            recompute_method="uniform",
            recompute_num_layers=1,
            cross_entropy_loss_fusion=False,
        )

        def capture(models):
            module = unwrap_model(models[0])
            captured[name] = module
            captured[name + "_before"] = {
                key: value.detach().clone() for key, value in module.named_parameters()
            }
            if name == "student":

                def check_mask(module, args, kwargs):
                    mask = kwargs["loss_mask"]
                    assert not mask[:, :8].any()
                    assert mask[:, 8:].all()

                module.register_forward_pre_hook(check_mask, with_kwargs=True)
            return models

        model.register_post_wrap_hook(capture)
        return model

    checkpoint = str(tmp_path / "checkpoints")
    config = ConfigContainer(
        model=provider("student"),
        train=TrainingConfig(train_iters=train_steps, global_batch_size=2, micro_batch_size=1),
        validation=ValidationConfig(eval_iters=0, eval_interval=1),
        optimizer=OptimizerConfig(
            optimizer="adam", lr=1e-2, min_lr=0, weight_decay=0, use_distributed_optimizer=True
        ),
        scheduler=SchedulerConfig(
            lr_decay_style="constant",
            lr_warmup_iters=0,
            start_weight_decay=0,
            end_weight_decay=0,
            use_checkpoint_opt_param_scheduler=True,
        ),
        ddp=DistributedDataParallelConfig(
            average_in_collective=False, use_distributed_optimizer=True
        ),
        dataset=MockGPTDatasetConfig(
            seq_length=16,
            random_seed=123,
            reset_position_ids=False,
            reset_attention_mask=False,
            eod_mask_loss=False,
            dataloader_type="single",
            num_workers=0,
        ),
        tokenizer=TokenizerConfig(tokenizer_type="NullTokenizer", vocab_size=128),
        checkpoint=CheckpointConfig(
            save=checkpoint, load=checkpoint, save_interval=1, async_save=False
        ),
        logger=LoggerConfig(log_interval=1),
        rng=RNGConfig(seed=123),
        mixed_precision="bf16_mixed",
    )
    run_training(config, recipe, 8, provider("teacher") if qad else None)
    return captured, checkpoint


def _reset_worker(rank, world_size):
    reset_megatron_global_state()


def _warmup(rank, world_size, path, recipe):
    _train(path / "dp", False, recipe)
    reset_megatron_global_state()
    _train(path / "tp", True, recipe, tp_size=world_size)


@pytest.fixture(scope="module")
def compiled_state_training(tmp_path_factory, project_root_path, num_gpus):
    """Warm up at most two ranks before timing the DP QAT / TP QAD tests."""
    if not num_gpus:
        pytest.skip("Requires CUDA")
    recipe = json.loads(
        (
            project_root_path / "examples/llm_qat/linear_attention/configs/decode_state_int8.json"
        ).read_text()
    )
    workers = DistributedWorkerPool(min(num_gpus, 2), teardown_fn=_reset_worker)
    try:
        workers.run(_warmup, tmp_path_factory.mktemp("compile_state_training"), recipe)
        yield workers, recipe
    finally:
        workers.shutdown()


@pytest.mark.parametrize("qad", [False, True], ids=["qat", "qad"])
def test_state_training(compiled_state_training, tmp_path, qad):
    workers, recipe = compiled_state_training
    workers.run(_check_training, tmp_path, qad, recipe)


def _check_training(rank, world_size, tmp_path, qad, recipe):
    tp_size = world_size if qad else 1
    captured, checkpoint = _train(tmp_path, qad, recipe, tp_size=tp_size)
    student = captured["student"]
    assert has_modelopt_state(checkpoint)
    assert is_quantized(student)
    layers = [m for m in student.modules() if hasattr(m, "gdn_state_quantizer")]
    assert len(layers) == 1
    assert layers[0].gdn_state_quantizer.is_enabled
    assert layers[0].linear_attention_config.decode.state_codec == "int8_hadamard32"
    changed = []
    for name, parameter in student.named_parameters():
        assert torch.isfinite(parameter).all()
        if not torch.equal(parameter, captured["student_before"][name]):
            assert parameter.requires_grad
            changed.append(name)
    assert changed
    assert all(
        module._linear_attention_prefill_lengths is None
        for module in student.modules()
        if hasattr(module, "_linear_attention_prefill_lengths")
    )
    if qad:
        teacher = captured["teacher"]
        assert not is_quantized(teacher)
        for name, parameter in teacher.named_parameters():
            assert not parameter.requires_grad
            assert torch.equal(parameter, captured["teacher_before"][name])
        before_resume = {
            name: parameter.detach().clone()
            for name, parameter in student.named_parameters()
            if parameter.requires_grad
        }
        reset_megatron_global_state()
        # An empty recipe verifies that resume restores the saved quantization policy.
        resumed, _ = _train(tmp_path, True, {}, train_steps=2, tp_size=tp_size)
        assert has_modelopt_state(str(tmp_path / "checkpoints/iter_0000002"))
        assert is_quantized(resumed["student"])
        assert not is_quantized(resumed["teacher"])
        assert any(
            not torch.equal(parameter, before_resume[name])
            for name, parameter in resumed["student"].named_parameters()
            if name in before_resume
        )
