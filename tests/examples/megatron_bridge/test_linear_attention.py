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
from megatron.core.utils import unwrap_model

from modelopt.torch.quantization.utils import is_quantized

run_training = runpy.run_path(str(MODELOPT_ROOT / "examples/llm_qat/linear_attention/train.py"))[
    "run_training"
]


def _train(qad, recipe):
    captured = {}

    def provider(name):
        model = HybridModelProvider(
            num_layers=1,
            hidden_size=64,
            ffn_hidden_size=128,
            num_attention_heads=2,
            hybrid_layer_pattern="G",
            vocab_size=128,
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
            if name == "student":
                captured["before"] = (
                    module.decoder.layers[0].self_attention.out_proj.weight.detach().clone()
                )
            return models

        model.register_post_wrap_hook(capture)
        return model

    config = ConfigContainer(
        model=provider("student"),
        train=TrainingConfig(train_iters=1, global_batch_size=1, micro_batch_size=1),
        validation=ValidationConfig(eval_iters=0, eval_interval=1),
        optimizer=OptimizerConfig(
            optimizer="adam", lr=1e-2, min_lr=0, weight_decay=0, use_distributed_optimizer=True
        ),
        scheduler=SchedulerConfig(
            lr_decay_style="constant",
            lr_warmup_iters=0,
            start_weight_decay=0,
            end_weight_decay=0,
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
        checkpoint=CheckpointConfig(async_save=False),
        logger=LoggerConfig(log_interval=1),
        rng=RNGConfig(seed=123),
        mixed_precision="bf16_mixed",
    )
    run_training(config, recipe, 8, provider("teacher") if qad else None)
    return captured


@pytest.fixture(scope="module")
def compiled_state_training(project_root_path):
    """Compile one tiny GDN shape before timing the single-GPU training checks."""
    if not torch.cuda.is_available():
        pytest.skip("Requires CUDA")
    recipe = json.loads(
        (
            project_root_path / "examples/llm_qat/linear_attention/configs/decode_state_int8.json"
        ).read_text()
    )
    try:
        _train(True, recipe)
    finally:
        reset_megatron_global_state()
    return recipe


@pytest.mark.parametrize("qad", [False, True], ids=["qat", "qad"])
def test_state_training(compiled_state_training, qad):
    captured = _train(qad, compiled_state_training)
    attention = captured["student"].decoder.layers[0].self_attention
    assert attention.gdn_state_quantizer.is_enabled
    assert torch.isfinite(attention.out_proj.weight).all()
    assert not torch.equal(attention.out_proj.weight, captured["before"])
    if qad:
        teacher = captured["teacher"]
        assert not is_quantized(teacher)
        assert not any(parameter.requires_grad for parameter in teacher.parameters())
