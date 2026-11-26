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

"""Load the Qwen3.6 QAD training checkpoint in NeMo RL.

QAD saves the language model weights and ModelOpt state in Megatron format.
The Hugging Face export combines those language weights with the original
Qwen vision weights to make a full vision-language model. This worker loads
the Megatron checkpoint using the NeMo RL revision in experiment_environment.yaml.

See Qwen36QuantPolicyWorker for the loading details.
"""

import sys
from dataclasses import replace
from functools import wraps
from pathlib import Path
from types import MethodType

import nemo_rl.models.policy.workers.megatron_policy_worker as policy_worker
import ray
from megatron.bridge.training.post_training.checkpointing import has_modelopt_state
from megatron.core import dist_checkpointing
from megatron.core.dist_checkpointing.mapping import ShardedBase, ShardedTensorFactory
from megatron.core.dist_checkpointing.validation import StrictHandling
from megatron.core.utils import unwrap_model
from nemo_rl.distributed.ray_actor_environment_registry import ACTOR_ENVIRONMENT_REGISTRY
from nemo_rl.modelopt.models.policy.workers.megatron_quant_policy_worker import (
    MegatronQuantPolicyWorker,
)
from nemo_rl.models.policy.utils import POLICY_WORKER_OVERRIDES
from sm103_attention import enable_sm103_flash
from torch.distributed.checkpoint import FileSystemReader

from modelopt.torch.quantization.nn.modules.tensor_quantizer import TensorQuantizer

__all__ = ["register_qwen36_worker", "validate_checkpoint"]

_BASE_WORKER = "nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker"


def validate_checkpoint(path: Path) -> Path:
    """Return the absolute path to a Megatron iteration with saved quantization state."""
    path = path.resolve(strict=True)
    if not has_modelopt_state(str(path)):
        raise ValueError(f"Expected saved ModelOpt quantization state in {path}")
    return path


def register_qwen36_worker() -> None:
    """Select the Qwen checkpoint worker for quantized Megatron training.

    NeMo RL uses this override when it creates its training workers. Ray starts
    those workers with the same Python interpreter as the training launcher.
    """
    worker = f"{__name__}.Qwen36QuantPolicyWorker"
    POLICY_WORKER_OVERRIDES[_BASE_WORKER] = worker
    ACTOR_ENVIRONMENT_REGISTRY[worker] = sys.executable


def _without_language_prefix(value):
    """Remove the language_model prefix from requested checkpoint keys."""
    if isinstance(value, ShardedTensorFactory):
        build = value.build_fn

        def build_without_prefix(*args, **kwargs):
            return _without_language_prefix(build(*args, **kwargs))

        # Grouped expert weights can create their sharded tensor keys lazily.
        return replace(value, build_fn=build_without_prefix)
    if isinstance(value, ShardedBase):
        return replace(value, key=value.key.removeprefix("language_model."))
    if isinstance(value, dict):
        return {key: _without_language_prefix(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_without_language_prefix(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_without_language_prefix(item) for item in value)
    return value


def _adapt_checkpoint_load(checkpoint: Path) -> None:
    """Match the weight names Bridge requests to the names QAD saved.

    If QAD saved the weights without the language_model prefix, remove it from
    the requested keys. Require every requested key to be present.
    """
    saved_keys = FileSystemReader(str(checkpoint)).read_metadata().state_dict_metadata
    strip_prefix = not any(key.startswith("language_model.") for key in saved_keys)
    original = dist_checkpointing.load

    @wraps(original)
    def load(state, checkpoint_dir, sharded_strategy=None, **kwargs):
        if Path(checkpoint_dir).resolve() == checkpoint:
            if strip_prefix:
                state = _without_language_prefix(state)
            kwargs["strict"] = StrictHandling.RAISE_UNEXPECTED
        return original(state, checkpoint_dir, sharded_strategy, **kwargs)

    dist_checkpointing.load = load


def _expand_text_positions(module, args, kwargs):
    """Provide the three rotary position axes Qwen expects.

    The full VLM wrapper normally computes time, height, and width positions
    from the tokens and any image or video grid. Here NeMo RL calls the language
    model directly with text positions, so repeat each position on all three axes.
    """
    positions = kwargs.get("position_ids", args[1] if len(args) > 1 else None)
    if positions is not None and positions.ndim == 2:
        positions = positions.unsqueeze(0).expand(3, -1, -1)
        if "position_ids" in kwargs:
            kwargs = {**kwargs, "position_ids": positions}
        else:
            args = (args[0], positions, *args[2:])
    return args, kwargs


def _provide_language_model(provider, *args, **kwargs):
    """Build Qwen's language model and adapt its rotary positions for text batches."""
    model = provider.provide_language_model(*args, **kwargs)
    if model.position_embedding_type == "mrope":
        model.register_forward_pre_hook(_expand_text_positions, with_kwargs=True)
    return model


def _language_layer_spec(provider, vp_stage=None):
    """Use Qwen's native hybrid attention layout for the requested pipeline stage."""
    return provider.build_language_spec(vp_stage=vp_stage)


# Expose the worker as a Ray actor so NeMo RL can start it in its own process.
@ray.remote
class Qwen36QuantPolicyWorker(MegatronQuantPolicyWorker.__ray_actor_class__):
    """Adapt NeMo RL's Megatron worker for the Qwen3.6 QAD checkpoint.

    - Megatron Bridge QAD saves only Qwen's language weights. The default NeMo RL
      worker builds the full VLM, so the custom worker builds only the language model.
    - The default worker looks for ModelOpt state under iter_0000000. The custom
      worker uses the supplied QAD iteration directory, such as iter_0000500.
    - The VLM wrapper normally computes rotary positions. The custom worker calls
      the language model directly, so it repeats each text position across three axes.
    """

    def __init__(self, config, *args, **kwargs):
        """Set up checkpoint key handling before NeMo RL loads the weights."""
        _adapt_checkpoint_load(Path(config["pretrained_checkpoint"]["path"]))
        super().__init__(config, *args, **kwargs)

    def _patch_validate_model_paths(self):
        """Skip the quantized worker's checkpoint path rewrite.

        The parent hook looks for ModelOpt state under iter_0000000 and can
        redirect the checkpoint path. Megatron Bridge QAD saves the state
        directly in the iteration directory, such as iter_0000500. This
        empty override keeps NeMo RL's normal path validation in place.
        """

    def _patch_setup_model_and_optimizer(self):
        """Build the Qwen language model and its native attention layout.

        Qwen's Hugging Face config describes the full VLM. Select Bridge's
        language model provider so Megatron requests the weights QAD saved.
        """
        original = policy_worker.setup_model_and_optimizer

        def setup(policy_cfg, megatron_cfg, *args, **kwargs):
            # Apply the original GB300 attention workaround before building the model.
            enable_sm103_flash()
            provider = megatron_cfg.model
            provider.provide = MethodType(_provide_language_model, provider)
            provider.transformer_layer_spec = _language_layer_spec
            provider.restore_modelopt_state = True
            return original(policy_cfg, megatron_cfg, *args, **kwargs)

        policy_worker.setup_model_and_optimizer = setup

    def _restore_modelopt_state_pre_load(self, state, model):
        """Check for active quantizers before loading the weights.

        NeMo RL's original hook restores the saved ModelOpt state. The check
        catches restores that leave all quantizers inactive.
        """
        super()._restore_modelopt_state_pre_load(state, model)
        if not any(
            isinstance(module, TensorQuantizer) and module.is_enabled
            for chunk in unwrap_model(model)
            for module in chunk.modules()
        ):
            raise RuntimeError("No active ModelOpt quantizers after restoring checkpoint state")
