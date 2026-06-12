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

"""Module implementing ``gradnas`` pruning algorithm for search.

Summary:
--------
`gradnas` algorithm gives a better score to sort various pruning choices over L1 norm (`fastnas`) for language models.

Details:
--------
Further,  we can get scores for hparams which are implemented even abstractly.
For example, we can use this algorithm to sort the heads in a multi-head attention layer. The attention heads
do not have a unique tensor parameter associated to it.

We are ranking the prunable choices for a particular hparam based on Sum((gradient of loss wrt pruning mask)^2).
The pruning mask of an hparam is a binary mask indicating which choices of the hparam are pruned
(0 means pruned and 1 means not pruned).

While calculating the backward gradient of loss, the masks are set to 1 at all tensors.
See more about masks being used to measure sensitivity in this paper: https://arxiv.org/pdf/1905.10650.pdf
"""

import types
import warnings
from collections.abc import Callable, Iterable
from contextlib import contextmanager
from functools import partial
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from pydantic import create_model
from torch.utils.hooks import RemovableHandle
from tqdm import tqdm

import modelopt.torch.nas.modules as dnn
from modelopt.torch.nas.conversion import NASModeRegistry
from modelopt.torch.nas.registry import DMRegistry
from modelopt.torch.nas.traced_hp import TracedHp
from modelopt.torch.nas.utils import get_subnet_config, sample, select
from modelopt.torch.opt.config import ModeloptBaseConfig, get_kwargs_for_create_model_with_rules
from modelopt.torch.opt.dynamic import DynamicModule
from modelopt.torch.opt.searcher import BaseSearcher, SearchConfig
from modelopt.torch.opt.utils import named_hparams
from modelopt.torch.utils import get_module_device, standardize_model_args, torch_to, unwrap_model

from .fastnas import BinarySearcher, FastNASModeDescriptor, _get_fastnas_default_rules
from .pruning import PruneModeRegistry

if TYPE_CHECKING:
    from modelopt.torch.nas.plugins.transformers import (
        _DynamicAttention,
        _DynamicBertAttention,
        _DynamicGPTJAttention,
    )

try:
    from transformers.models.bert.modeling_bert import BertAttention
    from transformers.models.gptj.modeling_gptj import GPTJAttention

    HAS_HF = True
except ImportError:
    HAS_HF = False


class GradientDataManager:
    """Class for managing gradient data for an hparam."""

    def __init__(self, shape, model, reduce_func=lambda x: x):
        """Initialize GradientDataManager."""
        self.mask = torch.ones(shape, requires_grad=True, device=get_module_device(model))
        self._score = torch.zeros_like(self.mask, requires_grad=False)
        self._reduce_func = reduce_func

    # TODO: Implement score averaging and early stopping based on score convergence
    def process_gradient(self):
        """Process gradient of the mask."""
        self._score += self.mask.grad.detach().pow(2)
        self.mask.grad = None

    @property
    def score(self):
        """The score of the hparam based on the stored gradients."""
        return self._reduce_func(self._score)


def _setup_grad_manager_linear(
    module: dnn._DynamicLinear,
) -> tuple[GradientDataManager, RemovableHandle]:
    def forward_hook(_modelopt_mask, module, input, output):
        return output * _modelopt_mask

    grad_data = GradientDataManager(module.get_hparam("out_features").max, module)
    hook_handle = module.register_forward_hook(partial(forward_hook, grad_data.mask))
    return grad_data, hook_handle


def _setup_grad_manager_hf_attention(
    module: "_DynamicAttention", head_mask_idx: int
) -> tuple[GradientDataManager, RemovableHandle]:
    def forward_pre_hook(_modelopt_mask, module, args, kwargs):
        head_mark_in_args = False
        head_mask_in_kwargs = False
        if "head_mask" in kwargs:
            head_mask = kwargs["head_mask"]
            head_mask_in_kwargs = True
        elif len(args) > head_mask_idx:
            head_mask = args[head_mask_idx]
            head_mark_in_args = True
        else:
            head_mask = None

        # head_mask shape: 1 x num_attention_heads x 1 x 1
        # https://github.com/huggingface/transformers/blob/8f7969/src/transformers/modeling_utils.py#L841
        head_mask = _modelopt_mask if head_mask is None else head_mask * _modelopt_mask

        if head_mask_in_kwargs or not head_mark_in_args:
            kwargs["head_mask"] = head_mask
        elif head_mark_in_args:
            args = list(args)
            args[head_mask_idx] = head_mask
            args = tuple(args)

        return args, kwargs

    grad_data = GradientDataManager(
        (module.get_hparam("num_attention_heads").max, 1, 1),
        module,
        reduce_func=lambda x: x.squeeze(),
    )
    hook_handle = module.register_forward_pre_hook(
        partial(forward_pre_hook, grad_data.mask), with_kwargs=True
    )
    return grad_data, hook_handle


def _setup_grad_manager_bert_attention(
    module: "_DynamicBertAttention",
) -> tuple[GradientDataManager, RemovableHandle]:
    # See forward signature here:
    # https://github.com/huggingface/transformers/blob/b86482/src/transformers/models/bert/modeling_bert.py#L415-L424

    return _setup_grad_manager_hf_attention(module, head_mask_idx=2)


def _setup_grad_manager_gptj_attention(
    module: "_DynamicGPTJAttention",
) -> tuple[GradientDataManager, RemovableHandle]:
    # See forward signature here:
    # https://github.com/huggingface/transformers/blob/0ea42e/src/transformers/models/gptj/modeling_gptj.py#L194-L202

    return _setup_grad_manager_hf_attention(module, head_mask_idx=4)


class GradientBinarySearcher(BinarySearcher):
    """Binary searcher for gradient algorithm."""

    SETUP_GRADIENT_FUNC: dict[
        type[DynamicModule], Callable[[DynamicModule], tuple[GradientDataManager, RemovableHandle]]
    ]

    @property
    def default_search_config(self) -> SearchConfig:
        """Get the default config for the searcher."""
        config = super().default_search_config
        config["max_iter_data_loader"] = 128  # Default 50 is not optimal for gradient estimation
        return config

    def before_search(self) -> None:
        """Setup search with gradient-based score."""
        # initialize here to keep DMRegistry empty during imports
        GradientBinarySearcher.SETUP_GRADIENT_FUNC = {
            DMRegistry[nn.Linear]: _setup_grad_manager_linear,
        }
        if HAS_HF:
            GradientBinarySearcher.SETUP_GRADIENT_FUNC.update(
                {
                    DMRegistry[BertAttention]: _setup_grad_manager_bert_attention,
                    DMRegistry[GPTJAttention]: _setup_grad_manager_gptj_attention,
                }
            )
        # TODO: Support distributed models for gradient search
        self.model = unwrap_model(self.model, raise_error=True)

        hps_for_grad_calc = self._estimate_gradient_scores(
            self.config["data_loader"],
            self.config["loss_func"],
            self.config["collect_func"],
            self.config["max_iter_data_loader"],
        )

        # This is a hack so that `sort_parameters` will use `score_tensor`
        # to sort channels/features of the respective hparams in `super().before_search`
        with self._overwrite_hp_importance(hps_for_grad_calc):
            super().before_search()

            # features/heads corresponding to the hparams have already been sorted, now sort the
            # score_tensors
            for hp in hps_for_grad_calc.values():
                assert hasattr(hp, "score_tensor")
                hp.score_tensor = hp.score_tensor.sort(descending=True).values

    def sanitize_search_config(self, config: SearchConfig | None) -> SearchConfig:
        """Sanitize the search config dict."""
        config = config or {}
        if config.get("score_func") is not None:
            warnings.warn("`score_func` should not be provided for `gradnas`.")
        config["score_func"] = GradientBinarySearcher.gradnas_score_func
        config = super().sanitize_search_config(config)
        assert config["data_loader"] is not None, "data_loader must be provided for `gradnas`."
        return config

    @property
    def hparam_names_for_search(self) -> set[str]:
        """We can only optimize over certain types of hparams in gradient binary search."""
        return {"out_features", "num_attention_heads"}

    @staticmethod
    def gradnas_score_func(model: nn.Module) -> float:
        """Score function for `gradnas` algorithm.

        If we prune N neurons from layer L, the total degradation is the sum of degradation values of the
        N pruned neurons. In `fast` algorithm, the degradation due to pruning is estimated directly from
        `validation_score(model after pruning)`. Rest of the algorithm is exactly the same as `fast` algorithm.
        """
        score = 0
        for _, hp in named_hparams(model, configurable=True):
            if hasattr(hp, "score_tensor"):
                # Get the score for the pruned away parameters
                hp_score = torch.sum(hp.score_tensor) - torch.sum(hp.score_tensor[hp.active_slice])
                score = max(score, hp_score)
        return -score

    def _estimate_gradient_scores(
        self,
        data_loader: Iterable,
        loss_func: Callable[[Any, Any], torch.Tensor],
        collect_func: Callable | None = None,
        max_iter_data_loader: int = 128,
    ) -> dict[str, TracedHp]:
        """Estimate gradient scores for the searchable hparams in the model."""
        if collect_func is None:

            def collect_func(data):
                return data[0]

        config = get_subnet_config(self.model)
        sample(self.model, max)

        hp_grad_data: dict[str, GradientDataManager] = {}
        mod_to_hook = {}
        hps_for_grad_calc = self._get_binary_search_hps()
        for hp_name in hps_for_grad_calc:
            module_name = hp_name.rpartition(".")[0]
            module = self.model.get_submodule(module_name)
            hp_grad_data[hp_name], mod_to_hook[hp_name] = (
                GradientBinarySearcher.SETUP_GRADIENT_FUNC[type(module)](module)
            )

        device = get_module_device(self.model)
        for idx, batch in tqdm(
            enumerate(data_loader), desc="Estimating gradient scores", total=max_iter_data_loader
        ):
            args = standardize_model_args(self.model, collect_func(batch), use_kwargs=True)
            args = torch_to(args, device)
            output = self.model(*args[:-1], **args[-1])

            loss = loss_func(output, batch)
            loss.backward()

            for grad_data in hp_grad_data.values():
                grad_data.process_gradient()

            if idx >= max_iter_data_loader:
                break

        # TODO: Normalize the score_tensors so max(sum(hp.score_tensor)) = 1
        for hp_name, grad_data in hp_grad_data.items():
            hps_for_grad_calc[hp_name].score_tensor = grad_data.score

        # remove all forward hooks
        for hook in mod_to_hook.values():
            hook.remove()

        select(self.model, config)

        return hps_for_grad_calc

    @contextmanager
    def _overwrite_hp_importance(self, hps_for_grad_calc: dict[str, TracedHp]):
        """Context manager to overwrite `_get_importance` for hparams in the model."""
        for hp in hps_for_grad_calc.values():
            assert getattr(hp, "score_tensor") is not None
            setattr(hp, "_get_importance_bkp", hp._get_importance)
            hp._get_importance = types.MethodType(lambda x: x.score_tensor, hp)

        yield

        for hp in hps_for_grad_calc.values():
            assert hasattr(hp, "_get_importance_bkp")
            hp._get_importance = hp._get_importance_bkp
            delattr(hp, "_get_importance_bkp")


GradNASConfig: type[ModeloptBaseConfig] = create_model(
    "GradNASConfig",
    **get_kwargs_for_create_model_with_rules(
        registry=DMRegistry,
        default_rules=_get_fastnas_default_rules(),
        doc='Configuration for the ``"gradnas"`` mode.',
    ),
)


@NASModeRegistry.register_mode
@PruneModeRegistry.register_mode
class GradNASModeDescriptor(FastNASModeDescriptor):
    """Class to describe the ``"gradnas"`` mode.

    The properties of this mode can be inspected via the source code.
    """

    @property
    def name(self) -> str:
        """Returns the value (str representation) of the mode."""
        return "gradnas"

    @property
    def config_class(self) -> type[ModeloptBaseConfig]:
        """Specifies the config class for the mode."""
        return GradNASConfig

    @property
    def search_algorithm(self) -> type[BaseSearcher]:
        """Specifies the search algorithm to use for this mode (if any)."""
        return GradientBinarySearcher
