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
# ruff: noqa: E501
"""High-level API to automatically prune and optimize your model with various algorithms."""

from typing import Any

from torch import nn

import modelopt.torch.nas as mtn
from modelopt.torch.opt.mode import ModeLike, _ModeRegistryCls
from modelopt.torch.opt.searcher import ConstraintsDict, SearchConfig

PruneModeRegistry = _ModeRegistryCls("prune")

__all__ = ["prune"]


def prune(
    model: nn.Module,
    mode: ModeLike,
    constraints: ConstraintsDict,
    dummy_input: Any | tuple,
    config: SearchConfig | None = None,
) -> tuple[nn.Module, dict[str, Any]]:
    """Prune a given model by searching for the best architecture within the design space.

    Args:
        model: A standard model that contains standard building blocks to be pruned in-place.
        mode: A (list of) string(s) or Mode(s) or a list of tuples containing the mode and its
            config indicating the desired mode(s) (and configurations) for the convert
            process. Modes set up the model for different algorithms for model optimization. The
            following modes are available:

            *   :class:`"fastnas"<modelopt.torch.prune.fastnas.FastNASModeDescriptor>`: The ``model`` will
                be converted into a search space and set up to automatically perform operations
                required for FastNAS pruning & search. The mode's config
                is described in :class:`FastNASConfig<modelopt.torch.prune.config.FastNASConfig>`.
                This mode is recommended to prune Computer Vision models.
            *   :class:`"gradnas"<modelopt.torch.prune.gradnas.GradNASModeDescriptor>`: The ``model`` will
                be converted into a search space and set up to automatically perform operations
                required for gradient-based pruning & search. The mode's config
                is described in :class:`GradNASConfig<modelopt.torch.prune.config.GradNASConfig>`.
                This mode is recommended to prune Hugging Face language models like BERT and GPT-J.
            *   :class:`"mcore_minitron"<modelopt.torch.prune.plugins.mcore_minitron.MCoreMinitronModeDescriptor>`: The ``model``
                will be converted into a search space and set up to automatically perform operations
                required for Minitron-style pruning & search. The mode's config
                is described in :class:`MCoreMinitronConfig<modelopt.torch.prune.config.MCoreMinitronConfig>`.
                This mode is required to prune NVIDIA Megatron-Core / NeMo GPT-type models.

            If the mode argument is specified as a dictionary, the keys should indicate the mode and
            the values specify the per-mode configuration. If not provided, then default
            configuration would be used.
        constraints: A dictionary mapping constraint names to their respective values that the pruned model must
            satisfy. Currently, the supported constraints are ``flops``, ``params``, and ``export_config``. If the key
            is ``flops`` or ``params``, the value should be an upper bound number or percentage of original. For
            ``export_config``, the value is a dictionary mapping hyperparameter names to their pruned values. For e.g.,:

                .. code-block:: python

                    # Specify a flops upper bound as 4.5 GFLOPs
                    constraints = {"flops": 4.5e6}

                    # Specify a percentage-based constraint
                    # (e.g., search for a model with <= 60% of the original model params)
                    constraints = {"params": "60%"}

                    # Specify export_config with pruned hyperparameters
                    # This is supported and required if the model is converted via ``mcore_minitron`` mode.
                    constraints = {
                        "export_config": {
                            "ffn_hidden_size": 128,
                            "num_attention_heads": 16,
                            "num_query_groups": 4,
                        }
                    }

        dummy_input: Arguments of ``model.forward()``. This is used for exporting and calculating
            inference-based metrics, such as FLOPs. The format of ``dummy_inputs`` follows the
            convention of the ``args`` argument in
            `torch.onnx.export <https://pytorch.org/docs/stable/onnx.html#torch.onnx.export>`_.
            Specifically, ``dummy_input`` can be:

            #. a single argument (``type(dummy_input) != tuple``) corresponding to

               .. code-block:: python

                    model.forward(dummy_input)

            #. a tuple of arguments corresponding to

               .. code-block:: python

                    model.forward(*dummy_input)

            #. a tuple of arguments such that ``type(dummy_input[-1]) == dict`` corresponding to

               .. code-block:: python

                    model.forward(*dummy_input[:-1], **dummy_input[-1])

               .. warning::

                   In this case the model's ``forward()`` method **cannot** contain keyword-only
                   arguments (e.g. ``forward(..., *, kw_only_args)``) or variable keyword arguments
                   (e.g. ``forward(..., **kwargs)``) since these cannot be sorted into positional
                   arguments.

            .. note::

                In order to pass a dict as last non-keyword argument, you need to use a tuple as
                ``dummy_input`` and add an *empty* dict as the last element, e.g.,

                .. code-block:: python

                    dummy_input = (x, {"y": y, "z": z}, {})

                The empty dict at the end will then be interpreted as the keyword args.

            See `torch.onnx.export <https://pytorch.org/docs/stable/onnx.html#torch.onnx.export>`_
            for more info.

            Note that if you provide a ``{arg_name}`` with batch size ``b``, the results will be
            computed based on batch size ``b``.
        config: Additional optional arguments to configure the search. Currently, we support:

            * ``checkpoint``: Path to save/restore checkpoint with dictionary containing intermediate
              search state. If provided, the intermediate search state will be automatically
              restored before search (if exists) and stored/saved during search.
            * ``verbose``: Whether to print detailed search space profiling and search stats during search.
            * ``forward_loop``: A ``Callable`` that takes a model as input and runs a forward loop
              on it. It is recommended to choose the data loader used inside the forward loop
              carefully to reduce the runtime. Cannot be provided at the same time as
              ``data_loader`` and ``collect_func``.
            * ``data_loader``: An iterator yielding batches of data for calibrating the
              normalization layers in the model or compute gradient scores. It is recommended to use
              the same data loader as for training but with significantly fewer iterations. Cannot
              be provided at the same time as ``forward_loop``.
            * ``collect_func``: A ``Callable`` that takes a batch of data from the data loader as
              input and returns the input to ``model.forward()`` as described in
              :meth:`run_forward_loop <modelopt.torch.utils.network.run_forward_loop>`. Cannot
              be provided at the same time as ``forward_loop``.
            * ``max_iter_data_loader``: Maximum number of iterations to run the data loader.
            * ``score_func``: A callable taking the model as input and returning a single accuracy/score
              metric (float). This metric will be maximized during search.

                .. note::

                    The ``score_func`` is required only for ``fastnas`` mode. It will be
                    evaluated on models in eval mode (``model.eval()``).
            * ``loss_func``: A ``Callable`` which takes the model output (i.e output of ``model.forward()``)
              and the batch of data as its inputs and returns a scalar loss.
              This is a required argument if the model is converted via ``gradnas`` mode.

              It should be possible to run a backward pass on the loss value returned by this method.

              ``collect_func`` will be used to gather the inputs to ``model.forward()``
              from a batch of data yielded by``data_loader``.

              ``loss_func`` should support the following usage:

                .. code-block:: python

                    for i, batch in enumerate(data_loader):
                        if i >= max_iter_data_loader:
                            break

                        # Assuming collect_func returns a tuple of arguments
                        output = model(*collect_func(batch))

                        loss = loss_func(output, batch)
                        loss.backward()

            .. note::

                Additional configuration options may be added by individual algorithms. Please
                refer to the documentation of the individual algorithms for more information.

    Returns: A tuple (subnet, state_dict) where
        subnet is the searched subnet (nn.Module), which can be used for subsequent tasks like
        fine-tuning, state_dict contains the history and detailed stats of the search procedure.

    .. note::

        The given model is modified (exported) in-place to match the best subnet found by the
        search algorithm. The returned subnet is thus a reference to the same model instance as the
        input model.
    """
    # apply prune mode(s) to model and convert it to DynamicModule
    model = mtn.convert(model, mode, registry=PruneModeRegistry)

    # now run the search and return the result
    return mtn.search(
        model,
        constraints,
        dummy_input,
        config=config,
    )
