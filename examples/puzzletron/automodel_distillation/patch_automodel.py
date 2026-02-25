# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Runtime patch so NeMo AutoModel.from_pretrained(..., anymodel_descriptor=..., block_configs_path=...)
uses ModelOpt's AnyModel support (ModelDescriptorFactory + deci_x_patcher).

Requires ModelOpt to be installed. Call apply_patch() before loading models; call remove_patch() to restore.
"""

import functools
import json
import logging
import threading
from contextlib import nullcontext
from pathlib import Path

logger = logging.getLogger(__name__)

_anymodel_ctx = threading.local()


def _get_ctx_stack():
    if not hasattr(_anymodel_ctx, "stack"):
        _anymodel_ctx.stack = []
    return _anymodel_ctx.stack


def load_block_configs(block_configs_path: str | Path) -> list[dict]:
    path = Path(block_configs_path)
    if not path.exists():
        raise FileNotFoundError(f"Block configs not found: {path}")
    with open(path) as f:
        out = json.load(f)
    logger.info("Loaded %d block configs from %s", len(out), path)
    return out


def auto_detect_block_configs(checkpoint_dir: str | Path) -> list[dict] | None:
    checkpoint_dir = Path(checkpoint_dir)
    block_configs_path = checkpoint_dir / "block_configs.json"
    if block_configs_path.exists():
        return load_block_configs(block_configs_path)
    return None


def apply_patch() -> None:
    """Patch nemo_automodel so from_pretrained(..., anymodel_descriptor=..., block_configs_path=...)
    uses ModelOpt's deci_x_patcher for heterogeneous (AnyModel) checkpoints.
    """
    import nemo_automodel._transformers.auto_model as _auto_model

    if getattr(_auto_model, "_anymodel_patch_applied", False):
        logger.debug("AutoModel AnyModel patch already applied")
        return

    from modelopt.torch.puzzletron.anymodel import ModelDescriptorFactory, deci_x_patcher

    _orig_init_model = _auto_model._init_model
    _orig_from_pretrained = _auto_model._BaseNeMoAutoModelClass.from_pretrained.__func__

    def _patched_init_model(cls, *model_args, **kwargs):
        stack = _get_ctx_stack()
        block_configs, anymodel_descriptor = stack[-1] if stack else (None, None)

        patcher_ctx = nullcontext()
        if block_configs is not None and anymodel_descriptor is not None:
            descriptor = ModelDescriptorFactory.get(anymodel_descriptor)
            if descriptor is not None:
                patcher_ctx = deci_x_patcher(
                    model_descriptor=descriptor,
                    block_configs=block_configs,
                )
                logger.info(
                    "Using deci_x_patcher with %d heterogeneous layer configs (descriptor=%s)",
                    len(block_configs),
                    anymodel_descriptor,
                )
            else:
                logger.warning(
                    "anymodel_descriptor=%r not found in ModelDescriptorFactory; skipping deci_x_patcher",
                    anymodel_descriptor,
                )

        with patcher_ctx:
            return _orig_init_model(cls, *model_args, **kwargs)

    def _patched_from_pretrained_impl(cls, *args, **kwargs):
        kwargs = dict(kwargs)
        pretrained_model_name_or_path = kwargs.pop("pretrained_model_name_or_path", None)
        anymodel_descriptor = kwargs.pop("anymodel_descriptor", None)
        block_configs_path = kwargs.pop("block_configs_path", None)
        if args:
            pretrained_model_name_or_path = pretrained_model_name_or_path or args[0]
            model_args = args[1:]
        else:
            model_args = ()
        if pretrained_model_name_or_path is None:
            raise TypeError(
                "from_pretrained() missing 1 required argument: 'pretrained_model_name_or_path'"
            )

        block_configs = None
        if anymodel_descriptor is not None:
            if block_configs_path is not None:
                block_configs = load_block_configs(block_configs_path)
            else:
                checkpoint_dir = Path(pretrained_model_name_or_path)
                if checkpoint_dir.is_dir():
                    block_configs = auto_detect_block_configs(checkpoint_dir)
                    if block_configs:
                        logger.info(
                            "Auto-detected %d block configs from %s/block_configs.json",
                            len(block_configs),
                            checkpoint_dir,
                        )

        stack = _get_ctx_stack()
        stack.append((block_configs, anymodel_descriptor))
        kwargs_for_orig = {
            k: v
            for k, v in kwargs.items()
            if k not in ("anymodel_descriptor", "block_configs_path")
        }
        if isinstance(pretrained_model_name_or_path, type):
            raise TypeError(
                "pretrained_model_name_or_path must be a path (str or PathLike), got a type. "
                "Ensure the config model.pretrained_model_name_or_path is the checkpoint path."
            )
        try:
            return _orig_from_pretrained(
                cls,
                pretrained_model_name_or_path,
                *model_args,
                **kwargs_for_orig,
            )
        finally:
            stack.pop()

    class _FromPretrainedDescriptor:
        def __get__(self, obj, owner):
            if owner is None:
                return self
            return functools.partial(_patched_from_pretrained_impl, owner)

    _auto_model._init_model = _patched_init_model
    _auto_model._BaseNeMoAutoModelClass.from_pretrained = _FromPretrainedDescriptor()
    _auto_model._anymodel_patch_applied = True
    _auto_model._anymodel_orig_init_model = _orig_init_model
    _auto_model._anymodel_orig_from_pretrained = _orig_from_pretrained
    logger.info("Applied AnyModel patch to nemo_automodel._transformers.auto_model (ModelOpt)")


def remove_patch() -> None:
    """Restore nemo_automodel to its original state."""
    import nemo_automodel._transformers.auto_model as _auto_model

    if not getattr(_auto_model, "_anymodel_patch_applied", False):
        logger.debug("AutoModel AnyModel patch was not applied")
        return

    _auto_model._init_model = _auto_model._anymodel_orig_init_model
    _auto_model._BaseNeMoAutoModelClass.from_pretrained = _auto_model._anymodel_orig_from_pretrained
    del _auto_model._anymodel_orig_init_model
    del _auto_model._anymodel_orig_from_pretrained
    _auto_model._anymodel_patch_applied = False
    logger.info("Removed AnyModel patch from nemo_automodel._transformers.auto_model")
