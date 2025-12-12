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

# mypy: ignore-errors
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, ContextManager, Iterable, Mapping, Optional, Union

import torch.nn as nn
from typing_extensions import override

from ..utils import (
    ActivityContext,
    dynamo_skip,
    fake_tensors,
    has_fake_tensor,
    is_submodule_of,
    is_submodule_or_same,
    real_tensors,
)


@dataclass
class InputArgs:
    args: list[Any]
    kwargs: dict[str, Any]

    def __init__(self, *args, **kwargs):
        self.args = list(args)
        self.kwargs = dict(kwargs)

    def __add__(self, other: Any) -> InputArgs:
        assert isinstance(other, InputArgs)
        result = InputArgs(*self.args, *other.args, **{**self.kwargs, **other.kwargs})
        return result

    def drop_args(self, index: int | slice | None = None) -> InputArgs:
        new_args = InputArgs(*self.args, **self.kwargs)
        if index is None:
            new_args.args.clear()
        else:
            del new_args.args[index]

        return new_args

    def drop_kwargs(self, keys: Sequence[str] | None = None) -> InputArgs:
        new_args = InputArgs(*self.args, **self.kwargs)
        if keys is None:
            new_args.kwargs.clear()
        else:
            for key in keys:
                new_args.kwargs.pop(key, None)

        return new_args

    @classmethod
    def from_value(cls, v):
        if isinstance(v, cls):
            return v
        elif isinstance(v, InputArgs):
            return cls(*v.args, **v.kwargs)
        elif isinstance(v, Sequence):
            return cls(*v)
        else:
            return cls(v)


OutputValue = Any


@dataclass
class PassageInputAdapter:
    adapter_fn: Callable[[InputArgs, Optional[str], Optional[nn.Module]], InputArgs]

    def __call__(
        self, original_input: InputArgs, module_name: Optional[str], module: Optional[nn.Module]
    ) -> InputArgs:
        result = self.adapter_fn(original_input, module_name, module)
        return result


@dataclass
class PassageOutputAdapter:
    adapter_fn: Callable[[Any, Optional[str], Optional[nn.Module]], Any]

    def __call__(
        self, original_output: Any, module_name: Optional[str], module: Optional[nn.Module]
    ) -> Any:
        result = self.adapter_fn(original_output, module_name, module)
        return result


class PassageInputOverrides(dict[str, Union[PassageInputAdapter, InputArgs]]):
    def __init__(self, input_overrides: Mapping[str, PassageInputAdapter | InputArgs] = {}):
        for k, v in input_overrides.items():
            self[k] = v

    # def __setitem__(self, key: str, value: InputAdapter | InputArgs) -> None:
    #     if isinstance(key, InputArgs):
    #         def adapter_fn(original_input: InputArgs) -> InputArgs:
    #             assert isinstance(value, InputArgs)
    #             return value
    #         self[key] = InputAdapter(adapter_fn)
    #     else:
    #         self[key] = value


class PassageOutputOverrides(dict[str, Union[PassageOutputAdapter, Any]]):
    def __init__(self, output_overrides: Mapping[str, PassageOutputAdapter | Any] = {}):
        for k, v in output_overrides.items():
            self[k] = v


class NoActivePassageContextError(RuntimeError):
    pass


class RequiredPassageOutputsCapturedSignal(Exception):
    pass


@dataclass
class PassageOutput:
    captured_inputs: dict[str, InputArgs]
    captured_outputs: dict[str, Any]
    captured_fake_outputs: dict[str, Any]
    module_output: Any


Predicate = Callable[[str, nn.Module], bool]


def always_false_predicate(module_name: str, module: nn.Module) -> bool:
    return False


def always_true_predicate(module_name: str, module: nn.Module) -> bool:
    return True


class Passage(nn.Module):
    create_fn_context = ActivityContext[None](max_depth=1)
    active_passages_context = ActivityContext["Passage"](no_duplicates=True, reversed=True)

    def __init__(
        self,
        module: nn.Module,
        *,
        inputs_to_capture: Iterable[str] = [],
        outputs_to_capture: Iterable[str] = [],
        input_overrides: Mapping[str, PassageInputAdapter | InputArgs] = {},
        output_overrides: Mapping[str, PassageOutputAdapter | Any] = {},
        outputs_cache: dict[str, Any] = {},
        capture_fake_outputs_predicate: Predicate = always_false_predicate,
        capture_cache_outputs_predicate: Predicate = always_false_predicate,
        early_exit: bool = False,
        name: Optional[str] = None,
    ):
        super().__init__()

        if not self.create_fn_context.is_active():
            raise RuntimeError("Please use Passage.create(...) in order to create a new Passage")

        self.active_context_manager: Optional[ContextManager] = None

        self.name = name
        self.module = module
        self.module_to_name_mapping = {id(v): k for k, v in module.named_modules()}
        self.inputs_to_capture = set(inputs_to_capture)
        self.outputs_to_capture = set(outputs_to_capture)
        self.input_overrides = input_overrides
        self.output_overrides = output_overrides
        self.outputs_cache = outputs_cache
        self.capture_fake_outputs_predicate = capture_fake_outputs_predicate
        self.capture_cache_outputs_predicate = capture_cache_outputs_predicate
        self.early_exit = early_exit

        self.reset()

    @property
    def input_overrides(self) -> PassageInputOverrides:
        return self._input_overrides

    @input_overrides.setter
    def input_overrides(self, value: Mapping[str, PassageInputAdapter | InputArgs]):
        self._input_overrides = PassageInputOverrides(value)

    @property
    def output_overrides(self) -> PassageOutputOverrides:
        return self._output_overrides

    @output_overrides.setter
    def output_overrides(self, value: Mapping[str, PassageOutputAdapter | Any]):
        self._output_overrides = PassageOutputOverrides(value)

    def reset(self):
        self.required_capture_count = (
            (len(self.inputs_to_capture) + len(self.outputs_to_capture))
            if self.early_exit
            else None
        )
        self.captured_outputs: dict[str, Any] = {}
        self.captured_inputs: dict[str, InputArgs] = {}
        self.captured_fake_outputs: dict[str, Any] = {}

    @classmethod
    def module_name_relative_to_active_passage(cls, module: PatchedModule) -> str:
        root_passage = Passage.active_passages_context.get_active()
        assert root_passage is not None
        module_name = root_passage.module_to_name_mapping[id(module)]
        return module_name

    @classmethod
    def create(
        cls,
        module: nn.Module,
        *,
        inputs_to_capture: Iterable[str] = [],
        outputs_to_capture: Iterable[str] = [],
        input_overrides: Mapping[str, PassageInputAdapter | InputArgs] = {},
        output_overrides: Mapping[str, PassageOutputAdapter | Any] = {},
        outputs_cache: dict[str, Any] = {},
        capture_fake_outputs_predicate: Predicate = always_false_predicate,
        capture_cache_outputs_predicate: Predicate = always_false_predicate,
        early_exit: bool = False,
        name: Optional[str] = None,
    ) -> Passage:
        with cls.create_fn_context(None):
            passage = cls(
                module=module,
                inputs_to_capture=inputs_to_capture,
                outputs_to_capture=outputs_to_capture,
                input_overrides=input_overrides,
                output_overrides=output_overrides,
                outputs_cache=outputs_cache,
                capture_fake_outputs_predicate=capture_fake_outputs_predicate,
                capture_cache_outputs_predicate=capture_cache_outputs_predicate,
                early_exit=early_exit,
                name=name,
            )

        for submodule_name, submodule in module.named_modules(remove_duplicate=False):
            patch_module(submodule_name, submodule)

        # register_passage_hooks(module, descriptor)

        return passage

    def is_active(self) -> bool:
        result = self.active_context_manager is not None
        return result

    def __enter__(self):
        assert self.active_context_manager is None
        self.active_context_manager = Passage.active_passages_context(self)
        self.active_context_manager.__enter__()
        self.module_to_name_mapping = {id(v): k for k, v in self.named_modules()}

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        assert self.active_context_manager is not None
        self.active_context_manager.__exit__(exc_type, exc_val, exc_tb)

    def freeze(self):
        self.eval()
        self.requires_grad_(False)

    def unfreeze(self):
        self.train()
        self.requires_grad_(True)

    def run(self, *args, **kwargs) -> PassageOutput:
        return self(*args, **kwargs)

    @override
    def __call__(self, *args, **kwargs) -> PassageOutput:
        return super().__call__(*args, **kwargs)

    @dynamo_skip
    @override
    def forward(self, *args, **kwargs) -> PassageOutput:
        self.reset()

        with Passage.active_passages_context(self):
            try:
                module_output = self.module(*args, **kwargs)
            except RequiredPassageOutputsCapturedSignal:
                module_output = None

            output = PassageOutput(
                captured_inputs=self.captured_inputs,
                captured_outputs=self.captured_outputs,
                captured_fake_outputs=self.captured_fake_outputs,
                module_output=module_output,
            )

            self.reset()

            return output


class PatchedModule: ...


def patch_module(module_name_: str, module: nn.Module):
    # orig_forward = module.forward

    if isinstance(module, PatchedModule):
        # if module_name != Passage.module_name_relative_to_active_passage(module):
        #     logger.warn(f'Module "{module_name}" already patched for module "{Passage.module_name_relative_to_active_passage(module)}". Could lead to bugs.')
        return

    orig_class = module.__class__

    class PassageModuleWrapper(orig_class, PatchedModule):
        # Defined as a static method to avoid potential collision with original class methods
        @staticmethod
        @dynamo_skip
        def can_be_skipped(_self: PassageModuleWrapper, depth: int) -> bool:
            passages_beyond_depth = Passage.active_passages_context[depth:]
            module_name = Passage.module_name_relative_to_active_passage(_self)

            results = [
                (
                    module_name in passage.outputs_cache
                    and not any(
                        is_submodule_or_same(k, module_name) for k in passage.outputs_to_capture
                    )
                    and not any(
                        is_submodule_of(k, module_name)
                        for k, v in passage.input_overrides.items()
                        if v is not None
                    )
                    and not any(
                        is_submodule_of(k, module_name)
                        for k, v in passage.output_overrides.items()
                        if v is not None
                    )
                )
                for passage in passages_beyond_depth
            ]

            result = all(results)

            return result

        # Defined as a static method to avoid potential collision with original class methods
        @staticmethod
        @dynamo_skip
        def run_passage(_self: PassageModuleWrapper, depth: int, args, kwargs):
            if depth + 1 > len(Passage.active_passages_context):
                output = super(PassageModuleWrapper, _self).__call__(*args, **kwargs)
                return output

            module_name = Passage.module_name_relative_to_active_passage(_self)
            passage = Passage.active_passages_context[depth]

            has_output_override = module_name in passage.output_overrides
            output_override = passage.output_overrides.get(module_name)

            if has_output_override and not isinstance(output_override, PassageOutputAdapter):
                output = output_override
            else:
                input_override = passage.input_overrides.get(module_name)
                if input_override is not None:
                    original_input_args = InputArgs(*args, **kwargs)

                    if isinstance(input_override, PassageInputAdapter):
                        new_input_args = input_override(original_input_args, module_name, module)
                    else:
                        new_input_args = input_override

                    args, kwargs = new_input_args.args, new_input_args.kwargs

                if (
                    output_override is None
                    and PassageModuleWrapper.can_be_skipped(_self, depth)
                    and (has_fake_tensor(args) or has_fake_tensor(kwargs))
                ):
                    cached_output = passage.outputs_cache[module_name]
                    return cached_output

                output = PassageModuleWrapper.run_passage(
                    _self=_self,
                    depth=depth + 1,
                    args=args,
                    kwargs=kwargs,
                )

                if isinstance(output_override, PassageOutputAdapter):
                    output = output_override(output, module_name, module)

            if passage.capture_fake_outputs_predicate(module_name, module):
                fake_output = fake_tensors(output)
                passage.captured_fake_outputs[module_name] = fake_output

            if not module_name in passage.outputs_cache and passage.capture_cache_outputs_predicate(
                module_name, module
            ):
                fake_output = fake_tensors(output)
                passage.outputs_cache[module_name] = fake_output

            if module_name in passage.inputs_to_capture:
                real_args, real_kwargs = real_tensors(args), real_tensors(kwargs)
                passage.captured_inputs[module_name] = InputArgs(*real_args, **real_kwargs)

                if passage.required_capture_count is not None:
                    passage.required_capture_count -= 1

            if module_name in passage.outputs_to_capture:
                real_output = real_tensors(output)
                output_value = real_output
                passage.captured_outputs[module_name] = output_value

                if passage.required_capture_count is not None:
                    passage.required_capture_count -= 1

            if passage.required_capture_count == 0:
                raise RequiredPassageOutputsCapturedSignal()

            return output

        @dynamo_skip
        @override
        def __call__(self, *args, **kwargs):
            output = self.run_passage(
                _self=self,
                depth=0,
                args=args,
                kwargs=kwargs,
            )
            return output

    # module.forward = forward
    PassageModuleWrapper.__name__ = f"ModuleWrapper({module.__class__.__name__})"
    module.__class__ = PassageModuleWrapper
