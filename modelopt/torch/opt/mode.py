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

"""Interface and utilities for optimization modes/algorithms.

A mode is a specific type or algorithms for model optimization, e.g., some type of algorithm for
pruning or quantization. It can also specify a single step within an optimization algorithm instead
of the whole algorithm. For example, a mode can prepare a model for pruning or export (i.e. fix the
optimal model configuration) after pruning.

Within ``modelopt``, a ``mode`` constitutes the unit for model optimization. We can define arbitrary
modes, each mode gets recorded in the model's modelopt state dict, and we can define workflows as a
sequence of modes.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, TypeVar

import torch.nn as nn

from modelopt.torch.utils import val2list

from .config import ConfigDict, ModeloptBaseConfig
from .searcher import BaseSearcher

MetadataDict = dict[str, Any]  # metadata dict for one mode
ModeConfigList = list[tuple[str, ConfigDict]]  # config list for multiple modes
ModeState = dict[str, ConfigDict | MetadataDict]  # state dict for one mode

ModeEntrypoint = Callable[
    [nn.Module, ModeloptBaseConfig, MetadataDict], tuple[nn.Module, MetadataDict]
]
ConvertReturnType = tuple[nn.Module, MetadataDict]
_ConvertEntrypoint = Callable[[nn.Module, ModeloptBaseConfig], ConvertReturnType]
_ConvertEntrypointWithKwargs = Callable[[nn.Module, ModeloptBaseConfig, Any], ConvertReturnType]
ConvertEntrypoint = _ConvertEntrypoint | _ConvertEntrypointWithKwargs
RestoreEntrypoint = Callable[[nn.Module, ModeloptBaseConfig, MetadataDict], nn.Module]
UpdateEntrypoint = Callable[[nn.Module, ModeloptBaseConfig, MetadataDict], None]

__all__ = ["ModeDescriptor"]


class ModeDescriptor(ABC):
    """Abstract class to describe a mode."""

    def __str__(self) -> str:
        return str(self.name)

    def __repr__(self) -> str:
        return f"<{type(self).__name__}: '{self.name}'>"

    def __hash__(self):
        return hash(self.name)

    @property
    @abstractmethod
    def name(self) -> str:
        """Returns the string name of the mode."""

    @property
    @abstractmethod
    def config_class(self) -> type[ModeloptBaseConfig]:
        """Specifies the config class for the mode."""

    @property
    def next_modes(self) -> set[str] | None:
        """Modes that must immediately follow this mode.

        Certain modes only makes sense if they are followed by certain other modes.

        An empty set indicates that _no_ mode can follow this mode. A None value indicates that
        there are no restrictions on the following mode.

        Returns:
            A set of mode names that must immediately follow this mode. Defaults to None.
        """
        return None

    @property
    def next_prohibited_modes(self) -> set[str] | None:
        """Modes that should not be applied after this mode.

        This is used to specify that certain modes should not be applied after this mode.
        A ``None`` value indicates that incompatibitly check is skipped for this mode.
        This is useful if there are many allowed modes but only a few prohibited modes.

        Returns:
            A set of mode names that should not be applied after this mode. Defaults to None.
        """
        return None

    @property
    def export_mode(self) -> str | None:
        """The mode that corresponds to the export mode of this mode.

        Certain modes require a subsequent export step. For example, after pruning, we might want to
        fine-tune the model and then export the model. This property specifies that mode if it
        exists.

        None indicates that there exists no such mode.

        Returns:
            The (optional) mode name that corresponds to the export mode of this mode. Defaults to
            None.
        """
        return None

    @property
    def is_export_mode(self) -> bool:
        """Whether the mode is an export mode.

        Returns:
            True if the mode is an export mode, False otherwise. Defaults to False.
        """
        return False

    @property
    def search_algorithm(self) -> type[BaseSearcher]:
        """Specifies the search algorithm to use for this mode (if any)."""
        raise RuntimeError("Search is not supported for this mode.")

    @property
    @abstractmethod
    def convert(self) -> ConvertEntrypoint:
        """The mode's entrypoint for converting a model.

        The function signature of the convert entrypoint is described below:

        Args:
            model: Model to be restored.
            config: Config used for the model that was also used during convert.
            mode_kwargs: Additional keyword arguments to pass to the convert entrypoint.

        Returns:
            A tuple consisting of
                1.  the in-place modified model. If the modification failed, the entrypoint can
                    return None instead
                2.  The config dict that can be used to call the restore entrypoint to instantly *restore*
                    the modified model.
                3.  The metadata that can be used to call the restore entrypoint to instantly
                    *restore* the modified model from the provided initial state, see below's
                    description for the restore entrypoint to get more info about ``metadata``.

        Raises:
            :meth:`ApplyModeError<modelopt.torch.opt._conversion.ApplyModeError>` to indicate that the
            conversion process failed. This error can be caught by user-facing APIs if they want to
            enable a fall-back behavior.
        """

    @property
    @abstractmethod
    def restore(self) -> RestoreEntrypoint:
        """The mode's entrypoint for restoring a model.

        The function signature of the restore entrypoint is described below:

        Args:
            model: Model to be restored.
            config: Config used for the model that was also used during convert.
            metadata: The metadata is used during restoration of the model architecture to instantly
                restore the modified model. The metadata is used on top of the config to ensure that
                the model can be instantly restored/modified from the provided state. This is
                helpful when the ``convert`` entrypoint contains non-deterministic operations whose
                outcome can be stored in the metadata to ensure that the model can be restored
                reliably.A few examples of potential non-deterministic operations are provided
                below:
                (1) Latency measurements: if the conversion leverages latency measurements during
                conversion the conversion process may become non-deterministic.
                (2) Random operations: if the conversion leverages random operations during
                conversion, we should store the samples or random seed.
                (3) Module's train flag: the conversion process might be affected by the module's
                train flag (e.g. tracing is indirectly affected by train flag since the
                forward may be affected by the train flag). If so, we should store the train
                flag in the metadata and set the model into the correct mode.

        Returns:
            The in-place modified and restored model. If the modification failed, the entrypoint can
                    return None instead

        Raises:
            :meth:`ApplyModeError<modelopt.torch.opt._conversion.ApplyModeError>` to indicate that the
            conversion process failed. This error can be caught by user-facing APIs if they want to
            enable a fall-back behavior.
        """

    @property
    def update_for_save(self) -> UpdateEntrypoint:
        """The mode's (optional) entrypoint for updating a model's config and metadata before saving.

        This is useful if metadata or config needs to be updated for saving (and restoring) the mode.

        The function signature of this update entrypoint is described below:

        Args:
            model: Model to be restored.
            config: The config as described above. It should be modified IN-PLACE.
            metadata: The metadata as described above. It should be modified IN-PLACE.

        Returns:
            None.
        """
        return lambda model, config, metadata: None

    @property
    def update_for_new_mode(self) -> UpdateEntrypoint:
        """The mode's (optional) entrypoint for updating a model's config and metadata before a new mode.

        This is useful if metadata or config needs to be updated before adding a new mode. For example, a
        after adding a new mode, the current mode's restore might only need a subset of the metadata/config.

        The function signature of this update entrypoint is described below:

        Args:
            model: Model to be restored.
            config: The config as described above. It should be modified IN-PLACE.
            metadata: The metadata as described above. It should be modified IN-PLACE.

        Returns:
            None.
        """
        return lambda model, config, metadata: None

    @property
    def require_model_like(self) -> bool:
        """Whether the mode requires a ModelLike input?

        Returns:
            False
        """
        return False

    @property
    def save_mode_in_state(self) -> bool:
        """Whether the mode should be saved into the modelopt state.

        This is useful if the mode is intended to be manually re-applied every time it's used.

        Returns:
            True
        """
        return True

    def assert_compatibility_as_next_mode_of(self, other_mode: "ModeDescriptor | str") -> None:
        """Assert that this mode is compatible as a next mode of the other mode."""
        if isinstance(other_mode, str):
            other_mode = _ModeRegistryCls.get_from_any(other_mode)

        if other_mode.next_modes is not None:
            assert str(self) in other_mode.next_modes, (
                f"Cannot add {self} after {other_mode}! Next modes of {other_mode} are {other_mode.next_modes}."
            )

        if other_mode.next_prohibited_modes is not None:
            assert str(self) not in other_mode.next_prohibited_modes, (
                f"Cannot add {self} after {other_mode}! {other_mode} does not allow {self} to be its next mode."
            )


ModeType = ModeDescriptor | str
ModeLike = ModeType | list[ModeType] | ModeConfigList
ModeKwargsType = dict[str, Any] | list[dict[str, Any]]


class _ModeRegistryCls:
    """A registry to keep track of available modes."""

    T = TypeVar("T", bound=ModeDescriptor)

    # global list to keep track of all registries we initialize
    _all_registries: list["_ModeRegistryCls"] = []

    def __init__(self, registry_name: str) -> None:
        """Initialize the registry with the lookup dictionaries."""
        self._registry_name = registry_name  # quantization, distill, nas, prune, speculative, etc.
        self._name2descriptor: dict[str, ModeDescriptor] = {}
        self._all_registries.append(self)

    def register_mode(self, cls_descriptor: type[T]) -> type[T]:
        """Register a new mode with the given descriptor."""
        # initialize descriptor and get name
        descriptor = cls_descriptor()
        name = descriptor.name

        # check if we have a descriptor instance already and use that instance instead.
        if self.contained_in_any(name):
            descriptor = self.get_from_any(name)

        # check if mode_name/value is already taken
        if name in self._name2descriptor:
            raise ValueError(f"Mode {name} already registered: {self._name2descriptor}")

        # register mode
        self._name2descriptor[name] = descriptor
        return cls_descriptor

    def remove_mode(self, mode: ModeType) -> None:
        """Remove a mode from the registry."""
        # remove mode
        del self._name2descriptor[str(mode)]

    def get(self, mode: ModeType) -> ModeDescriptor | None:
        """Get the mode by value or throw an error."""
        return self._name2descriptor.get(str(mode))

    def __getitem__(self, mode: ModeType) -> ModeDescriptor:
        """Get the mode by value or throw an error."""
        return self._name2descriptor[str(mode)]

    def __contains__(self, mode: ModeType) -> bool:
        """Check if mode is registered in this registry."""
        return str(mode) in self._name2descriptor

    def __del__(self) -> None:
        """Remove the registry from the global list."""
        self._all_registries.remove(self)

    @classmethod
    def contained_in_any(cls, mode: ModeType) -> bool:
        """Check if mode is registered in any registry."""
        return any(str(mode) in registry._name2descriptor for registry in cls._all_registries)

    @classmethod
    def get_from_any(cls, mode: ModeType) -> ModeDescriptor:
        """Get the mode by value from any registry or throw a KeyError.

        Adds a sanity check to ensure that the mode is not ambiguous, i.e., there is only one
        instance.
        """
        mode_ds = [registry[mode] for registry in cls._all_registries if mode in registry]
        if not mode_ds:
            raise KeyError(f"Mode {mode} not found in any registry.")
        assert all(mode_ds[0] == m_d for m_d in mode_ds), f"Mode {mode} is ambiguous."
        return mode_ds[0]

    @classmethod
    def get_registry_by_name(cls, registry_name: str) -> "_ModeRegistryCls":
        """Get the registry by name."""
        for registry in cls._all_registries:
            if registry._registry_name == registry_name:
                return registry
        raise KeyError(f"Registry {registry_name} not found.")


def get_mode_config(mode_like: ModeLike) -> ModeConfigList:
    """Standardize mode to ModeConfigDict and return."""
    mode_and_config = [
        ((m, {}) if isinstance(m, str) else (m[0], m[1] or {})) for m in val2list(mode_like)
    ]

    return mode_and_config
