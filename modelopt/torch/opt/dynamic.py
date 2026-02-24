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

"""Basic dynamic module class and hparam class."""

# cython: annotation_typing = False

import inspect
import warnings
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from itertools import chain
from typing import Any, TypeVar

import torch
import torch.nn as nn
from pydantic import create_model
from torch.fx.proxy import Proxy
from torch.nn.parameter import Parameter

from modelopt.torch.utils import get_unwrapped_name, is_channels_last, unwrap_model
from modelopt.torch.utils.network import bind_forward_method

from .config import ModeloptBaseRule, RulesDict
from .hparam import Hparam

__all__ = ["DynamicModule", "DynamicSpace", "_DMRegistryCls"]

_pytorch_managed = type("_pytorch_managed", (), {})  # pylint: disable=invalid-name
_da_val_default = type("_da_val_default", (), {})  # pylint: disable=invalid-name


DynamicAttributeCallback = Callable[["DynamicModule", Any], Any]
SetAttrHook = Callable[["DynamicModule", str, Any], None]
DelAttrHook = Callable[["DynamicModule", str], None]


class _FoldedCallback:
    """A callback functor that enables folding additional callbacks into an existing callback."""

    def __init__(self, *callbacks: DynamicAttributeCallback):
        self._callbacks: list[DynamicAttributeCallback] = []
        self.extend(callbacks)

    @property
    def callback(self) -> DynamicAttributeCallback:
        """Return the main callback."""
        return self._callbacks[0]

    @callback.setter
    def callback(self, cb: DynamicAttributeCallback):
        """Set the main callback."""
        self._callbacks[0] = cb

    def __iter__(self) -> Iterator[DynamicAttributeCallback]:
        """Iterate over all callbacks."""
        yield from self._callbacks

    def __call__(self, self_module: "DynamicModule", val: Any) -> Any:
        """Call the callback and all other callbacks."""
        if isinstance(val, Proxy):
            return val
        for cb in self:
            val = cb(self_module, val)
        return val

    def __len__(self) -> int:
        """Return the number of callbacks."""
        return len(self._callbacks)

    def extend(self, cbs_other: Iterable[DynamicAttributeCallback]):
        """Extend the list of other callbacks."""
        self._callbacks.extend(cbs_other)


class _DMAttributeManager:
    """A class to manage the special attributes for dynamic modules in a hierarchical fashion.

    Abbreviations:
        - og_cls: original class
        - hp: hparam
        - da: dynamic attribute
        - cb: callback
        - attr: generic temporary attributes

    This class handles all types of special attributes for dynamic modules. It is used to store
    them, manage them, and provide access to them. Moreover, this class stores the hierarchy of
    dynamic module conversions that were applied to the module and can associate the special
    attributes with the correct level of inheritance.

    Specifically, we handle the following types of special attributes:

        1. **Original classes**:
           We maintain a stack of original classes as we stack/convert dynamic modules. As we stack
           dynamic modules on top or remove/export them, we maintain the stack.

        2. **Hparams**:
           We store hparams in a list of dictionaries. This way we can associate each hparam with
           the corresponding level of hierarchy/inheritance. For hparams we do NOT allow adding or
           overwriting hparams that exist in a different level of the stack, i.e., hparams must be
           _unique_ across all levels of inheritance. Note that you can still access hparams from
           any level of inheritance.

        3. **Dynamic attributes**:
           We store dynamic attributes in a dictionary. Additionally, a dynamic attribute can be
           associated with a callback in an _arbitrary_ number of levels in the hierarchy. The
           callbacks are stored in a list of dictionaries. This way we can associate each callback
           with the corresponding level of hierarchy/inheritance and still allow for multiple
           callbacks for the same dynamic attribute.

        4. **Generic attributes**:
           These are generic attributes that are not hparams or dynamic attributes. They are kept
           track of so that we can remove them again upon the final export. They must be unique
           and cannot correspond to an hparam or dynamic attribute.
    """

    def __init__(self):
        self._og_cls_all: list[type[nn.Module]] = []
        self._hp_all: list[dict[str, Hparam]] = []
        self._da: dict[str, Any] = {}
        self._da_cb_all: list[dict[str, _FoldedCallback]] = []
        self._attr: dict[str, tuple[SetAttrHook, DelAttrHook]] = {}

    def __bool__(self) -> bool:
        all_data = [self._og_cls_all, self._da, self._da_cb_all, self._hp_all, self._attr]
        has_state = [bool(x) for x in all_data]
        if not has_state[0]:
            # if we have no original class, we should have no other state either!
            assert not any(has_state), "Inconsistent state for DM attributes!"
        return has_state[0]

    def _get_lookup(self, name: str, lookup_all: list[dict[str, Any]]) -> dict[str, Any]:
        """Return the lookup for the given name."""
        for lookup in lookup_all:
            if name in lookup:
                return lookup
        raise KeyError(f"{name} not found!")

    @property
    def og_cls(self) -> type[nn.Module]:
        """Return the original class of the dynamic module."""
        return self._og_cls_all[-1]

    @property
    def level(self) -> int:
        """Return the current level of stacked inheritance."""
        return len(self._og_cls_all)

    def append_level(self, original_cls: type[nn.Module]):
        """Initialize a new level of special attributes."""
        self._og_cls_all.append(original_cls)
        self._hp_all.append({})
        self._da_cb_all.append({})

    def pop_level(self) -> type[nn.Module]:
        """Remove the last level of special attributes and return the original class."""
        # sanity checks on hparams
        hp_lookup = self._hp_all[-1]
        assert not hp_lookup, "Some hparams were not removed properly!"

        # sanity checks on dynamic attributes and corresponding callbacks
        cb_lookup = self._da_cb_all[-1]
        assert not cb_lookup, "Some dynamic attribute callbacks were not removed properly!"

        da_keys_remaining = set(chain(*self._da_cb_all[:-1]))
        assert da_keys_remaining == self.da_keys(), "Dynamic attributes were not removed properly!"

        # sanity checks on temporary attributes on last level
        if self.level <= 1:
            assert not self._attr, "Some attributes were not removed properly!"

        # after clean-up, we can pop and return the original class
        self._hp_all.pop()
        self._da_cb_all.pop()
        return self._og_cls_all.pop()

    def hp_keys(self, all: bool = True) -> set[str]:
        """Return the keys of current or all hparams."""
        idx_start = 0 if all else -1
        return set(chain(*self._hp_all[idx_start:]))

    def get_hp(self, name: str) -> Hparam:
        """Return the hparam with the given name."""
        hp_lookup = self._get_lookup(name, self._hp_all)
        return hp_lookup[name]

    def named_hps(self, all: bool = True) -> Iterator[tuple[str, Hparam]]:
        """Return the name and hparam for all or current hparams."""
        idx_start = 0 if all else -1
        for hp_lookup in self._hp_all[idx_start:]:
            yield from hp_lookup.items()

    def set_hp(self, name: str, hparam: Hparam):
        """Store hparam by the provided name in the right location."""
        # we do not allow the same hparam in different levels of inheritance
        if name in self.hp_keys():
            assert name in self.hp_keys(all=False), "Hparam already exists in the base cls!"
        self._hp_all[-1][name] = hparam

    def pop_hp(self, name: str) -> Any:
        """Pop hparam by the provided name."""
        hp_lookup = self._get_lookup(name, self._hp_all)
        return hp_lookup.pop(name)

    def da_keys(self) -> set[str]:
        """Return the keys of all dynamic attributes."""
        return set(self._da)

    def set_da(
        self,
        name: str,
        val: Any = _da_val_default,
        cb: DynamicAttributeCallback | None = None,
    ):
        """Store a dynamic attribute together with its callback method."""
        # sanity checks
        if name in self.da_keys():
            val = self.get_da_value(name) if val is _da_val_default else val
        else:
            assert val is not _da_val_default, "Value must be provided for new dynamic attribute!"
            assert cb is not None, "Callback must be provided for new dynamic attribute!"

        # store value
        self._da[name] = val

        # store wrapped callback
        # NOTE: we only allow updating the original callback of this level of inheritance. Updating
        # folded callbacks is not allowed!
        if cb is not None:
            if name in self._da_cb_all[-1]:
                self._da_cb_all[-1][name].callback = cb
            else:
                self._da_cb_all[-1][name] = _FoldedCallback(cb)

    def pop_da(self, name: str) -> Any:
        """Pop dynamic attribute by the provided name."""
        val = self._da.pop(name)
        for cb_lookup in self._da_cb_all:
            cb_lookup.pop(name, None)
        return val

    def get_da_value(self, name: str) -> Any:
        """Return the raw value of the dynamic attribute."""
        return self._da[name]

    def get_da_cb(self, name: str) -> _FoldedCallback:
        """Return the full callback for the dynamic attribute."""
        cbs_all = list(chain.from_iterable(cb_d[name] for cb_d in self._da_cb_all if name in cb_d))
        return _FoldedCallback(*cbs_all)

    @contextmanager
    def retain_cbs(self):
        """Context manager to retain all callback information for dynamic attributes.

        Any changes that are made to the callbacks within this context manager are not retained upon
        exiting the context manager. However, changes to the value of dynamic attributes as well
        as well deletions and additions of dynamic attributes are retained.
        """
        # retain original callbacks while exposing a list of empty dictionaries
        da_cb_all = self._da_cb_all
        da_keys_before = self.da_keys()
        self._da_cb_all = [{} for _ in da_cb_all]
        try:
            yield None
        finally:
            # 1. We will restore callbacks to the original state for da's that still exist
            # 2. We will maintain callbacks for newly added da's
            # 3. We will throw away callbacks for da's that were removed
            da_keys_removed = da_keys_before - self.da_keys()
            da_keys_new = self.da_keys() - da_keys_before
            da_keys_same = da_keys_before & self.da_keys()

            da_cb_all_final = []
            for cb_lookup_before, cb_lookup_now in zip(da_cb_all, self._da_cb_all):
                cb_lookup_new = {k: v for k, v in cb_lookup_now.items() if k in da_keys_new}
                cb_lookup_same = {k: v for k, v in cb_lookup_before.items() if k in da_keys_same}
                cb_lookup_final = {**cb_lookup_new, **cb_lookup_same}
                assert not (cb_lookup_final.keys() & da_keys_removed)  # final sanity check
                da_cb_all_final.append({**cb_lookup_same, **cb_lookup_new})
            self._da_cb_all = da_cb_all_final

    def fold_cbs(self) -> set[str]:
        """Fold all callbacks that appear in lower levels into the corresponding lower level.

        After this call, only dynamic attributes that appear solely in the top-level remain in the
        top-level.

        Returns:
            The set of dynamic attributes that remain in the top-level.
        """
        cb_lookup = self._da_cb_all[-1]

        # check if we can fold the callbacks into a lower level
        for cb_lookup_other in reversed(self._da_cb_all[:-1]):
            for k_other, cb_other in cb_lookup_other.items():
                if k_other in cb_lookup:
                    cb_other.extend(cb_lookup.pop(k_other))

        # sanity check: at this point there should be no overlap between top-level and other levels
        assert not any(k in cb_lookup for k in chain(*self._da_cb_all[:-1])), "Callback overlap!"

        return set(cb_lookup)

    def attr_keys(self) -> set[str]:
        """Return the keys of all or current other attributes."""
        return set(self._attr)

    def get_attr_set_hook(self, name: str) -> SetAttrHook:
        """Return the setter for the given name."""
        return self._attr[name][0]

    def set_attr(self, name: str, set_hook: SetAttrHook | None, del_hook: DelAttrHook | None):
        """Store the name of an attribute in the other list."""
        assert name not in self.attr_keys(), "Attribute already exists!"

        set_hook = set_hook or (lambda m, n, v: None)
        del_hook = del_hook or (lambda m, n: None)
        self._attr[name] = (set_hook, del_hook)

    def pop_attr(self, name: str) -> DelAttrHook:
        """Pop attr by the provided name and return hook for delattr."""
        return self._attr.pop(name)[1]


class DynamicModule(nn.Module):
    """Base class for dynamic modules.

    Dynamic modules are usually extended from ``torch.nn.Module``'s. They
    dynamically support a family of `torch.nn.Module`'s with different architectural
    parameters, such as input/output channel numbers.

    Dynamic modules can also be used to construct the basic searchable unit in a search space with
    the option to select and sample a candidate unit.

    Candidate units are usually described using ``Hparam`` objects and dynamic attributes. Each
    hparam describes a basic searchable unit (e.g. number of output channels in ``DynamicConv2d``).
    Dynamic attributes are callbacks to dynamically construct the attribute depending on the current
    value of the hparam(s), e.g., the ``weight`` tensor in ``DynamicConv2d`` that depends on the
    hparam ``out_channels``.

    In addition, dynamic modules also support registering general attributes that are removed upon
    export the module. This is useful for storing temporary attributes that are not hparams or
    existing attributes that are converted to dynamic attributes.

    For a ``DynamicModule`` class that contains other dynamic modules, the class implementation
    should ensure only to expose ``hparams`` in the outermost class and handle other ``hparams``
    internally including ``hparams`` of child modules that are exposed on their own usually
    (e.g. block module implementations containing DynamicLinear).
    """

    # this is needed to store the special attributes for dynamic modules
    _dm_attribute_manager: _DMAttributeManager

    def __init__(self, *args, **kwargs):
        """Initializing a dynamic module is not allowed!"""
        raise RuntimeError("DynamicModule cannot be initialized directly; use convert instead!")

    def _get_dm_attribute_manager(self, use_default: bool = False) -> _DMAttributeManager:
        """Return the attribute manager or a default one if not available."""
        if "_dm_attribute_manager" not in self.__dict__:
            if use_default:
                return _DMAttributeManager()
            raise AttributeError("DynamicModule.convert() must be called before using the module!")
        return self._dm_attribute_manager

    def _register_hparam(self, name: str, hparam: Hparam):
        """Register hparam by the provided name and do some sanity checks during registering.

        Args:
            name: The name of the hparam. Note that an hparam can be a new attribute or an existing
                attribute that is not already registered as temporary attribute or dynamic
                attribute. In addition, an hparam cannot be registered for tensor and therefore
                neither for parameters nor buffers. Finally, an hparam cannot be registered when an
                hparam under the same exists in one of the parent classes. However, one can re-
                register an hparam of itself.
            hparam: The hparam object to be registered.

        An hparam is useful to configure the dynamic module on the fly. The collection of hparams
        define the configurable space of the dynamic module.
        """
        # retrieve manager
        manager = self._get_dm_attribute_manager()

        # various sanity checks
        if not isinstance(hparam, Hparam):
            raise TypeError(f"Cannot assign {type(hparam)} to {name}. Hparam required!")
        elif name in manager.attr_keys():
            raise KeyError(f"Cannot use the reserved name {name} to assign an hparam!")
        elif name in manager.da_keys():
            raise RuntimeError(f"Cannot register attribute {name} that is dynamic as hparam.")
        elif name in manager.hp_keys(all=True) and name not in manager.hp_keys(all=False):
            raise RuntimeError(f"Cannot overwrite hparam {name} that is an hparam in the base cls.")
        elif "." in name:
            raise KeyError('hparam name can\'t contain "."')
        elif name == "":
            raise KeyError('hparam name can\'t be empty string ""')
        elif name in ["_parameters", "_buffers", "_modules"]:
            raise KeyError(f"Cannot use reserved name {name} to assign hparam.")
        # delete name if it already exists
        elif hasattr(self, name):
            if isinstance(getattr(self, name), (torch.Tensor, torch.nn.Module)):
                raise RuntimeError(f"Cannot register tensor/module attribute {name} as hparam.")
            delattr(self, name)

        # store new hparam
        manager.set_hp(name, hparam)

    def _register_dynamic_attribute(self, name: str, callback: DynamicAttributeCallback):
        """Register a dynamic attribute together with its callback method.

        Args:
            name: The name of the dynamic attribute.
            callback: The callback method that is executed when the dynamic attribute is accessed.
                Generally, this callback can access the whole module (``self``) as well as expect
                to have access to the original attribute without any modifications.

        Note that a dynamic attribute should be an existing attribute that is dynamically affected
        when the value of one or more hparams changes. It is not allowed to register a dynamic
        attribute for a non-existing attribute. However, specifically tensors, parameters, and
        buffers are supported. Moreover, a dynamic attribute can also be registered for an existing
        dynamic attribute in the parent class. In this case, the ``callback`` will be executed
        *after* the callback of the parent class.
        """
        # retrieve manager
        manager = self._get_dm_attribute_manager()

        # various sanity checks
        if not hasattr(self, name):
            raise AttributeError(f"{name} is not a valid attribute.")
        elif name in manager.attr_keys():
            raise KeyError(f"Cannot use the reserved name {name} to assign a dynamic attribute!")
        elif name in manager.hp_keys():
            raise RuntimeError(f"Cannot register attribute {name} that is an hparam as dynamic.")
        elif "." in name:
            raise KeyError('dynamic attribute name can\'t contain "."')
        elif name == "":
            raise KeyError('dynamic attribute name can\'t be empty string ""')

        # however we have to handle regular attributes and params/buffers separately. Specifically,
        # params/buffers won't appear in self.__dict__ since they are already managed by pytorch
        # internally.
        if name in self.__dict__["_parameters"] or name in self.__dict__["_buffers"]:
            # here the attribute is already managed, so simply storing a fake reference.
            value = _pytorch_managed
        elif name in self.__dict__:
            # here, we retrieve the value and delete the attribute from the regular __dict__ so we
            # manage it.
            value = getattr(self, name)
            delattr(self, name)
        elif name in manager.da_keys():
            # here we previously registered the attribute as dynamic
            value = manager.get_da_value(name)
        else:
            raise RuntimeError(f"The value of attribute {name} cannot be retrieved correctly!")

        # store new dynamic attribute
        manager.set_da(name, value, callback)

    def _register_temp_attribute(
        self,
        name: str,
        val: Any,
        set_hook: SetAttrHook | None = None,
        del_hook: DelAttrHook | None = None,
    ):
        """Register a temporary attribute to the instance that is deleted upon the final export.

        Args:
            name: The name of the attribute.
            val: The value of the attribute.
            set_hook: A hook that is executed before the attribute is set (Defaults to
                ``lambda m, n, v: None``).
            del_hook: A hook that is executed before the attribute is deleted (Defaults to
                ``lambda m, n: None``).

        Any attribute that is set like this will be removed upon the final export, i.e., only when
        the module is not a dynamic module anymore after export. This is helpful if you want to
        store extra attributes in the class that are not dynamic attributes or hparams but should
        still be removed upon the final export, e.g., attributes or buffers that may access during
        the dynamic attribute callbacks.

        Unlike dynamic attributes, this can be used to set attributes that did not exist before and
        *cannot* be used to overwrite existing attributes.
        """
        # retrieve manager
        manager = self._get_dm_attribute_manager()

        # sanity checks
        if hasattr(self, name):
            raise KeyError(f"Attribute {name} already exists!")
        elif isinstance(val, Hparam):
            raise RuntimeError("Please register hparam directly via _register_hparam()!")

        # store in manager and set new attribute
        manager.set_attr(name, set_hook, del_hook)
        setattr(self, name, val)

    @torch.no_grad()
    def export(self) -> nn.Module:
        """Export self (a dynamic module) **in-place** and return the exported module.

        The export process will remove the top-level dynamic module and replace it with the original
        module class. Note that the original class may be either another type of dynamic module or
        a vanilla nn.Module. Consequently, any methods (including properties) that are implemented
        in the child class will be removed. Hparams as well as dynamic and temporary attributes are
        handled in a special fashion, see below.

        In order to ensure that the exported module is still consistent there a several mechanisms
        in place to handle hparams, dynamic attributes, and temporary attributes:

        * **Hparams** of the current type are replaced with their currently active value.
            Note that we do not need to explicitly handle hparams of the parent class as they are
            mutually-exclusive, i.e., hparams are unique across all levels of inheritance.
        * **Dynamic Attributes** are handled depending on whether they exist in a parent class:
            1. The same dynamic attribute exists in a parent class. In this case, the callback is
               folded into ("appended to") the callback for the same dynamic attribute of the parent
               class. This way we ensure that the final value of the attribute remains consistent.
            2. The dynamic attribute does not exist in a parent class. In this case, the attribute
               is not dynamic anymore as there are no more callbacks that could affect the value.
               Therefore, we simply overwrite the underlying original object with the current value
               and revert it to a regular attribute.
        * **Temporary Attributes** are kept until the final export, i.e., until the resulting class
          is not a dynamic module anymore. This is to ensure that folded callbacks that may need
          access to these attributes can still access them.
        """
        manager = self._get_dm_attribute_manager()

        # dissolve dynamic attributes when their corresponding callbacks cannot be folded into the
        # parent class callbacks anymore.
        da_removable = manager.fold_cbs()
        for k in da_removable:
            val = getattr(self, k)
            if isinstance(val, torch.Tensor):
                val = val.detach().clone()
                if k in self._parameters:
                    val = Parameter(val)
            manager.pop_da(k)  # remove dynamic attribute from manager
            setattr(self, k, val)  # now we set it as regular attribute/parameter/buffer

        # replace hparams with active values and delete removable attributes in last level
        with self._dict_with_special():  # this way we avoid recursion issues
            for k in manager.hp_keys(all=False):
                val = getattr(self, k)
                manager.pop_hp(k)
                setattr(self, k, val)

            # remove attributes in the last level
            if manager.level <= 1:
                for k in manager.attr_keys():
                    delattr(self, k)

        # pop original class and the whole level of inheritance
        self.__class__ = manager.pop_level()

        # double-check that any remaining dynamic attributes still work
        try:
            for k in manager.da_keys():
                getattr(self, k)
        except Exception as e:
            raise RuntimeError(f"Dynamic attribute {k} unretrievable after export!") from e

        # remove manager if not needed anymore and check whether it should still be a DynamicModule
        is_dynamic = isinstance(self, DynamicModule)
        if manager:
            assert is_dynamic, "Exported module should still be a DynamicModule!"
        else:
            assert not is_dynamic, "Exported module must not be a DynamicModule anymore!"
            delattr(self, "_dm_attribute_manager")

            # If this module had a monkey-patched forward before DynamicModule.convert(), we may have
            # overridden it by binding the dynamic forward onto the instance (to follow the MRO).
            # On final export, restore the original forward to avoid leaking a dynamic forward
            # (e.g., DistillationModel.forward) onto the exported (non-dynamic) module instance.
            # please see: https://github.com/NVIDIA/Model-Optimizer/pull/824
            if hasattr(self, "_forward_pre_dm"):
                setattr(self, "forward", getattr(self, "_forward_pre_dm"))
                delattr(self, "_forward_pre_dm")

        return self

    @classmethod
    @torch.no_grad()
    def convert(cls, module: nn.Module, **setup_kwargs: Any) -> "DynamicModule":
        """Converts a module in-place into its dynamic counterpart by patching its class.

        Args:
            module: The module to be converted into a dynamic module.
            **setup_kwargs: Keyword arguments to be passed to the ``DynamicModule._setup`` method.

        Returns:
            The converted dynamic module.

        This should generally be a *final* method and child classes should inherit ``_setup()``
        instead to customize the conversion process.

        Patching is achieved by updating the ``__class__`` attribute of the module to its dynamic
        counterpart. The dynamic counterpart is a subclass of the original class, hence, we ensure
        the module is fully compatible with the original class. Simultaneously, we can inject the
        corresponding dynamic behavior in a standardized and rigoruos fashion.
        """

        def bind_forward_method_if_needed(self):
            # Modules with monkey patched forward method will not call the correct forward path according to the MRO
            # We will correctly bind the forward method for modules patched by HF accelerate
            # For other cases, we will just warn the user that the module might not work

            if hasattr(self.forward, "__func__") and (
                self.forward.__func__ is self.__class__.forward
            ):
                return

            if hasattr(self, "_hf_hook"):
                # accelerate patched module
                bind_forward_method(self, self.__class__.forward)
            else:
                if not hasattr(self, "_forward_pre_dm"):
                    # Keep the patched forward for downstream modules that want to call it.
                    self._forward_pre_dm = self.forward
                bind_forward_method(self, self.__class__.forward)
                warnings.warn(
                    "Received a module with monkey patched forward method. Dynamic converted module"
                    " might not work."
                )

        # update class
        original_cls = type(module)
        module.__class__ = cls
        assert isinstance(module, cls), f"Failed to convert {original_cls} to {cls}!"  # for mypy

        # setup/update the attribute manager
        if issubclass(original_cls, DynamicModule):
            assert hasattr(module, "_dm_attribute_manager"), "Attribute manager not found!"
        else:
            assert not hasattr(module, "_dm_attribute_manager"), "Attribute manager found!"
            module._dm_attribute_manager = _DMAttributeManager()
        module._dm_attribute_manager.append_level(original_cls)

        bind_forward_method_if_needed(module)

        # setup new hparams and dynamic attributes
        module._setup(**setup_kwargs)

        return module

    def _setup(self, **setup_kwargs: Any):
        """Setup dynamic attributes and hparams after the convert call.

        This method should be overridden by the child class!

        Args:
            **setup_kwargs: Keyword arguments to be passed to the setup method.
        """
        raise NotImplementedError("_setup() must be implemented by child class!")

    def modify(self, *args, **kwargs):
        """Modify the module's dynamic choices in a standardized & scalable fashion.

        This method can be overridden by the child class! While users can also directly modify
        the choices of individual hparams, this method should provide a way to modify a batch of
        dynamic modules with the same arguments, e.g., ``out_features_ratio`` for ``DynamicLinear``.

        Note that arguments of the modify method that are exposed to the user via the rule system
        should be specified as **keyword-only arguments**. When they are exposed as keyword-only
        arguments, the `_DMRegistryCls` can automatically generate the corresponding config class
        on the fly that lets user provide configs and then they are automatically validated before
        being passed to the ``modify`` method.

        If possible, modify()'s keyword arguments should have default values that leave the hparams
        intact if not provided, e.g., one might call ``some_dynamic_module.modify()`` without any
        arguments and the module will remain unchanged.
        """

    def freeze(self):
        """Restrict the hparams of the dynamic module to the original choices.

        This is useful to enforce the behavior of the parent class.

        .. note::

            After this call, the module's hparams can no longer be modified although the underlying
            type is still a dynamic module.
        """
        for _, hp in self.named_hparams(configurable=True):
            hp.active = hp.original
            hp.choices = [hp.original]

    @contextmanager
    def reset_dynamic_attributes(self):
        """Context manager to temporarily remove any dynamic attributes and re-register values.

        This context manager is intended to be used when we want to access a dynamic attribute in
        its original unmodified version, i.e., without this class interfering with its original
        value and its corresponding getattr/setattr/delattr behavior.

        Upon exiting the context manager, the dynamic attributes are re-registered and the same
        callbacks are re-registered together with the new value.
        """
        manager = self._get_dm_attribute_manager()

        with manager.retain_cbs():
            # pop all dynamic attributes and make sure they are accessible in their original form
            da_keys = manager.da_keys()
            for k in da_keys:
                val = manager.pop_da(k)
                if not hasattr(self, k):
                    setattr(self, k, val)
            assert not manager.da_keys(), "Dynamic attributes were not removed properly!"
            try:
                yield None
            finally:
                # re-register dynamic attributes with current values + dummy callback (--> will be
                # thrown away!)
                # NOTE that we respect if attributes were deleted entirely.
                for k in da_keys:
                    if hasattr(self, k):
                        self._register_dynamic_attribute(k, lambda _, v: v)

    @contextmanager
    def _dict_with_special(self):
        """Context manager that checks that __dict__ contains _modules, _parameters, _buffers."""
        # build up set of special keys that we temporarily add to __dict__
        manager = self._get_dm_attribute_manager(use_default=True)
        nn_special = set()
        for key in ["_modules", "_parameters", "_buffers"]:
            if key not in self.__dict__ and (key in manager.da_keys() or key in manager.hp_keys()):
                nn_special.add(key)

        # temporarily add to __dict__
        for key in nn_special:
            self.__dict__[key] = getattr(self, key)
        try:
            yield None
        finally:
            # remove from __dict__
            for key in nn_special:
                del self.__dict__[key]

    def __setattr__(self, name: str, value: Any):
        """Set attr and specifically handle hparams as well as dynamic & temporary attributes."""
        # retrieve manager
        manager = self._get_dm_attribute_manager(use_default=True)

        if isinstance(value, Hparam):
            self._register_hparam(name, value)
        elif name in manager.hp_keys():
            manager.get_hp(name).active = value
        elif name in manager.da_keys() and manager.get_da_value(name) is not _pytorch_managed:
            manager.set_da(name, value)
        else:
            if name in manager.attr_keys():
                manager.get_attr_set_hook(name)(self, name, value)
            with self._dict_with_special():
                super().__setattr__(name, value)

    def __getattr__(self, name: str) -> torch.Tensor | torch.nn.Module:
        """Get attr and specifically handle hparams as well as dynamic & temporary attributes."""
        # retrieve manager
        manager = self._get_dm_attribute_manager(use_default=True)

        # check if we can get value from our hparams
        if name in manager.hp_keys():
            return manager.get_hp(name).active

        # check for dynamic attributes
        if name in manager.da_keys():
            value = manager.get_da_value(name)
            # we might also need to grab value from super call (pytorch managed)
            value = super().__getattr__(name) if value is _pytorch_managed else value
            # apply all callbacks in order
            return manager.get_da_cb(name)(self, value)

        # regular case
        with self._dict_with_special():
            attr = super().__getattr__(name)
        return attr

    def __delattr__(self, name: str):
        """Del an attr and specifically handle hparams as well as dynamic & temporary attributes."""
        manager = self._get_dm_attribute_manager(use_default=True)
        if name in manager.da_keys():
            manager.pop_da(name)
            # check if it still exists
            if hasattr(self, name):
                delattr(self, name)
        elif name in manager.hp_keys():
            manager.pop_hp(name)
        else:
            if name in manager.attr_keys():
                del_hook = manager.pop_attr(name)
                del_hook(self, name)
            return super().__delattr__(name)

    def get_hparam(self, target: str) -> Hparam:
        """Look up and return hparam (like "torch.nn.Module.get_parameter()" but for hparam)."""
        return self._get_dm_attribute_manager().get_hp(target)

    def named_hparams(self, configurable: bool | None = None) -> Iterator[tuple[str, Hparam]]:
        """Return an iterator over all hparams of the module.

        Args:
            configurable: Whether to include configurable hparams.

        Yields:
            (name, Hparam): tuple containing the name and hparam.

        Default behavior is to iterate over configurable and non-configurable hparams. Set
        ``configurable`` accordingly to only iterate over either. If ``configurable`` is set to
        ``True``, only configurable symbols are iterated over. If ``configurable`` is set to
        ``False``, configurable symbols are skipped over (only non-configurable symbols).
        """
        for hp_name, hp in self._get_dm_attribute_manager().named_hps():
            if configurable is None or hp.is_configurable == configurable:
                yield hp_name, hp

    def extra_repr(self):
        """Generate extra_repr making sure all dynamic keys exist in self.__dict__.

        Pytorch heavily uses self.__dict__ to generate extra_repr. However, we remove certain
        attributes from self.__dict__ so we can manage them dynamically. Temporarily, adding them
        back in here and removing them again afterwards.
        """
        added_keys = set()
        manager = self._get_dm_attribute_manager()
        for name in chain(manager.hp_keys(), manager.da_keys()):
            val = getattr(self, name)
            if name in self.__dict__ or val is None:
                continue
            added_keys.add(name)
            self.__dict__[name] = getattr(self, name)

        # make super call
        extra_repr = super().extra_repr()

        # remove keys again
        for key in added_keys:
            del self.__dict__[key]

        return extra_repr

    @property
    def original_cls(self) -> type[nn.Module]:
        """Return the original class of the dynamic module.

        Returns:
            The original class of the dynamic module at the current level.
        """
        return self._get_dm_attribute_manager().og_cls

    def get_original_cls_by_level(self, level: int = -1) -> type[nn.Module]:
        """Return the original class of the dynamic module.

        Args:
            level: The level of inheritance to get the original class from.

        Returns:
            The original class of the dynamic module at the specified level.
        """
        return self._get_dm_attribute_manager()._og_cls_all[level]


class _DMRegistryCls:
    """A registry to keep track of available dynamic modules.

    The registry can also dynamically generate new entries when we have a class that inherits from
    the registered nn.Module.
    """

    T = TypeVar("T", bound=DynamicModule)

    def __init__(self, prefix: str, dm_base_cls: type[DynamicModule] | None = None):
        """Initialize the registry.

        Args:
            prefix: The prefix to use while creating dynamic classes.
            dm_base_cls: The common base class for creating dynamic classes. If None, `DynamicModule`
                is used as the common base class.
        """
        super().__init__()

        self._prefix = prefix  # global prefix to use to register dynamic classes

        self._dm_base_cls = DynamicModule if dm_base_cls is None else dm_base_cls

        self._registry: dict[type[nn.Module], type[DynamicModule]] = {}  # registered classes
        self._key_registry: dict[type[nn.Module], str] = {}  # registered str-keys for classes

        self._dynamic_classes: dict[type[nn.Module], type[DynamicModule]] = {}  # generated DMs
        # generated ModeloptBaseRule classes for the dynamic classes
        self._rule_classes: dict[type[DynamicModule], type[ModeloptBaseRule]] = {}

    def _generate_rule_class(self, dm_cls: type[DynamicModule]) -> type[ModeloptBaseRule]:
        """Generate a ModeloptBaseRule type for the given dynamic module class."""
        spec = inspect.getfullargspec(dm_cls.modify)
        if spec.kwonlyargs:
            assert spec.kwonlydefaults is not None, "modify method should have default arguments!"
            fields = {n: (spec.annotations[n], spec.kwonlydefaults[n]) for n in spec.kwonlyargs}
        else:
            fields = {}
        return create_model(f"{dm_cls.__name__}Config", __base__=ModeloptBaseRule, **fields)

    def _get_dynamic_class_name(self, nn_cls: type[nn.Module]) -> str:
        """Generate a name for the dynamic class."""

        def _is_occupied(name: str) -> bool:
            """Check if a name is already occupied by a dynamic class."""
            return any(name == dm.__name__ for dm in self._dynamic_classes.values())

        # first try: simply prepend self._prefix to the nn_cls name but only if not occupied already
        name = f"{self._prefix}{nn_cls.__name__}"
        if not _is_occupied(name):
            return name

        # 2nd try: add self._prefix and module name to the nn_cls name (should never be occupied!)
        name = f"{nn_cls.__module__.replace('.', '_')}_{self._prefix}{nn_cls.__name__}"
        assert not _is_occupied(name), f"Dynamic class name {name} should not be occupied!"
        return name

    def _get_registered_nn_class(self, nn_cls: type[nn.Module]) -> type[nn.Module] | None:
        """Optionally return the nn module class that should be used to register nn_cls.

        This method loops through the registry to see if there are any nn_cls matches, i.e., any
        subclass with a shared forward method.
        """
        for nn_cls_ in self._registry:
            if issubclass(nn_cls, nn_cls_) and nn_cls.forward is nn_cls_.forward:
                return nn_cls_
        return None

    def __contains__(self, item: nn.Module | type[nn.Module] | str):
        if isinstance(item, str):
            return item in self._key_registry.values()
        nn_cls = type(item) if isinstance(item, nn.Module) else item
        return self._get_registered_nn_class(nn_cls) is not None

    def __getitem__(self, nn_cls: type[nn.Module] | str) -> type[DynamicModule]:
        """Return the dynamic module class for the given nn_cls type or registered string."""
        dm_cls = self.get(nn_cls)
        if dm_cls is None:
            raise KeyError(f"{nn_cls} is not registered for a dynamic module!")
        return dm_cls

    def _create_new_dynamic_class(
        self, dm_class: type, nn_cls: type[nn.Module]
    ) -> type[DynamicModule]:
        """Create a new dynamic class.

        This method creates a new dynamic class that inherits from the registered dynamic class,
        the common dynamic base class and the original nn_cls (in this order).
        """
        clses = [dm_class]
        if not issubclass(clses[-1], self._dm_base_cls):
            clses.append(self._dm_base_cls)
        clses.append(nn_cls)
        name = self._get_dynamic_class_name(nn_cls)
        return type(name, tuple(clses), {})

    def get(self, nn_cls: type[nn.Module] | str, default: Any = None) -> type[DynamicModule] | None:
        """Return the dynamic module class for the given nn_cls type or registered string."""
        # check string case first
        if isinstance(nn_cls, str):
            for nn_cls_, key in self._key_registry.items():
                if key == nn_cls:
                    return self.get(nn_cls_, default)
            return default

        assert issubclass(nn_cls, nn.Module), f"{nn_cls} is not a subclass of nn.Module!"

        # see if dynamic class is already available and return
        if nn_cls in self._dynamic_classes:
            return self._dynamic_classes[nn_cls]

        # see if we can generate a new dynamic class and then return
        nn_cls_ = self._get_registered_nn_class(nn_cls)
        if nn_cls_:
            dm_cls_from_registry = self._registry[nn_cls_]
            dm_class = self._create_new_dynamic_class(dm_cls_from_registry, nn_cls)
            self._dynamic_classes[nn_cls] = dm_class
            self._rule_classes[dm_class] = self._generate_rule_class(dm_class)
            return self.get(nn_cls, default)

        # default return
        return default

    def get_key_from_dm(self, dm_cls: type[DynamicModule] | DynamicModule) -> str:
        """Retrieve the key that is registered for a given dynamic module class."""
        dm_cls = type(dm_cls) if isinstance(dm_cls, DynamicModule) else dm_cls
        for nn_cls, dm_cls_ in self._dynamic_classes.items():
            if dm_cls == dm_cls_:
                return self.get_key(nn_cls)
        raise KeyError(f"{dm_cls} is not registered for a dynamic module!")

    def get_key(self, nn_cls: type[nn.Module] | str) -> str:
        """Retrieve the key that is registered for a given nn module class."""
        # sanity check
        assert nn_cls in self, f"{nn_cls} is not registered for a dynamic module!"

        # handle string case
        if isinstance(nn_cls, str):
            return nn_cls

        # handle module case
        nn_cls_ = self._get_registered_nn_class(nn_cls)
        assert nn_cls_ is not None
        return self._key_registry[nn_cls_]

    def get_rule_class(self, nn_cls: type[nn.Module] | str) -> type[ModeloptBaseRule]:
        """Retrieve the rule config class that is registered for a given nn module class."""
        dm_cls = self.get(nn_cls)
        if dm_cls is None:
            raise KeyError(f"{nn_cls} is not registered for a dynamic module!")
        return self._rule_classes[dm_cls]

    def register(self, cls_to_key: dict[type[nn.Module], str]) -> Callable[[type[T]], type[T]]:
        """Use this to register a new dynamic base module.

        Usage:

        .. code-block:: python

            @DMRegistry.register({nn.Linear: "nn.Linear"})
            class DynamicLinear(DynamicModule):
                pass

        .. note::

            The dynamic base module must **NOT** inherit from the nn.Module that is registered.
            Instead the registry will automatically generate a new class that inherits from both
            the dynamic class (``DynamicLinear`` above) and the nn.Module (``nn.Linear`` above) for
            an MRO that corresponds to ``class AutoGenerated(DynamicLinear, nn.Linear): pass``.
        """

        def decorator(dm_class: type[_DMRegistryCls.T]) -> type[_DMRegistryCls.T]:
            """Register dnn_class with appropriate nn_class."""
            for nn_cls_, key in cls_to_key.items():
                assert nn_cls_ not in self._registry, f"{nn_cls_} already registered!"
                self._registry[nn_cls_] = dm_class
                self._key_registry[nn_cls_] = key
            return dm_class

        return decorator

    def unregister(self, nn_cls: type[nn.Module] | type[T]) -> None:
        """Unregister a previously registered dynamic base module and all its inherited modules.

        It throws a KeyError if the dynamic base module is not registered.
        """
        # 0. sanity check
        if nn_cls not in self._registry:
            raise KeyError(f"{nn_cls} is not registered!")

        # 1. unregister any generated dynamic classes
        for nn_cls_ in list(self._dynamic_classes):
            if nn_cls == self._get_registered_nn_class(nn_cls_):
                dm_cls = self._dynamic_classes.pop(nn_cls_)
                self._rule_classes.pop(dm_cls)

        # 2. unregister the base dynamic class
        self._registry.pop(nn_cls)
        self._key_registry.pop(nn_cls)

    def convert(self, nn_mod: nn.Module, **setup_kwargs: Any) -> DynamicModule:
        """Converts the module into a dynamic module if registered or raise KeyError."""
        return self[type(nn_mod)].convert(nn_mod, **setup_kwargs)

    @property
    def prefix(self) -> str:
        """Return the prefix used for the dynamic classes."""
        return self._prefix


class DynamicSpace:
    """A class to represent all dynamic model choices over a model with multiple submodules."""

    def __init__(self, model: nn.Module) -> None:
        """Initialize the dynamic space from the model."""
        self.model = model

    def _should_be_converted(self, mod: nn.Module) -> bool:
        """Check if the module should be converted."""
        return True

    def convert_to_dynamic(
        self, rules: RulesDict | None, dm_registry: _DMRegistryCls
    ) -> dict[str, nn.Module]:
        """Convert the model to dynamic modules according to the rules and provided registry.

        Args:
            rules: A dictionary containing rules for the dynamic modules.
            dm_registry: A registry containing the dynamic modules to be converted to.

        Returns:
            A dictionary containing the converted modules with submodule names as keys and the
            converted dynamic modules as values.
        """
        # check if the model is DataParallel
        unwrap_model(self.model, raise_error=True)

        # check that it's not channels last
        assert not is_channels_last(self.model)

        # 1. patch the model with dynamic units
        # NOTE: duplicate modules are handled automatically since we convert/patch modules in-place
        mods_converted: dict[str, nn.Module] = {}
        for name, mod in self.model.named_modules():
            if mod in dm_registry and self._should_be_converted(mod):
                dm_registry.convert(mod)
                mods_converted[name] = mod

        # 2. modify search space if rules are provided
        if rules is None:
            return mods_converted

        # change all keys to strings
        rules = {dm_registry.get_key(key): rule for key, rule in rules.items()}

        # iterator through all dynamic modules and modify them according to rules
        for mod_name, mod in mods_converted.items():
            # get the key for this module that is used in the rules
            key = dm_registry.get_key_from_dm(mod)

            # validate + construct custom rules using the rule class stored in the dm_registry
            rule_custom = dm_registry.get_rule_class(key).customize_rule(rules.get(key), mod_name)

            if rule_custom is None:
                # freeze module if no rule is provided
                mod.freeze()
            else:
                # modify module according to rule
                mod.modify(**rule_custom)

        return mods_converted

    def named_dynamic_modules(self) -> Iterator[tuple[str, DynamicModule]]:
        """Recursively yield the name and instance of *all* DynamicModules.

        Yields:
            (name, DynamicModule): tuple containing the name and module.
        """
        for name, module in self.model.named_modules():
            if isinstance(module, DynamicModule):
                yield name, module

    def is_dynamic(self) -> bool:
        """Check if any module is dynamic.

        Returns:
            True if the model contains DynamicModule(s).
        """
        return any(True for _ in self.named_dynamic_modules())

    def named_hparams(
        self, configurable: bool | None = None, unique: bool | None = None
    ) -> Iterator[tuple[str, Hparam]]:
        """Recursively yield the name and instance of *all* hparams.

        Args:
            configurable: Whether to include configurable hparams.
            unique: Whether to include unique hparams. If ``configurable`` is set to ``True``,
                then ``unique`` will be set to ``True`` by default.

        Yields:
            (name, Hparam): tuple containing the name and hparam.

        Default behavior is to iterate over all hparams. If ``configurable`` is set to ``True``,
        only configurable, non-duplicate symbols are iterated over.
        """
        _memo = set()
        assert configurable in [None, True], "Only all or configurable hparams are supported!"
        assert unique in [None, True], "Only all or unique hparams are supported!"
        if configurable:
            unique = True

        for mod_name, mod in self.named_dynamic_modules():
            for hp_name, hp in mod.named_hparams(configurable=configurable):
                if unique is None or hp not in _memo:
                    yield mod_name + ("." if mod_name else "") + hp_name, hp
                    _memo.add(hp)

    def get_hparam(self, name: str) -> Hparam:
        """Get the hparam with the given name."""
        mod_name, _, hp_name = name.rpartition(".")
        mod = self.model.get_submodule(mod_name)
        assert isinstance(mod, DynamicModule), f"Module {mod} must be a DynamicModule!"
        return mod.get_hparam(hp_name)

    def is_configurable(self) -> bool:
        """Check if the model has any configurable hyperparameters.

        Args:
            model: A model to be checked for DynamicModule(s) with configurable hyperparameters.

        Returns:
            True if the model contains DynamicModule(s) with configurable hyperparameters w/ more
            than one choice.
        """
        return any(True for _ in self.named_hparams(configurable=True))

    def size(self) -> int:
        """Get the search space size of the model.

        Returns:
            A int representing the search space size of the model.
        """
        space_size = 1
        for _, hp in self.named_hparams(configurable=True):
            space_size *= len(hp.choices)
        return space_size

    def config(self, configurable: bool | None = None) -> dict[str, Any]:
        """Return the config dict of all hyperparameters.

        Args:
            model: A model that contains DynamicModule(s).
            configurable: None -> all hps, True -> configurable hps, False -> non-configurable hps

        Returns:
            A dict of ``(parameter_name, choice)`` that specifies an active subnet.
        """
        return {
            get_unwrapped_name(name, self.model): hp.active
            for name, hp in self.named_hparams(configurable)
        }

    def select(self, config: dict[str, Any], strict: bool = True) -> None:
        """Select the subnet provided by config.

        If `strict` is set, then `config` must contain the exact set of keys representing both the
        configurable and non-configurable hparams.
        """
        configurables = dict(self.named_hparams(configurable=True))

        # check if we have any overlap with dependent keys
        check_non_configurable = any(
            name in config and name not in configurables for name, hp in self.named_hparams()
        )

        # go through config, select based on current config, and track key errors
        missing_keys = []
        inconsistent_keys = []
        unexpected_keys = dict.fromkeys(config.keys(), True)

        # assign free/searchable hparams from config
        for name, hparam in configurables.items():
            if name in config:
                hparam.active = config[name]
                unexpected_keys[name] = False
            elif strict:
                missing_keys.append(name)

        # do a sanity check on the provided dynamic keys
        if check_non_configurable and strict:
            for name, hparam in self.named_hparams():
                if name in configurables:
                    continue
                if name not in config:
                    missing_keys.append(name)
                    continue
                unexpected_keys[name] = False
                if hparam.active != config[name]:
                    inconsistent_keys.append(
                        f"{name}: active={hparam.active}, config={config[name]}"
                    )

        # raise error for missing and unexpected keys if strict
        unexpected_keys = [k for k, val in unexpected_keys.items() if val]
        error_msg = ""
        if strict and len(missing_keys) > 0:
            error_msg += "\n\t".join(["Missing keys in config for:", *missing_keys])
            error_msg += "\nMake sure all keys are present in config or set strict=False."
            raise RuntimeError(error_msg)
        if strict and len(unexpected_keys) > 0:
            error_msg += "\n\t".join(["Unexpected keys in config for:", *unexpected_keys])
            error_msg += "\nMake sure only required keys are in config or set strict=False."
        if strict and len(inconsistent_keys) > 0:
            error_msg += "\n\t".join(["Inconsistent keys in config:", *inconsistent_keys])
            error_msg += "\nMake sure keys in config are consistent or set strict=False."
        if error_msg:
            raise RuntimeError(f"Error in selecting config:\n{error_msg}")

    def export(self, dm_registry: _DMRegistryCls) -> nn.Module:
        """Recursively export the module including self and return the result.

        Args:
            dm_registry: A dynamic module registry to check for dynamic modules that should be
                exported.

        Returns:
            The model after exporting the dynamic modules found in the registry.
        """

        def _recursive_export(mod: nn.Module) -> None:
            """Recursively export the module."""
            for n, m in mod.named_children():
                if isinstance(m, DynamicModule) and m.original_cls in dm_registry:
                    setattr(mod, n, m.export())  # re-assigning needed for DynamicParallelKDModule
                _recursive_export(m)

        if isinstance(self.model, DynamicModule):
            self.model = self.model.export()
        _recursive_export(self.model)

        return self.model

    def __repr__(self):
        lines = [f"{type(self).__name__}("]
        spaces = "  "
        attr_line = [f"{x}={getattr(self, x)()}" for x in ["is_dynamic", "is_configurable", "size"]]
        lines.append(spaces + ", ".join(attr_line) + ",")
        model_lines = str(self.model).split("\n")
        lines.extend([spaces + ("" if i else "model=") + x for i, x in enumerate(model_lines)])
        lines.append(")")
        return "\n".join(lines)
