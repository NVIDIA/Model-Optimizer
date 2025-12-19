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

"""Modelopt's pydantic BaseModel used for any type of configuration in algorithms and mode."""

import fnmatch
import json
from collections.abc import Callable, ItemsView, Iterator, KeysView, ValuesView
from importlib.resources import files
from pathlib import Path
from typing import Any, TypeAlias

import pydantic
import yaml
from omegaconf import OmegaConf
from pydantic import (
    BaseModel,
    Field,
    TypeAdapter,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_validator,
)
from pydantic_core import PydanticUndefined

# A simple type alias for a config dictionary that is used as input to initialize a ModeloptBaseConfig.
ConfigDict = dict[str, Any]  # config dict for one mode

# These are a static type annotations for rules that can be used to annotate functions that accept
# various rule dictionaries. Proper type checking is done by the respective rule config class, see
# below.
# NOTE: the ``WrappedRule`` and ``Rule`` types are further refined via
#       ``ModeloptBaseRule.get_rule_type(wrapped_only=True)`` and
#       ``ModeloptBaseRule.get_rule_type(wrapped_only=False)``, respectively.
SimpleRule = dict[str, Any]  # a simple rule type
WrappedRule = dict[str, SimpleRule | None]  # a wrapped rule type with glob patterns
Rule = SimpleRule | WrappedRule | None  # a rule type that can be simple, wrapped, or None
RulesDict = dict[str, Rule]  # a dictionaries containing rules for different types


def ModeloptField(default: Any = PydanticUndefined, **kwargs):  # noqa: N802
    """A pydantic.Field that enforces setting a default value."""
    assert default is not PydanticUndefined, "A default value must be set for ModeloptField."
    return Field(default=default, **kwargs)


# TODO: expand config classes to searcher


class ModeloptBaseConfig(BaseModel):
    """Our config base class for mode configuration.

    The base class extends the capabilities of pydantic's BaseModel to provide additional methods
    and properties for easier access and manipulation of the configuration.
    """

    model_config = pydantic.ConfigDict(extra="forbid", validate_assignment=True)

    def model_dump(self, **kwargs):
        """Dump the config to a dictionary with aliases and no warnings by default."""
        kwargs = {"by_alias": True, "warnings": False, **kwargs}
        return super().model_dump(**kwargs)

    def model_dump_json(self, **kwargs):
        """Dump the config to a json with aliases and no warnings by default."""
        kwargs = {"by_alias": True, "warnings": False, **kwargs}
        return super().model_dump_json(**kwargs)

    @property
    def _iterable_model_extra(self) -> dict[str, Any]:
        """Return model_extra or empty dict if None so we can iterate over it."""
        return self.model_extra or {}

    def get_field_name_from_key(self, key: str) -> str:
        """Get the field name from the given key (can be name or alias of field)."""
        assert isinstance(key, str), f"key must be a string, got {type(key)}"

        if key in type(self).model_fields or key in self._iterable_model_extra:
            return key
        else:
            for name, field_info in type(self).model_fields.items():
                if field_info.alias == key:
                    return name
            raise AttributeError(f"Key {key} not found in the config.")

    def __contains__(self, key: str) -> bool:
        """Check if the given key is present in the config by its actual name or alias."""
        try:
            self.get_field_name_from_key(key)
            return True
        except AttributeError:
            return False

    def __getitem__(self, key: str) -> Any:
        """Get the value for the given key (can be name or alias of field)."""
        return getattr(self, self.get_field_name_from_key(key))

    def __setitem__(self, key: str, value: Any) -> None:
        """Set the value for the given key (can be name or alias of field)."""
        setattr(self, self.get_field_name_from_key(key), value)

    def get(self, key: str, default: Any = None) -> Any:
        """Get the value for the given key (can be name or alias) or default if not found."""
        try:
            return self[key]
        except AttributeError:
            return default

    def __len__(self) -> int:
        """Return the length of the config."""
        return len(self.model_fields) + len(self._iterable_model_extra)

    def __iter__(self) -> Iterator[str]:
        """Iterate over aliases (or name if alias is not defined) of fields."""
        for field_name, field_info in type(self).model_fields.items():
            yield field_info.alias or field_name
        yield from self._iterable_model_extra

    def _get_kv_dict(self) -> dict[str, Any]:
        """Return a dictionary with keys as aliases if possible."""
        return {k: self[k] for k in self}

    def keys(self) -> KeysView[str]:
        """Return the keys (aliases prioritized over names) of the config."""
        return self._get_kv_dict().keys()

    def values(self) -> ValuesView[Any]:
        """Return the values of the config."""
        return self._get_kv_dict().values()

    def items(self) -> ItemsView[str, Any]:
        """Return the items of the config with keys as aliases if possible."""
        return self._get_kv_dict().items()

    def update(self, config: ConfigDict) -> None:
        """Update the config with the given config dictionary."""
        for key, value in config.items():
            self[key] = value


class ModeloptBaseRule(ModeloptBaseConfig):
    """Our base config class for rule-based config classes.

    Rules are what governs the configuration for modifying dynamic module classes.
    """

    @classmethod
    def get_rule_type(cls, wrapped_only: bool = False) -> "TypeAlias":
        """Get the rule type for the given ModeloptBaseConfig."""
        optional_rule_type: TypeAlias = cls | None  # type: ignore[valid-type]
        rule_type = dict[str, optional_rule_type]
        if wrapped_only:
            return rule_type
        return optional_rule_type | rule_type

    @classmethod
    def validate_rule(cls, rule: Rule) -> WrappedRule:
        """Validate a rule with the current cls rule.

        We will check the full rule type (wrapped and unwrapped) and then return the wrapped type.
        """
        # validate the rule with full rule tupe
        TypeAdapter(cls.get_rule_type()).validate_python(rule)

        # we want to reduce it to the wrapped type
        wrapped_rule: WrappedRule
        if rule is None:
            wrapped_rule = {"*": rule}
        else:
            try:
                # if this call doesn't fail, it means that field is a valid rule_type and we
                # need to wrap it.
                cls.model_validate(rule)
                wrapped_rule = {"*": rule}
            except ValidationError:
                wrapped_rule = rule

        # now check wrapped type only
        TypeAdapter(cls.get_rule_type(wrapped_only=True)).validate_python(wrapped_rule)

        return wrapped_rule

    @classmethod
    def customize_rule(cls, rule: Rule, key: str) -> SimpleRule | None:
        """Construct custom rule according to the provided key which is matched."""
        # validate rule first
        rule = cls.validate_rule(rule)

        # create custom rule now
        rule_custom: SimpleRule | None = None
        for pattern, subrule in rule.items():
            if fnmatch.fnmatch(key, pattern):
                rule_custom = (
                    subrule
                    if rule_custom is None or subrule is None
                    else {**rule_custom, **subrule}
                )
        return rule_custom


class ModeloptBaseRuleConfig(ModeloptBaseConfig):
    """Our config base class for mode configuration that are purely made from rules.

    The base class extends the capabilities of pydantic's BaseModel to provide additional methods
    and properties for easier access and manipulation of the configuration.
    """

    model_config = pydantic.ConfigDict(extra="allow")

    @classmethod
    def __init_subclass__(cls, *args, registry, **kwargs):
        cls._registry = registry
        extra_default: dict[str, WrappedRule] = {}
        cls._extra_default = extra_default
        super().__init_subclass__(*args, **kwargs)

    @classmethod
    def register_default(cls, extra_default: dict[str, WrappedRule]) -> None:
        """Register a new default value for the given key."""
        # we don't allow overwriting existing field defaults
        all_keys = cls.model_fields.keys() | {v.alias or k for k, v in cls.model_fields.items()}
        overlap = all_keys & extra_default.keys()
        assert not overlap, f"Updating default values for regular fields not allowed: {overlap}"

        # iterate through provided defaults
        for k, v in extra_default.items():
            # run the rule validator and store
            cls._extra_default[k] = cls._registry.get_rule_class(k).validate_rule(v)

    @classmethod
    def unregister_default(cls, key: str) -> None:
        """Unregister the default value for the given key."""
        cls._extra_default.pop(key)

    @model_validator(mode="after")
    def _check_for_extra_fields(self):
        """Check for extra fields and either apply their field validator or raise an error."""
        _check_field = "_updating_extra_fields"
        if hasattr(self, _check_field):
            # avoid recursion
            return self
        setattr(self, _check_field, True)
        for k, v in {**self._extra_default, **self._iterable_model_extra}.items():
            # try validating and updating the field from registry
            setattr(self, k, self._registry.get_rule_class(k).validate_rule(v))
        delattr(self, _check_field)
        return self


def _get_default_description(
    prefix: str, alias: str, rule_cls: type[ModeloptBaseRule], default: Rule
) -> str:
    prefix = prefix.lower()

    newline_with_indent = "\n" + " " * 4

    default_config = newline_with_indent.join(json.dumps(default, indent=2).splitlines())
    default_config = " " * 4 + default_config
    return f"""Configuration for {prefix} {alias} module.

If the ``"{alias}"`` key is not specified, the default configuration (shown in JSON) will be used:

.. code-block:: json

{default_config}

To deactivate any {prefix} {alias} module, use ``None`` instead of providing a dictionary ``{{}}``.

To specify layer-specific configurations, you can specify a config for each submodule with the key
specifying a glob pattern that matches the submodule name. For example, to convert to a {prefix}
module for all ``{alias}`` layers except for those in the ``"lm_head"`` submodule use:

.. code-block:: python

    {{
        "*": {{...}},
        "*lm_head*": None,
    }}

Note that glob expressions are processed sequentially in the order they are specified. Later keys in
the config will overwrite earlier keys if they match the same submodule name.

If you want to specify the same configuration for all submodules, you can provide an unnested
dictionary as well:

.. code-block:: python

        {{...}}

which is short for

.. code-block:: python

        {{
            "*": {{...}},
        }}
"""


def _get_field_validator(alias: str) -> Callable:
    def _validate_rule_with_correct_signature(cls, field: Any, info: ValidationInfo) -> Any:
        return cls._registry.get_rule_class(alias).validate_rule(field)

    return field_validator(_get_field_name(alias))(_validate_rule_with_correct_signature)


def _get_field_name(alias: str) -> str:
    return alias.replace(".", "_").lower()


# TODO: ideally we annotate the correct type for registry but this would require a refactor since
# we don't wanna introduce the `modelopt.torch.dynamic` dependency here. Leaving this to future work.
def get_kwargs_for_create_model_with_rules(
    registry: Any, default_rules: RulesDict, doc: str
) -> dict[str, Any]:
    """Generate the kwargs for ``pydantic.create_model`` to auto-generate a rule config class.

    Args:
        registry: The dynamic module registry that contains all relevant dynamic modules.
        rule_fields: The fields that the rule-based config class should have.
        doc: The docstring for the rule-based config class.

    A rule-based config class is a config class that purely consists of fields that pertain to
    rules. We can procedurally generate these rule config classes by using

    .. code-block:: python

        from pydantic import create_model

        MyRuleConfigs = create_model(
            "MyRuleConfigs", **get_create_model_kwargs_for_rule_model(registry, rule_fields)
        )

    For more info and example usage, you can take a look at
    :meth:`SparseMagnitudeConfig<modelopt.torch.sparsity.config.SparseMagnitudeConfig>`.

    .. note::

        We have this convenience function in place since autodocs only get generated when
        ``create_model`` is *explicitly* called in the respective config file. So this function is a
        workaround to at least lower the burden of correctly calling ``create_model``.


    """
    # generate fields
    field_specs = {}
    for alias, default in default_rules.items():
        rule_class: type[ModeloptBaseRule] = registry.get_rule_class(alias)
        default_validated = rule_class.validate_rule(default)
        field_specs[_get_field_name(alias)] = (
            rule_class.get_rule_type(),
            ModeloptField(
                alias=alias,
                default=default_validated,
                title=f"{alias} config",
                description=_get_default_description(
                    registry.prefix, alias, rule_class, default_validated
                ),
            ),
        )

    # generate validators
    field_validators = {
        f"_validate_for_{_get_field_name(alias)}": _get_field_validator(alias)
        for alias in default_rules
    }

    return {
        "__base__": ModeloptBaseRuleConfig,
        "__validators__": field_validators,
        "__doc__": doc,
        "__cls_kwargs__": {"registry": registry},
        **field_specs,
    }


BUITIN_CONFIG_PATH = files("modelopt.config")


def load_config(config_name_path: str | Path):
    """Load a config yaml.

    config_name_path:

    Could be an internal predefined config name or a OS file path to a user specified config yaml file.

    A config name is simply the sub path to an built-in config yaml file without the yml/yaml suffix.

    For example, for the built-in FP8 default quantization config, the config name is "quantization/fp8_default".
    It simply points to the config file modelopt/config/quantization/fp8_default.yml

    """
    if isinstance(config_name_path, str):
        if not config_name_path.endswith(".yml") or config_name_path.endswith(".yaml"):
            # config name case
            config_name = config_name_path.lower()
            config_file = BUITIN_CONFIG_PATH.joinpath(f"{config_name}.yml")
            if not config_file.is_file():
                config_file = BUITIN_CONFIG_PATH.joinpath(f"{config_name}.yaml")
        else:
            config_file = Path(config_name_path)
    else:
        config_file = config_name_path

    if not config_file.is_file():
        raise ValueError(f"Cannot find config file of {config_name_path}")

    content = config_file.read_text(encoding="utf-8")

    config_data = yaml.safe_load(content)

    defaults = [load_config(default_cfg) for default_cfg in config_data.pop("defaults", [])]

    defaults.append(config_data)

    return OmegaConf.merge(*defaults)
