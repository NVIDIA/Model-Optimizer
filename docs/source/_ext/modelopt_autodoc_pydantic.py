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

"""A small extension on top of the autodoc_pydantic extension to show default configs."""

import json
import types
from collections.abc import Callable
from contextlib import contextmanager, nullcontext
from typing import Any

from sphinx.application import Sphinx
from sphinxcontrib.autodoc_pydantic import __version__
from sphinxcontrib.autodoc_pydantic.directives.autodocumenters import (
    PydanticFieldDocumenter,
    PydanticModelDocumenter,
)
from sphinxcontrib.autodoc_pydantic.directives.templates import to_collapsable


def _wrap_into_collabsable(summary: str):
    """Decorator to wrap the lines written by the function into a collapsable block."""

    # patch self.add_line to intercept the lines instead of writing them to the output
    def add_line_intercepted(
        self: "ModeloptPydanticModelDocumenter", line: str, source: str, *lineno: int
    ):
        if self._source_intercepted is None:
            self._source_intercepted = source
        else:
            assert self._source_intercepted == source, "Only one source supported!"
        assert not lineno, "No optional lineno argument supported!"
        self._lines_intercepted.append(line)

    def decorator(func: Callable) -> Callable:
        def func_with_collabsable(self: "ModeloptPydanticModelDocumenter", *args, **kwargs) -> Any:
            # patch the add_line method
            self._lines_intercepted = []
            self._source_intercepted = None
            self.add_line_unpatched = self.add_line.__func__
            self.add_line = types.MethodType(add_line_intercepted, self)

            # run method
            ret = func(self, *args, **kwargs)

            # clean up
            lines = self._lines_intercepted
            source = self._source_intercepted
            self.add_line = types.MethodType(self.add_line_unpatched, self)
            del self._lines_intercepted
            del self._source_intercepted
            del self.add_line_unpatched

            # check if we have any lines to wrap
            if lines:
                lines = to_collapsable(lines, summary, "autodoc_pydantic_collapsable_json")
                for line in lines:
                    self.add_line(line, source)

            # return the original return value
            return ret

        return func_with_collabsable

    return decorator


def _skip_lines(func: Callable) -> Callable:
    def func_with_skip_lines(self: "ModeloptPydanticModelDocumenter", *args, **kwargs) -> Any:
        # patch the add_line method
        self.add_line_unpatched = self.add_line.__func__
        self.add_line = types.MethodType(lambda self, *args, **kwargs: None, self)

        # run method
        ret = func(self, *args, **kwargs)

        # clean up
        self.add_line = types.MethodType(self.add_line_unpatched, self)
        del self.add_line_unpatched

        # return the original return value
        return ret

    return func_with_skip_lines


class ModeloptPydanticModelDocumenter(PydanticModelDocumenter):
    """We add the option to print defaults."""

    def __init__(self, *args: Any) -> None:
        super().__init__(*args)
        exclude_members = self.options["exclude-members"]
        exclude_members.add("model_computed_fields")

    @_wrap_into_collabsable("Show default config as JSON")
    def add_default_dict(self) -> None:
        # we use sanitized schema which means errors in the schema are ignored
        schema = self.pydantic.inspect.schema.sanitized
        config = {k: v.get("default") for k, v in schema["properties"].items()}

        # create valid rst lines from the config
        config_json = json.dumps(config, default=str, indent=3)
        lines = [f"   {line}" for line in config_json.split("\n")]
        lines = [":Default config (JSON):", "", ".. code-block:: json", "", *lines, ""]

        # add lines to autodoc
        source_name = self.get_sourcename()
        for line in lines:
            self.add_line(line, source_name)

    def add_collapsable_schema(self):
        if self.pydantic.options.is_true("modelopt-show-default-dict", True):
            self.add_default_dict()
        with (
            nullcontext()
            if self.pydantic.options.is_true("modelopt-show-json-schema", True)
            else self._skip_lines()
        ):
            super().add_collapsable_schema()

    @_wrap_into_collabsable("Show field summary")
    def add_field_summary(self):
        return super().add_field_summary()

    @contextmanager
    def _skip_lines(self):
        self.add_line_unpatched = self.add_line.__func__
        self.add_line = types.MethodType(lambda self, *args, **kwargs: None, self)
        yield
        self.add_line = types.MethodType(self.add_line_unpatched, self)
        del self.add_line_unpatched


class ModeloptPydanticFieldDocumenter(PydanticFieldDocumenter):
    """Collabsing field doc string by default and only showing summary."""

    @_wrap_into_collabsable("Show details")
    def add_description(self):
        super().add_description()


def setup(app: Sphinx) -> dict[str, Any]:
    """Setup the extension."""
    # we have one new option that we enable
    app.add_config_value("autodoc_pydantic_model_modelopt_show_default_dict", True, True, bool)
    app.add_config_value("autodoc_pydantic_model_modelopt_show_json_schema", True, True, bool)

    # we require this extension to be loaded first
    app.setup_extension("sphinx.ext.autodoc")
    app.setup_extension("sphinxcontrib.autodoc_pydantic")

    # we modify the model and field documenter on top of autodoch_pydantic
    app.add_autodocumenter(ModeloptPydanticModelDocumenter, override=True)
    app.add_autodocumenter(ModeloptPydanticFieldDocumenter, override=True)

    return {
        "version": __version__,
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
