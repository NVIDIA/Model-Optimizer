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

"""Conf.py file for Sphinx documentation."""

# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))

import os
import sys

import sphinx.application
from docutils import nodes
from docutils.nodes import Element
from sphinx.writers.html5 import HTML5Translator

from modelopt import __version__

sys.path.insert(0, os.path.abspath("../../"))
sys.path.append(os.path.abspath("./_ext"))

# -- Project information -----------------------------------------------------

project = "Model Optimizer"  # pylint: disable=C0103
copyright = "2023-2025, NVIDIA Corporation"  # pylint: disable=C0103
author = "NVIDIA Corporation"  # pylint: disable=C0103
version = __version__

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.githubpages",
    "sphinx.ext.napoleon",  # Support for NumPy and Google style docstrings
    "sphinxarg.ext",  # for command-line help documentation
    "sphinx_copybutton",  # line numbers getting copied so cannot use `:linenos:`
    "sphinx_inline_tabs",
    "sphinx_togglebutton",
    "sphinxcontrib.autodoc_pydantic",
    "modelopt_autodoc_pydantic",
]

# Only show copybutton for python code-blocks
copybutton_selector = ", ".join(
    [
        "div.highlight-python pre",
        "div.highlight-ipython3 pre",
        "div.highlight-bash pre",
        "div.highlight-shell pre",
    ]
)

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []
templates_path = ["_templates"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "style_external_links": True,
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

html_title = f"Model Optimizer {version}"
html_css_files = ["custom.css"]
html_permalinks_icon = "#"  # default icon not rendering properly

# TODO: left here as reference for future
# You can put all these files in the `_static` folder and then activate them as shown below
# html_favicon = "_static/Nvidia_Symbol.png"
# html_theme_options = {
#     "light_logo": "Nvidia_Light.png",
#     "dark_logo": "Nvidia_Dark.png",
# }


# Mock imports for autodoc
autodoc_mock_imports = ["mpi4py", "tensorrt_llm", "triton"]

autosummary_generate = True
autosummary_imported_members = False

# Only consider members from __all__ if available
autosummary_ignore_module_all = False

# Disable docstring inheritance
autodoc_inherit_docstrings = False

# show inheritance
autodoc_default_options = {"show-inheritance": True}

# Omit type package prefixes and type annotations as much as possible to reduce verbosity
add_module_names = False
python_use_unqualified_type_names = True

# Automatically extract typehints when specified and place them in
# descriptions of the relevant function/method.
autodoc_typehints = "description"

# Don't show class signature with the class' name.
autodoc_class_signature = "separated"

# Order of autodoc
# NOTE: summary table on the top of each page does not follow this order so we should also set __all__ in sorted order
autodoc_member_order = "alphabetical"  # can also use `bysource` or `groupwise` to sort members


# autodoc_pydantic model settings
autodoc_pydantic_model_show_config_summary = False
autodoc_pydantic_model_show_validator_summary = False
autodoc_pydantic_model_show_validator_members = False
autodoc_pydantic_model_show_field_summary = False
autodoc_pydantic_model_show_json = True  # we overwrite this to show the schema or default or both
autodoc_pydantic_model_signature_prefix = "ModeloptConfig"
autodoc_pydantic_model_modelopt_show_default_dict = True  # show default inside json
autodoc_pydantic_model_modelopt_show_json_schema = False  # hide json schema

# autodoc_pydantic field settings
autodoc_pydantic_field_swap_name_and_alias = True
autodoc_pydantic_field_doc_policy = "description"
autodoc_pydantic_field_show_alias = False
autodoc_pydantic_field_list_validators = False
autodoc_pydantic_field_show_default = False


class PatchedHTMLTranslator(HTML5Translator):
    """Open all external links in a new tab. Ref: https://stackoverflow.com/a/61669375 ."""

    def visit_reference(self, node: Element) -> None:
        """Visit a reference node."""
        atts = {"class": "reference"}
        if node.get("internal") or "refuri" not in node:
            atts["class"] += " internal"
        else:
            atts["class"] += " external"
            # ---------------------------------------------------------
            # Customize behavior (open in new tab, secure linking site)
            atts["target"] = "_blank"
            atts["rel"] = "noopener noreferrer"
            # ---------------------------------------------------------
        if "refuri" in node:
            atts["href"] = node["refuri"] or "#"
            if self.settings.cloak_email_addresses and atts["href"].startswith("mailto:"):
                atts["href"] = self.cloak_mailto(atts["href"])
                self.in_mailto = True
        else:
            assert "refid" in node, 'References must have "refuri" or "refid" attribute.'
            atts["href"] = "#" + node["refid"]
        if not isinstance(node.parent, nodes.TextElement):
            assert len(node) == 1 and isinstance(node[0], nodes.image)
            atts["class"] += " image-reference"
        if "reftitle" in node:
            atts["title"] = node["reftitle"]
        if "target" in node:
            atts["target"] = node["target"]
        self.body.append(self.starttag(node, "a", "", **atts))

        if node.get("secnumber"):
            self.body.append(("%s" + self.secnumber_suffix) % ".".join(map(str, node["secnumber"])))


def setup(app: sphinx.application.Sphinx) -> None:
    """Setup according to the Sphinx extension API."""
    app.set_translator("html", PatchedHTMLTranslator)
