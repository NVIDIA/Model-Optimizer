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

"""Standard command-line arguments shared by ModelOpt tools.

Each ``add_*_args`` function adds one group of arguments to an ``argparse`` parser, so a tool
composes the groups it needs and every tool spells them the same way. Multi-word flags are also
registered with dashes: vLLM's ``FlexibleArgumentParser`` rewrites ``--foo_bar`` to ``--foo-bar``
before matching.
"""

import argparse
from collections.abc import Sequence

__all__ = ["add_model_args"]


def add_model_args(
    parser: argparse.ArgumentParser,
    *,
    role: str | None = None,
    aliases: Sequence[str] = (),
    required: bool = True,
    help_text: str | None = None,
    trust_remote_code: bool = True,
) -> None:
    """Add the model to load, ``--model_name_or_path``, and ``--trust_remote_code`` to *parser*.

    ``--model_name_or_path`` (also ``--model``) takes a Hugging Face model ID or a local
    checkpoint directory and is stored as ``args.model_name_or_path``.

    Args:
        role: For a tool that takes several models: ``role="teacher"`` adds
            ``--teacher_model_name_or_path``, stored as ``args.teacher_model_name_or_path``,
            without the ``--model`` alias.
        aliases: The tool's existing spellings of the flag, so existing command lines keep
            working, e.g. ``("--pyt_ckpt_path",)``.
        trust_remote_code: Whether to add ``--trust_remote_code`` too. Pass ``False`` for a
            second model of the same tool, which already added it.

    Example:
        .. code-block:: python

            parser = argparse.ArgumentParser()
            add_model_args(parser, aliases=("--pyt_ckpt_path",))
            add_model_args(parser, role="teacher", trust_remote_code=False)
            args = parser.parse_args(
                ["--model", "Qwen/Qwen3-8B", "--teacher_model_name_or_path", "Qwen/Qwen3-32B"]
            )
            args.model_name_or_path, args.teacher_model_name_or_path
    """
    dest = f"{role}_model_name_or_path" if role else "model_name_or_path"
    flags = [f"--{dest}", f"--{dest.replace('_', '-')}", *([] if role else ["--model"]), *aliases]
    if repeated := sorted({flag for flag in flags if flags.count(flag) > 1}):
        # argparse accepts the repeat silently, which hides a misspelled alias.
        raise ValueError(f"aliases repeat flags add_model_args already adds: {repeated}")
    subject = f"The {role} model" if role else "The model"
    parser.add_argument(
        *flags,
        dest=dest,
        required=required,
        help=help_text
        or f"{subject} to load: a Hugging Face model ID or a local checkpoint directory.",
    )
    if trust_remote_code:
        parser.add_argument(
            "--trust_remote_code",
            "--trust-remote-code",
            action="store_true",
            help="Run the model's own code from its checkpoint; only for checkpoints you trust.",
        )
