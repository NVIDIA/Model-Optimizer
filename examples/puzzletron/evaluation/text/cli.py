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

"""Select a Puzzletron text-evaluation backend from one entry point."""

from __future__ import annotations

import argparse
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

__all__ = ["build_parser", "main"]


def _load_lmms_main() -> Callable[[list[str] | None], int]:
    # Keep unified help lightweight; the lmms runner imports ModelOpt and Torch.
    from examples.puzzletron.evaluate_lmms_checkpoint import main

    return main


def _load_nemo_main() -> Callable[[Sequence[str] | None], None]:
    # Load the backend only after selection so unified help has no evaluator dependencies.
    from examples.llm_eval.nel_config import main

    return main


def build_parser() -> argparse.ArgumentParser:
    """Build the backend-selection parser."""

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""backend arguments:
  python -m examples.puzzletron.evaluation.text --backend lmms --help
  python -m examples.puzzletron.evaluation.text --backend nemo --help

lmms runs an evaluation and is the default route for supported tasks.
nemo prepares a NeMo Evaluator config for missing or alternate task contracts.
""",
    )
    parser.add_argument(
        "--backend",
        choices=("lmms", "nemo"),
        default="lmms",
        help="Evaluation backend to run or prepare.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch unchanged backend arguments to the selected evaluator interface."""

    arguments = list(argv) if argv is not None else sys.argv[1:]
    parser = build_parser()
    backend_is_explicit = any(
        argument == "--backend" or argument.startswith("--backend=") for argument in arguments
    )
    if not backend_is_explicit and arguments in (["-h"], ["--help"]):
        parser.print_help()
        return 0

    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument("--backend", choices=("lmms", "nemo"), default="lmms")
    selected, backend_arguments = selector.parse_known_args(arguments)
    if selected.backend == "lmms":
        return _load_lmms_main()(backend_arguments)
    _load_nemo_main()(backend_arguments)
    return 0
