# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
from pathlib import Path


def test_prepare_dataset_imports_fire_only_for_cli_execution():
    source = Path("modelopt/torch/puzzletron/dataset/prepare_dataset.py").read_text()
    tree = ast.parse(source)

    top_level_fire_imports = [
        node
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
        and (
            (isinstance(node, ast.Import) and any(alias.name == "fire" for alias in node.names))
            or (isinstance(node, ast.ImportFrom) and node.module == "fire")
        )
    ]

    assert top_level_fire_imports == []
    assert "from fire import Fire" in source
