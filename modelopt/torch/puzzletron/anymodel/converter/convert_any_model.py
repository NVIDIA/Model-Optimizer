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

"""Convert a HuggingFace model to AnyModel format."""

from pathlib import Path

from modelopt.torch.puzzletron.anymodel.converter.converter import Converter
from modelopt.torch.puzzletron.anymodel.converter.converter_factory import ConverterFactory
from modelopt.torch.puzzletron.anymodel.model_descriptor import ModelDescriptorFactory

__all__ = ["convert_model"]


def convert_model(
    input_dir: str,
    output_dir: str,
    converter: Converter | str,
):
    """Convert a HuggingFace model to AnyModel format.

    This function converts a HuggingFace checkpoint to the AnyModel format used
    for compression. The conversion process:

    1. Copies non-weight files (config, tokenizer, etc.)
    2. Creates block_configs for each layer
    3. Reorganizes weights into subblock checkpoints

    Args:
        input_dir: Path to the input HuggingFace checkpoint directory.
        output_dir: Path to the output AnyModel checkpoint directory.
        converter: Either a converter name (e.g., "llama") or a Converter class.

    Example:
        >>> convert_model(
        ...     input_dir="/path/to/Llama-3.1-8B-Instruct",
        ...     output_dir="/path/to/output/ckpts/teacher",
        ...     converter="llama",
        ... )
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get descriptor and converter from factories (they use the same name)
    descriptor = ModelDescriptorFactory.get(converter)
    converter = ConverterFactory.get(converter)

    converter.convert(descriptor=descriptor, input_dir=input_dir, output_dir=output_dir)


if __name__ == "__main__":
    from fire import Fire

    Fire(convert_model)
