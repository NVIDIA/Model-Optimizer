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

from datetime import datetime

import lcb_runner.runner.main as main_module
from lcb_runner.lm_styles import LanguageModel, LanguageModelStore, LMStyle
from lcb_runner.runner.main import main
from lcb_runner.runner.parser import get_args


def get_args_override():
    """Patch original get_args function to add the custom model."""
    args = get_args()

    LanguageModelStore[args.model] = LanguageModel(
        args.model,
        args.model,
        LMStyle.OpenAIChat,
        datetime(2024, 1, 1),
        link="localhost",
    )

    return args


main_module.get_args = get_args_override

if __name__ == "__main__":
    main()
