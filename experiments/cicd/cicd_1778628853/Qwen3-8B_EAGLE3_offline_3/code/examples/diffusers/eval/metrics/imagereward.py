# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from collections import defaultdict

import ImageReward as ImageReward
import numpy as np
from tqdm import tqdm

# Load your model once outside the function
image_reward_model = ImageReward.load("ImageReward-v1.0", device="cuda")


def compute_image_reward_metrics(data):
    scores = defaultdict(list)
    for item in tqdm(data, desc="Computing image reward metrics"):
        prompt = item["prompt"]
        for model_name, image_path in item["images"].items():
            score = image_reward_model.score(prompt, image_path)
            scores[model_name].append(score)

    # Compute the mean for each model
    results = {model_name: np.mean(score_list) for model_name, score_list in scores.items()}
    return results
