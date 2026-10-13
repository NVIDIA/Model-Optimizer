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

import torch
from torchmetrics.multimodal import CLIPImageQualityAssessment, CLIPScore
from tqdm import tqdm
from utils import convert_img2tensor, reorganize_data


def compute_clip_iqa(data, device: str = "cuda"):
    results = defaultdict(list)
    reorg_data = reorganize_data(data)
    for model_name in reorg_data:
        clip_model = CLIPImageQualityAssessment(
            model_name_or_path="openai/clip-vit-large-patch14"
        ).to(device)
        for entry in tqdm(reorg_data[model_name], desc=f"Computing CLIP-IQA for {model_name}"):
            img_path = entry["image"]
            image_tensor = convert_img2tensor(img_path)
            clip_model.update(image_tensor.to(torch.float32).to(device).unsqueeze(0))
        results[model_name] = clip_model.compute().mean().item()
    return {"CLIP-IQA": results}


def compute_clip(data, device: str = "cuda"):
    results = defaultdict(list)
    reorg_data = reorganize_data(data)
    for model_name in reorg_data:
        clip_model = CLIPScore(model_name_or_path="openai/clip-vit-large-patch14").to(device)
        for entry in tqdm(reorg_data[model_name], desc=f"Computing CLIP for {model_name}"):
            prompt = entry["prompt"]
            img_path = entry["image"]
            image_tensor = convert_img2tensor(img_path)
            clip_model.update(image_tensor.to(torch.float32).to(device).unsqueeze(0), prompt)
        results[model_name] = clip_model.compute().mean().item()
    return {"CLIP": results}
