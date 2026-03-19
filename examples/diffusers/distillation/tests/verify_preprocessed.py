#!/usr/bin/env python3
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

"""Verify that preprocessed data loads correctly."""

import sys

from src.dataset import LatentDataset

data_root = sys.argv[1]
ds = LatentDataset(data_root)
print(f"Dataset: {len(ds)} samples")
sample = ds[0]
for k, v in sample.items():
    print(f"  {k}: shape={tuple(v.shape)} dtype={v.dtype}")
print("Preprocess verification: OK")
