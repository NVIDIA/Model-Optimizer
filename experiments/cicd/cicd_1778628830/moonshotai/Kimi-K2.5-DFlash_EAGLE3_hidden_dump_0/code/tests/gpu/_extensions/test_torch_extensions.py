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


import modelopt.torch.quantization.extensions as ext


# Compile extensions first so it does not count towards time used to run a test that needs it
def test_cuda_ext():
    assert ext.get_cuda_ext() is not None


def test_cuda_ext_fp8():
    assert ext.get_cuda_ext_fp8() is not None


def test_cuda_ext_mx():
    assert ext.get_cuda_ext_mx() is not None
