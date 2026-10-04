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

"""Linear-attention kernels for quantization.

``fla_chunk_delta_h.py`` and ``fla_chunk_gated_delta_rule.py`` are adapted copies of the chunked
GatedDeltaNet kernels of `flash-linear-attention <https://github.com/fla-org/flash-linear-attention>`_
(``fla.ops.common.chunk_delta_h`` and ``fla.ops.gated_delta_rule.chunk``) that can fake-quantize the
recurrent state carried between chunks to FP8 (``state_qdq``). They still import the surrounding
fla operators, so ``fla-core==0.5.1`` and Triton must be installed to use
them. This package initializer does not import the kernels, so importing it needs neither.
"""
