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

"""Shared DLA layout classifications for graph transformations."""

__all__ = []

_RANK_AGNOSTIC_OPS = frozenset(
    {
        "Abs",
        "Acos",
        "Acosh",
        "Add",
        "And",
        "Asin",
        "Asinh",
        "Atan",
        "Atanh",
        "Cast",
        "Ceil",
        "Clip",
        "Cos",
        "Cosh",
        "Div",
        "Elu",
        "Equal",
        "Erf",
        "Exp",
        "Floor",
        "Greater",
        "GreaterOrEqual",
        "HardSigmoid",
        "HardSwish",
        "Identity",
        "IsInf",
        "IsNaN",
        "LeakyRelu",
        "Less",
        "LessOrEqual",
        "Log",
        "Max",
        "Mean",
        "Min",
        "Mod",
        "Mul",
        "Neg",
        "Not",
        "Or",
        "PRelu",
        "Pow",
        "Reciprocal",
        "Relu",
        "Round",
        "Selu",
        "Sigmoid",
        "Sign",
        "Sin",
        "Sinh",
        "Softplus",
        "Softsign",
        "Sqrt",
        "Sub",
        "Sum",
        "Tan",
        "Tanh",
        "ThresholdedRelu",
        "Where",
        "Xor",
    }
)
