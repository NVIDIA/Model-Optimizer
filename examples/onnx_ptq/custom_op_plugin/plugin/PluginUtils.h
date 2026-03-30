/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Utility functions for TensorRT plugin error handling and logging.
// Based on https://github.com/leimao/TensorRT-Custom-Plugin-Example.

#ifndef TENSORRT_PLUGIN_UTILS_H
#define TENSORRT_PLUGIN_UTILS_H

#include <exception>

void caughtError(std::exception const &e);

#define PLUGIN_ASSERT(val) reportAssertion((val), #val, __FILE__, __LINE__)
void reportAssertion(bool success, char const *msg, char const *file, int32_t line);

#define PLUGIN_VALIDATE(val) reportValidation((val), #val, __FILE__, __LINE__)
void reportValidation(bool success, char const *msg, char const *file, int32_t line);

#endif // TENSORRT_PLUGIN_UTILS_H
