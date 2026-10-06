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

#include <cstring>
#include <sstream>

#include <NvInferRuntime.h>

void caughtError(std::exception const &e) {
  getLogger()->log(nvinfer1::ILogger::Severity::kINTERNAL_ERROR, e.what());
}

void reportAssertion(bool success, char const *msg, char const *file, int32_t line) {
  if (!success) {
    std::ostringstream stream;
    stream << "Assertion failed: " << msg << std::endl
           << file << ':' << line << std::endl
           << "Aborting..." << std::endl;
    getLogger()->log(nvinfer1::ILogger::Severity::kINTERNAL_ERROR, stream.str().c_str());
    std::abort();
  }
}

void reportValidation(bool success, char const *msg, char const *file, int32_t line) {
  if (!success) {
    std::ostringstream stream;
    stream << "Validation failed: " << msg << std::endl << file << ':' << line << std::endl;
    getLogger()->log(nvinfer1::ILogger::Severity::kINTERNAL_ERROR, stream.str().c_str());
  }
}
