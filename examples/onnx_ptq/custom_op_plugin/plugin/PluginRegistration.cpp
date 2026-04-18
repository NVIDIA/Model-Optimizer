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

// Plugin registration: provides the external C API that TensorRT calls at runtime
// to discover and load plugins from this shared library.
// Based on https://github.com/leimao/TensorRT-Custom-Plugin-Example.

#include <mutex>

#include <NvInferRuntime.h>

#include "IdentityConvPluginCreator.h"

class ThreadSafeLoggerFinder {
public:
  ThreadSafeLoggerFinder() = default;

  void setLoggerFinder(nvinfer1::ILoggerFinder *finder) {
    std::lock_guard<std::mutex> lk(mMutex);
    if (mLoggerFinder == nullptr && finder != nullptr) {
      mLoggerFinder = finder;
    }
  }

  nvinfer1::ILogger *getLogger() noexcept {
    std::lock_guard<std::mutex> lk(mMutex);
    if (mLoggerFinder != nullptr) {
      return mLoggerFinder->findLogger();
    }
    return nullptr;
  }

private:
  nvinfer1::ILoggerFinder *mLoggerFinder{nullptr};
  std::mutex mMutex;
};

ThreadSafeLoggerFinder gLoggerFinder;

extern "C" void setLoggerFinder(nvinfer1::ILoggerFinder *finder) {
  gLoggerFinder.setLoggerFinder(finder);
}

extern "C" nvinfer1::IPluginCreator *const *getPluginCreators(int32_t &nbCreators) {
  nbCreators = 1;
  static nvinfer1::plugin::IdentityConvCreator identityConvCreator{};
  static nvinfer1::IPluginCreator *const pluginCreatorList[] = {&identityConvCreator};
  return pluginCreatorList;
}
