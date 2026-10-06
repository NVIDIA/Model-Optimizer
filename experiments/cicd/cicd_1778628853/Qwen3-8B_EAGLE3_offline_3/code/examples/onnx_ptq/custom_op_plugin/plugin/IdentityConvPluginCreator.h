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

// TensorRT IdentityConv plugin creator (factory) header.
// Based on https://github.com/leimao/TensorRT-Custom-Plugin-Example.

#ifndef TENSORRT_IDENTITY_CONV_PLUGIN_CREATOR_H
#define TENSORRT_IDENTITY_CONV_PLUGIN_CREATOR_H

#include <string>
#include <vector>

#include <NvInferRuntime.h>

namespace nvinfer1 {
namespace plugin {

class IdentityConvCreator : public nvinfer1::IPluginCreator {
public:
  IdentityConvCreator();

  ~IdentityConvCreator() override = default;

  char const *getPluginName() const noexcept override;

  char const *getPluginVersion() const noexcept override;

  nvinfer1::PluginFieldCollection const *getFieldNames() noexcept override;

  nvinfer1::IPluginV2IOExt *
  createPlugin(char const *name, nvinfer1::PluginFieldCollection const *fc) noexcept override;

  nvinfer1::IPluginV2IOExt *deserializePlugin(char const *name, void const *serialData,
                                              size_t serialLength) noexcept override;

  void setPluginNamespace(char const *libNamespace) noexcept override { mNamespace = libNamespace; }

  char const *getPluginNamespace() const noexcept override { return mNamespace.c_str(); }

private:
  nvinfer1::PluginFieldCollection mFC;
  std::vector<nvinfer1::PluginField> mPluginAttributes;
  std::string mNamespace;
};

} // namespace plugin
} // namespace nvinfer1

#endif // TENSORRT_IDENTITY_CONV_PLUGIN_CREATOR_H
