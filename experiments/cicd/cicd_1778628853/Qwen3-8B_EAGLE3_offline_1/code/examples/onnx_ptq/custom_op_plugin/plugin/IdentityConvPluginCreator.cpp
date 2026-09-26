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

// TensorRT IdentityConv plugin creator (factory) implementation.
// Based on https://github.com/leimao/TensorRT-Custom-Plugin-Example.

#include <cstring>
#include <exception>

#include <NvInferRuntimePlugin.h>

#include "IdentityConvPlugin.h"
#include "IdentityConvPluginCreator.h"
#include "PluginUtils.h"

namespace nvinfer1 {
namespace plugin {

REGISTER_TENSORRT_PLUGIN(IdentityConvCreator);

IdentityConvCreator::IdentityConvCreator() {
  mPluginAttributes.clear();
  mPluginAttributes.emplace_back(
      nvinfer1::PluginField("kernel_shape", nullptr, PluginFieldType::kINT32, 2));
  mPluginAttributes.emplace_back(
      nvinfer1::PluginField("strides", nullptr, PluginFieldType::kINT32, 2));
  mPluginAttributes.emplace_back(
      nvinfer1::PluginField("pads", nullptr, PluginFieldType::kINT32, 4));
  mPluginAttributes.emplace_back(
      nvinfer1::PluginField("group", nullptr, PluginFieldType::kINT32, 1));

  mFC.nbFields = mPluginAttributes.size();
  mFC.fields = mPluginAttributes.data();
}

char const *IdentityConvCreator::getPluginName() const noexcept {
  return kIDENTITY_CONV_PLUGIN_NAME;
}

char const *IdentityConvCreator::getPluginVersion() const noexcept {
  return kIDENTITY_CONV_PLUGIN_VERSION;
}

nvinfer1::PluginFieldCollection const *IdentityConvCreator::getFieldNames() noexcept {
  return &mFC;
}

nvinfer1::IPluginV2IOExt *
IdentityConvCreator::createPlugin(char const *name,
                                  nvinfer1::PluginFieldCollection const *fc) noexcept {
  try {
    nvinfer1::PluginField const *fields{fc->fields};
    int32_t nbFields{fc->nbFields};

    PLUGIN_VALIDATE(nbFields == 4);

    int32_t group{};

    for (int32_t i{0}; i < nbFields; ++i) {
      char const *attrName = fields[i].name;
      if (!strcmp(attrName, "group")) {
        PLUGIN_VALIDATE(fields[i].type == nvinfer1::PluginFieldType::kINT32);
        PLUGIN_VALIDATE(fields[i].length == 1);
        group = *(static_cast<int32_t const *>(fields[i].data));
      }
    }

    IdentityConvParameters const params{.group = group};

    IdentityConv *const plugin{new IdentityConv{params}};
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
  } catch (std::exception const &e) {
    caughtError(e);
  }
  return nullptr;
}

nvinfer1::IPluginV2IOExt *IdentityConvCreator::deserializePlugin(char const *name,
                                                                 void const *serialData,
                                                                 size_t serialLength) noexcept {
  try {
    IdentityConv *plugin = new IdentityConv{serialData, serialLength};
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
  } catch (std::exception const &e) {
    caughtError(e);
  }
  return nullptr;
}

} // namespace plugin
} // namespace nvinfer1
