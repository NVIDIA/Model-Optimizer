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

// TensorRT IdentityConv custom plugin header.
// This plugin performs a simple identity (passthrough) operation using CUDA memcpy.
// Based on https://github.com/leimao/TensorRT-Custom-Plugin-Example.

#ifndef TENSORRT_IDENTITY_CONV_PLUGIN_H
#define TENSORRT_IDENTITY_CONV_PLUGIN_H

#include <cuda_runtime.h>

#include <NvInferRuntimePlugin.h>

constexpr char const *const kIDENTITY_CONV_PLUGIN_NAME{"IdentityConv"};
constexpr char const *const kIDENTITY_CONV_PLUGIN_VERSION{"1"};

namespace nvinfer1 {
namespace plugin {

struct IdentityConvParameters {
  int32_t group;
  nvinfer1::DataType dtype;
  int32_t channelSize;
  int32_t height;
  int32_t width;
  size_t dtypeBytes;
};

class IdentityConv : public nvinfer1::IPluginV2IOExt {
public:
  IdentityConv(IdentityConvParameters params);

  IdentityConv(void const *data, size_t length);

  ~IdentityConv() override = default;

  int32_t getNbOutputs() const noexcept override;

  nvinfer1::Dims getOutputDimensions(int32_t index, nvinfer1::Dims const *inputs,
                                     int32_t nbInputDims) noexcept override;

  int32_t initialize() noexcept override;

  void terminate() noexcept override;

  size_t getWorkspaceSize(int32_t maxBatchSize) const noexcept override;

  int32_t enqueue(int32_t batchSize, void const *const *inputs, void *const *outputs,
                  void *workspace, cudaStream_t stream) noexcept override;

  size_t getSerializationSize() const noexcept override;

  void serialize(void *buffer) const noexcept override;

  void configurePlugin(nvinfer1::PluginTensorDesc const *in, int32_t nbInput,
                       nvinfer1::PluginTensorDesc const *out, int32_t nbOutput) noexcept override;

  bool supportsFormatCombination(int32_t pos, nvinfer1::PluginTensorDesc const *inOut,
                                 int32_t nbInputs, int32_t nbOutputs) const noexcept override;

  char const *getPluginType() const noexcept override;

  char const *getPluginVersion() const noexcept override;

  void destroy() noexcept override;

  IPluginV2IOExt *clone() const noexcept override;

  nvinfer1::DataType getOutputDataType(int32_t index, nvinfer1::DataType const *inputType,
                                       int32_t nbInputs) const noexcept override;

  void setPluginNamespace(char const *pluginNamespace) noexcept override;

  char const *getPluginNamespace() const noexcept override;

  bool isOutputBroadcastAcrossBatch(int32_t outputIndex, bool const *inputIsBroadcasted,
                                    int32_t nbInputs) const noexcept override;

  bool canBroadcastInputAcrossBatch(int32_t inputIndex) const noexcept override;

private:
  void deserialize(uint8_t const *data, size_t length);

  IdentityConvParameters mParams;

  char const *mPluginNamespace;
};

} // namespace plugin
} // namespace nvinfer1

#endif // TENSORRT_IDENTITY_CONV_PLUGIN_H
