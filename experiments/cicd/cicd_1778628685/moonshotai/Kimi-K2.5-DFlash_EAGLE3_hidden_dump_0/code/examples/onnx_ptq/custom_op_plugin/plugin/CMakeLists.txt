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

# Based on https://github.com/leimao/TensorRT-Custom-Plugin-Example.

cmake_minimum_required(VERSION 3.18)

project(IDENTITY-CONV-PLUGIN VERSION 0.0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

# TensorRT libraries
find_library(NVINFER_LIB nvinfer HINTS /usr/lib/x86_64-linux-gnu/ PATH_SUFFIXES lib lib64 REQUIRED)
find_library(NVINFER_PLUGIN_LIB nvinfer_plugin HINTS /usr/lib/x86_64-linux-gnu/ PATH_SUFFIXES lib lib64 REQUIRED)

add_library(
    identity_conv_plugin
    SHARED
    PluginUtils.cpp
    IdentityConvPlugin.cpp
    IdentityConvPluginCreator.cpp
    PluginRegistration.cpp
)

target_include_directories(identity_conv_plugin PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(identity_conv_plugin PRIVATE ${NVINFER_LIB} ${NVINFER_PLUGIN_LIB} CUDA::cudart)
