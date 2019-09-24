#
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

cmake_minimum_required(VERSION 3.2 FATAL_ERROR)

file(STRINGS "/usr/include/x86_64-linux-gnu/NvInfer.h" VERSION_STRINGS REGEX "#define NV_TENSORRT_.*")

foreach(TYPE MAJOR MINOR PATCH BUILD)
  string(REGEX MATCH "NV_TENSORRT_${TYPE} [0-9]" TRT_TYPE_STRING ${VERSION_STRINGS})
  string(REGEX MATCH "[0-9]" TRT_${TYPE} ${TRT_TYPE_STRING})
endforeach(TYPE)

set(TRT_VERSION "${TRT_MAJOR}.${TRT_MINOR}.${TRT_PATCH}.${TRT_BUILD}" CACHE STRING "TRT project version")

foreach(TYPE MAJOR MINOR PATCH)
  string(REGEX MATCH "NV_TENSORRT_SONAME_${TYPE} [0-9]" TRT_TYPE_STRING ${VERSION_STRINGS})
  string(REGEX MATCH "[0-9]" TRT_SO_${TYPE} ${TRT_TYPE_STRING})
endforeach(TYPE)

set(TRT_SOVERSION "${TRT_SO_MAJOR}.${TRT_SO_MINOR}.${TRT_SO_PATCH}" CACHE STRING "TRT library so version")

message(STATUS "Building for TensorRT version: ${TRT_VERSION}, library version: ${TRT_SOVERSION}")

project(TensorRT
        LANGUAGES CXX ${FIND_CUDA}
        VERSION ${TRT_VERSION}
        DESCRIPTION "TensorRT is a C++ library that facilitates high performance inference on NVIDIA GPUs and deep learning accelerators."
        HOMEPAGE_URL "https://github.com/NVIDIA/TensorRT"
)


set(LIB_DIR ${CMAKE_BINARY_DIR})
add_subdirectory(castPlugin)
add_subdirectory(flattenConcatPlugin)
