cmake_minimum_required(VERSION 3.2 FATAL_ERROR)

project(GridAnchor
  LANGUAGES CXX CUDA
  VERSION ${TRT_VERSION}
)

message(STATUS "Building for GridAnchor version: ${TRT_VERSION}, library version: ${TRT_SOVERSION}")

# Enable all compile warnings
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-long-long -pedantic -Werror")
# Use C++11
set (CMAKE_CXX_STANDARD 11)

# Sets variable to a value if variable is unset.
macro(set_ifndef var val)
    if (NOT ${var})
        set(${var} ${val})
    endif()
    #message(STATUS "Configurable variable ${var} set to ${${var}}")
endmacro()

# -------- CONFIGURATION --------
find_package(CUDA REQUIRED)

set_ifndef(TRT_LIB /usr/lib/x86_64-linux-gnu)
set_ifndef(TRT_INCLUDE /usr/include/x86_64-linux-gnu)
set_ifndef(CUDA_ROOT /usr/local/cuda)

# Find dependencies:
message("\nThe following variables are derived from the values of the previous variables unless provided explicitly:\n")

# TensorRT's nvinfer lib
find_library(_NVINFER_LIB nvinfer HINTS ${TRT_LIB} PATH_SUFFIXES lib lib64)
set_ifndef(NVINFER_LIB ${_NVINFER_LIB})

# cuBLAS
find_library(_CUBLAS_LIB cublas HINTS ${CUDA_ROOT} PATH_SUFFIXES lib lib64)
set_ifndef(CUBLAS_LIB ${_CUBLAS_LIB})

# CUDA include dir
find_path(_CUDA_INC_DIR cuda_runtime_api.h HINTS ${CUDA_ROOT} PATH_SUFFIXES include)
set_ifndef(CUDA_INC_DIR ${_CUDA_INC_DIR})

#set(PLUGIN_SOURCES)
#set(COMMON_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../common)
#add_subdirectory(${COMMON_DIR})

message(STATUS "================================================")
message(STATUS "PROJECT_SOURCE_DIR=${PROJECT_SOURCE_DIR}")
message(STATUS "CMAKE_CURRENT_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "PLUGIN_SOURCES=${PLUGIN_SOURCES}")
message(STATUS "CUDA_TOOLKIT_ROOT_DIR=${CUDA_TOOLKIT_ROOT_DIR}")
find_library(CUDART_LIB cudart HINT ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
find_library(RT_LIB rt)
find_library(CUB_LIB cub)

message(STATUS "CUDART_LIB=${CUDART_LIB}")
message(STATUS "RT_LIB=${RT_LIB}")
message(STATUS "CUB_LIB=${CUB_LIB}")
message(STATUS "================================================")
# -------- BUILDING --------
set(SHARED_TARGET gridanchor)
include_directories(${TRT_INCLUDE} ${CUDA_INC_DIR})
add_library(${SHARED_TARGET} MODULE
    gridAnchorPlugin.cpp
    ${PLUGIN_SOURCES}
)

target_include_directories(${SHARED_TARGET}
  PUBLIC ${CUB_ROOT_DIR}
)

set_target_properties(${SHARED_TARGET} PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${LIB_DIR}
  VERSION ${TRT_VERSION}
  SOVERSION ${TRT_SOVERSION}
)
# Link TensorRT's nvinfer lib
target_link_libraries(${SHARED_TARGET} PRIVATE ${NVINFER_LIB} ${CUBLAS_LIB} ${CUDART_LIB} ${RT_LIB})

find_library (
    CJSON_LIB
    NAMES cJSON
    PATHS ${INSTALL_LIB_DIR}
    )

install(TARGETS ${SHARED_TARGET}
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)
