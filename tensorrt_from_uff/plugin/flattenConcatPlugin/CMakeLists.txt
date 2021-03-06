cmake_minimum_required(VERSION 3.2 FATAL_ERROR)

project(FlattenConcat
  LANGUAGES CXX
  VERSION ${TRT_VERSION}
)

message(STATUS "Building for FlattenConcat version: ${TRT_VERSION}, library version: ${TRT_SOVERSION}")

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

# -------- BUILDING --------
set(SHARED_TARGET flattenconcat)
include_directories(${TRT_INCLUDE} ${CUDA_INC_DIR})
add_library(${SHARED_TARGET} MODULE
    FlattenConcat.cpp
)

set_target_properties(${SHARED_TARGET} PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${LIB_DIR}
  VERSION ${TRT_VERSION}
  SOVERSION ${TRT_SOVERSION}
)
# Link TensorRT's nvinfer lib
target_link_libraries(${SHARED_TARGET} PRIVATE ${NVINFER_LIB} ${CUBLAS_LIB})

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
