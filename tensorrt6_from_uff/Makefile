SHELL=/bin/bash -o pipefail
TARGET?=$(shell uname -m)
LIBDIR?=lib
VERBOSE?=0
ifeq ($(VERBOSE), 1)
AT=
else
AT=@
endif
CUDA_TRIPLE=x86_64-linux
CUBLAS_TRIPLE=x86_64-linux-gnu
DLSW_TRIPLE=x86_64-linux-gnu
ifeq ($(TARGET), aarch64)
CUDA_TRIPLE=aarch64-linux
CUBLAS_TRIPLE=aarch64-linux-gnu
DLSW_TRIPLE=aarch64-linux-gnu
endif
ifeq ($(TARGET), qnx)
CUDA_TRIPLE=aarch64-qnx
CUBLAS_TRIPLE=aarch64-qnx-gnu
DLSW_TRIPLE=aarch64-unknown-nto-qnx
endif
ifeq ($(TARGET), ppc64le)
CUDA_TRIPLE=ppc64le-linux
CUBLAS_TRIPLE=ppc64le-linux
DLSW_TRIPLE=ppc64le-linux
endif
ifeq ($(TARGET), android64)
DLSW_TRIPLE=aarch64-linux-androideabi
CUDA_TRIPLE=$(DLSW_TRIPLE)
CUBLAS_TRIPLE=$(DLSW_TRIPLE)
endif

ARCH := $(shell uname -p)
UNAME := $(shell whoami)
UID := $(shell id -u `whoami`)
GROUPNAME := $(shell id -gn `whoami`)
GROUPID := $(shell id -g `whoami`)

# Conditional Docker flags
ifndef DOCKER_DETACH
DOCKER_DETACH := 0
endif
ifndef DOCKER_TAG
DOCKER_TAG := $(UNAME)
endif

PROJECT_ROOT := $(shell pwd)
BUILD_DIR    := $(PROJECT_ROOT)/build

# CuDNN and TensorRT Bindings
ifeq ($(ARCH), aarch64)
CUDA_VER   := 10.0
TRT_VER    := 6.0.1.5
UBUNTU_VER := 18.04
CUDNN_VER  := 7.6
CUB_VER    := 1.8.0
else
CUDA_VER   := 10.1
TRT_VER    := 6.0.1.5
UBUNTU_VER := 16.04
CUDNN_VER  := 7.6
CUB_VER    := 1.8.0
endif

# Set the include directory for Loadgen header files
INFERENCE_DIR = $(BUILD_DIR)/inference
LOADGEN_INCLUDE_DIR := $(INFERENCE_DIR)/loadgen
INFERENCE_HASH = 61220457dec221ed1984c62bd9d382698bd71bc6

# Set Environment variables to extracted contents
#export LD_LIBRARY_PATH := /usr/local/cuda-$(CUDA_VER)/lib64:/usr/lib/$(ARCH)-linux-gnu:$(LOADGEN_INCLUDE_DIR)/build/lib.linux-x86_64-2.7:$(LD_LIBRARY_PATH)
#export LIBRARY_PATH := /usr/local/cuda-$(CUDA_VER)/lib64:/usr/lib/$(ARCH)-linux-gnu:$(LOADGEN_INCLUDE_DIR)/build/lib.linux-x86_64-2.7:$(LIBRARY_PATH)
#export PATH := /usr/local/cuda-$(CUDA_VER)/bin:$(PATH)
export CPATH := /usr/local/cuda-$(CUDA_VER)/include:/usr/include/$(ARCH)-linux-gnu:/usr/include/$(ARCH)-linux-gnu/cub:$(CPATH)
#export CUDA_PATH := /usr/local/cuda-$(CUDA_VER)



BUILD_TYPE=Release
export TARGET
export VERBOSE
export LIBDIR
export CUDA_TRIPLE
export CUBLAS_TRIPLE
export DLSW_TRIPLE
samples=sampleMNIST sampleMobilenetV1


.PHONY: all clean help
all: download_model build_plugins
	$(AT)$(foreach sample,$(samples), $(MAKE) -C $(sample) &&) :

clean:
	rm -rf build
	$(AT)$(foreach sample,$(samples), $(MAKE) clean -C $(sample) &&) :

help:
	$(AT)echo "Sample building help menu."
	$(AT)echo "Samples:"
	$(AT)$(foreach sample,$(samples), echo "\t$(sample)" &&) :
	$(AT)echo "\nCommands:"
	$(AT)echo "\tall - build all samples."
	$(AT)echo "\tclean - clean all samples."
	$(AT)echo "\nVariables:"
	$(AT)echo "\tTARGET - Specify the target to build for."
	$(AT)echo "\tVERBOSE - Specify verbose output."
	$(AT)echo "\tCUDA_INSTALL_DIR - Directory where cuda installs to."

.PHONY: prepare
	@mkdir -p ./build

.PHONY: build_plugins
build_plugins: prepare
	mkdir -p build/plugins/NMSOptPlugin
	cd build/plugins/NMSOptPlugin && cmake -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) ../../../plugin/NMSOptPlugin && make -j

.PHONY: download_model
download_model: prepare
	@mkdir -p ./SSDMobileNet
	@if [ ! -f ./SSDMobileNet/ssd_mobilenet_v1_coco_2018_01_28/frozen_inference_graph.pb ]; then \
		echo "Downloading SSDMobileNet model..." \
			&& wget -nv http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz  \
			&& tar -xzvf ./SSDMobileNet/ssd_mobilenet_v1_coco_2018_01_28.tar.gz -C ./SSDMobileNet/  ssd_mobilenet_v1_coco_2018_01_28/frozen_inference_graph.pb \
			&& rm -f ./SSDMobileNet/SSDMobileNet.tar.gz; \
	fi
