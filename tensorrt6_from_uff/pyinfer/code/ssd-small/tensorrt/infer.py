#! /usr/bin/env python3
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
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

import ctypes
import os
import sys

#
# https://gist.github.com/xhlulu/f7735970704b97fd0b72203628c1cc77
#
category_map = {
    0: 'nothing',
    1: 'person',
    2: 'bicycle',
    3: 'car',
    4: 'motorcycle',
    5: 'airplane',
    6: 'bus',
    7: 'train',
    8: 'truck',
    9: 'boat',
    10: 'traffic light',
    11: 'fire hydrant',
    13: 'stop sign',
    14: 'parking meter',
    15: 'bench',
    16: 'bird',
    17: 'cat',
    18: 'dog',
    19: 'horse',
    20: 'sheep',
    21: 'cow',
    22: 'elephant',
    23: 'bear',
    24: 'zebra',
    25: 'giraffe',
    27: 'backpack',
    28: 'umbrella',
    31: 'handbag',
    32: 'tie',
    33: 'suitcase',
    34: 'frisbee',
    35: 'skis',
    36: 'snowboard',
    37: 'sports ball',
    38: 'kite',
    39: 'baseball bat',
    40: 'baseball glove',
    41: 'skateboard',
    42: 'surfboard',
    43: 'tennis racket',
    44: 'bottle',
    46: 'wine glass',
    47: 'cup',
    48: 'fork',
    49: 'knife',
    50: 'spoon',
    51: 'bowl',
    52: 'banana',
    53: 'apple',
    54: 'sandwich',
    55: 'orange',
    56: 'broccoli',
    57: 'carrot',
    58: 'hot dog',
    59: 'pizza',
    60: 'donut',
    61: 'cake',
    62: 'chair',
    63: 'couch',
    64: 'potted plant',
    65: 'bed',
    67: 'dining table',
    70: 'toilet',
    72: 'tv',
    73: 'laptop',
    74: 'mouse',
    75: 'remote',
    76: 'keyboard',
    77: 'cell phone',
    78: 'microwave',
    79: 'oven',
    80: 'toaster',
    81: 'sink',
    82: 'refrigerator',
    84: 'book',
    85: 'clock',
    86: 'vase',
    87: 'scissors',
    88: 'teddy bear',
    89: 'hair drier',
    90: 'toothbrush'
}
# The plugin .so file has to be loaded at global scope and before `import torch` to avoid cuda version mismatch.
NMS_OPT_PLUGIN_LIBRARY="../build/plugins/NMSOptPlugin/libnmsoptplugin.so"
if not os.path.isfile(NMS_OPT_PLUGIN_LIBRARY):
    raise IOError("{}\n{}\n".format(
        "Failed to load library ({}).".format(NMS_OPT_PLUGIN_LIBRARY),
        "Please build the NMS Opt plugin."
    ))
ctypes.CDLL(NMS_OPT_PLUGIN_LIBRARY)

sys.path.insert(0, os.getcwd())
#sys.path.append(os.getcwd())

import argparse
import enum
import json
import numpy as np
import pytest
import tensorrt as trt
import time
import cv2
import glob 

from code.common.runner import EngineRunner, get_input_format
from code.common import logging
import code.common.arguments as common_args
#from glob import glob
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# The output detections for each image is [keepTopK, 7]. The 7 elements are:
class PredictionLayout(enum.IntEnum):
    IMAGE_ID = 0
    YMIN = 1
    XMIN = 2
    YMAX = 3
    XMAX = 4
    CONFIDENCE = 5
    LABEL = 6


def preprocess_int8_chw4(batch_idx, img):

    start_time = time.time()
    img_resized = cv2.resize(img, (300, 300), interpolation=cv2.INTER_LINEAR)
    logging.info("Batch {:d} >> cv2.resize(img_rgba, (300, 300)):  {:f}".format(batch_idx, time.time() - start_time))

    img_rgba = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGBA)
    logging.info("Batch {:d} >> cv2.cvtColor(img, cv2.COLOR_BGR2RGBA):  {:f}".format(batch_idx, time.time() - start_time))

    start_time = time.time()
    img_int32 = np.array(img_rgba).astype(np.int32)
    logging.info("Batch {:d} >> np.array(img_rgba).astype(np.int32):  {:f}".format(batch_idx, time.time() - start_time))
    start_time = time.time()
    #img_float = img_float.transpose((2, 0, 1))
    img_int32 = img_int32 - 127
    img_int8  = img_int32.astype(dtype=np.int8, order='C')
    img_int8[:, :, 3] = 0
    img_int8_chw4 = img_int8
    #img_int8_chw4= np.moveaxis(np.pad(img_int8, ((0, 1), (0, 0),(0, 0)), "constant"), -3, -1)
    print("img_int8_chw4.flags['C_CONTIGUOUS'] = {}".format(img_int8_chw4.flags['C_CONTIGUOUS']))
    batch_images = np.expand_dims(img_int8_chw4, axis=0)

    return batch_images

def preprocess_float32_linear(batch_idx, img):

    start_time = time.time()
    img_resized = cv2.resize(img, (300, 300), interpolation=cv2.INTER_LINEAR)
    logging.info("Batch {:d} >> cv2.resize(img_rgba, (300, 300)):  {:f}".format(batch_idx, time.time() - start_time))

    img_rgba = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    logging.info("Batch {:d} >> cv2.cvtColor(img, cv2.COLOR_BGR2RGB):  {:f}".format(batch_idx, time.time() - start_time))

    start_time = time.time()
    img_float32 = np.array(img_rgba).astype(np.float32)
    logging.info("Batch {:d} >> np.array(img_rgba).astype(np.float32):  {:f}".format(batch_idx, time.time() - start_time))
    start_time = time.time()
    img_float32 = img_float32.transpose((2, 0, 1))
    img_float32 = np.ascontiguousarray(img_float32)
    print("img_float32.flags['C_CONTIGUOUS'] = {}".format(img_float32.flags['C_CONTIGUOUS']))
    img_float32 = (2.0 / 255.0) * img_float32 - 1.0
    batch_images = np.expand_dims(img_float32, axis=0)

    return batch_images

def run_SSDMobileNet_accuracy(engine_file, batch_size, num_images, verbose=False, output_file="build/out/SSDMobileNet/dump.json"):
    logging.info("Running SSDMobileNet functionality test for engine [ {:} ] with batch size {:}".format(engine_file, batch_size))

    runner = EngineRunner(engine_file, verbose=verbose)
    input_dtype, input_format = get_input_format(runner.engine)
    if input_dtype == trt.DataType.FLOAT:
        format_string = "fp32"
        preprocess = preprocess_float32_linear
    elif input_dtype == trt.DataType.INT8:
        if input_format == trt.TensorFormat.LINEAR:
            format_string = "int8_linear"
        elif input_format == trt.TensorFormat.CHW4:
            format_string = "int8_chw4"
            preprocess = preprocess_int8_chw4


    logging.info("Engine TensorFormat: {}".format(format_string))

    logging.info("Running validation on {:} images. Please wait...".format(num_images))
    batch_idx = 0
    img_paths = glob.glob(os.path.join("../data/coco", "*.jpg"))
    print (img_paths)
    for batch_idx, img_path in enumerate(img_paths):
        img = cv2.imread(img_path)
        img_height, img_width = img.shape[:2]
        logging.info("Dim = {}x{}".format(img_height, img_width))

        start_time = time.time()
        batch_images = preprocess(batch_idx, img)
        if verbose:
            logging.info("Batch {:d} >> Preprocessing time:  {:f}".format(batch_idx, time.time() - start_time))
        
        start_time = time.time()
        [outputs] = runner([batch_images], batch_size)
        if verbose:
            logging.info("Batch {:d} >> Inference time:  {:f}".format(batch_idx, time.time() - start_time))

        batch_detections = outputs.reshape(batch_size, 100*7+1)[:batch_size]

        for detections in batch_detections:
            keep_count = detections[100*7].view('int32')
            for detection in detections[:keep_count*7].reshape(keep_count,7):
                score = float(detection[PredictionLayout.CONFIDENCE])
                xmin = detection[PredictionLayout.XMIN] * img_width
                ymin = detection[PredictionLayout.YMIN] * img_height
                xmax = (detection[PredictionLayout.XMAX]) * img_width
                ymax = (detection[PredictionLayout.YMAX]) * img_height
                score = float(detection[PredictionLayout.CONFIDENCE])

                if score > 0.2:
                    cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (100, 255, 0), 2)
                    class_id = int(detection[PredictionLayout.LABEL])
                    class_label = category_map[class_id]
                    display_str = "{}:{}%".format(class_label, int(100*score))
                    cv2.putText(img, display_str, (int(xmin), int(ymin + 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 0), 5, cv2.LINE_8)
                    cv2.putText(img, display_str, (int(xmin), int(ymin + 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_8)

        cv2.imshow('display', img)
        key = cv2.waitKey(0)
        if key == ord('c'):
            continue
        if key == 27 or key == ord('q'):
            break



def main():
    args = common_args.parse_args(common_args.ACCURACY_ARGS)
    logging.info("Running accuracy test...")
    run_SSDMobileNet_accuracy(args["engine_file"], args["batch_size"], args["num_images"],
            verbose=args["verbose"])

if __name__ == "__main__":
    main()
