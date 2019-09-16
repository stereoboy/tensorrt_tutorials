'''
    python3 test_pet_detection.py
'''


from __future__ import absolute_import


# Importing Matplotlib which is a plotting library for the Python programming
# language and its numerical mathematics extension NumPy

from matplotlib import pyplot as plt

# libraries to read json config files

import argparse
import json

# Helper function for downloading models and datasets

from tensorrt.helper import download_model, download_dataset
from tensorrt.helper import MODELS as models


# urllib2 for http downloads
try:
    import urllib2
except ImportError:
    import urllib.request as urllib2
    
    
# tensorflow libraries    
import tensorflow as tf
import tensorflow.contrib.tensorrt as trt

# more common python libraries

from collections import namedtuple
from PIL import Image
import numpy as np
import time
import subprocess
import os
import glob
from os.path import join
from common import detect_frames, optimize_model

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0" #selects a specific device

print("Custom Operations")
print("Custom Relu6 Code Sample")

FROZEN_GRAPH_PATH = 'pet_models/frozen_model/frozen_inference_graph.pb'

config_path = "pet_models/frozen_model/pipeline.config"
checkpoint_path = "pet_models/frozen_model/model.ckpt"
print(config_path, checkpoint_path)

print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
print("\tNative")
path_to_graph = join('pet_models/frozen_model','frozen_inference_graph.pb') 
# Import a graph by reading it as a string, parsing this string then importing it using the tf.import_graph_def command
print('Importing graph...')
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(path_to_graph, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
print('Importing graph completed')

PATH_TO_LABELS = 'pet_models/pet_label_map.pbtxt'
PATH_TO_TEST_IMAGES_DIR = 'pet_models/test_data' #Change the dataset and view the detections
OUT_PATH = 'pet_models/test_result'

if not os.path.exists(OUT_PATH):
    os.makedirs(OUT_PATH)


detect_frames(detection_graph, PATH_TO_LABELS, PATH_TO_TEST_IMAGES_DIR, OUT_PATH)
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

def _read_image(image_path, image_shape):
    #image = Image.open(image_path).convert('RGB')
    image = Image.open(image_path).convert('RGB')
    if image_shape is not None:
        image = image.resize(image_shape[::-1])
    return np.array(image)

print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
print("\tFP32")
# optimize model using source model
frozen_graph_optimized = optimize_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    frozen_graph_path=FROZEN_GRAPH_PATH,
    use_trt=True,
    precision_mode="FP32",
    force_nms_cpu=True,
    replace_relu6=True,
    remove_assert=True,
    override_nms_score_threshold=0.3,
    max_batch_size=1
    )

print('Post-Optimization Run:')

# Import a graph by reading it as a string, parsing this string then importing it using the tf.import_graph_def command
print('Importing graph...')
if not os.path.exists('pet_models/optimized_model_FP32'):
    os.makedirs('pet_models/optimized_model_FP32')

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    tf.import_graph_def(frozen_graph_optimized, name='')
    with tf.gfile.FastGFile('pet_models/optimized_model_FP32/optimized_graph.pb', "w") as f:
        f.write(frozen_graph_optimized.SerializeToString())

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
#    with tf.Session(config=tf_config) as tf_sess:
#        init = tf.global_variables_initializer()
#        tf_sess.run(init)
#        saver = tf.train.Saver()
#        saver.save(tf_sess, 'pet_models/optimized_model_FP16')
#        tf.train.write_graph(tf_sess.graph.as_graph_def(), 'pet_models/optimized_model_FP16', 'optimized_model.pbtxt', as_text=True)
print('Importing graph completed')


PATH_TO_LABELS = 'pet_models/pet_label_map.pbtxt'
PATH_TO_TEST_IMAGES_DIR = 'pet_models/test_data' #Change the dataset and view the detections
OUT_PATH = 'pet_models/test_result_FP32'

if not os.path.exists(OUT_PATH):
    os.makedirs(OUT_PATH)

detect_frames(detection_graph, PATH_TO_LABELS, PATH_TO_TEST_IMAGES_DIR, OUT_PATH)
detect_frames(detection_graph, PATH_TO_LABELS, PATH_TO_TEST_IMAGES_DIR, OUT_PATH)
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")



print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
print("\tFP16")
# optimize model using source model
frozen_graph_optimized = optimize_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    frozen_graph_path=FROZEN_GRAPH_PATH,
    use_trt=True,
    precision_mode="FP16",
    force_nms_cpu=True,
    replace_relu6=True,
    remove_assert=True,
    override_nms_score_threshold=0.3,
    max_batch_size=1
    )

print('Post-Optimization Run:')

# Import a graph by reading it as a string, parsing this string then importing it using the tf.import_graph_def command
print('Importing graph...')
if not os.path.exists('pet_models/optimized_model_FP16'):
    os.makedirs('pet_models/optimized_model_FP16')

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    tf.import_graph_def(frozen_graph_optimized, name='')
    with tf.gfile.FastGFile('pet_models/optimized_model_FP16/optimized_graph.pb', "w") as f:
        f.write(frozen_graph_optimized.SerializeToString())

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
print('Importing graph completed')


PATH_TO_LABELS = 'pet_models/pet_label_map.pbtxt'
PATH_TO_TEST_IMAGES_DIR = 'pet_models/test_data' #Change the dataset and view the detections
OUT_PATH = 'pet_models/test_result_FP16'

if not os.path.exists(OUT_PATH):
    os.makedirs(OUT_PATH)

detect_frames(detection_graph, PATH_TO_LABELS, PATH_TO_TEST_IMAGES_DIR, OUT_PATH)
detect_frames(detection_graph, PATH_TO_LABELS, PATH_TO_TEST_IMAGES_DIR, OUT_PATH)
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")


print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
print("\tINT8")
# optimize model using source model
frozen_graph_optimized = optimize_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    frozen_graph_path=FROZEN_GRAPH_PATH,
    use_trt=True,
    precision_mode="INT8",
    force_nms_cpu=True,
    replace_relu6=True,
    remove_assert=True,
    override_nms_score_threshold=0.3,
    max_batch_size=1,
    calib_images_dir='pet_models/test_data',
    num_calib_images=100,
    )

print('Post-Optimization Run:')

#
# Save optimized graph on a file
#
# references:
#   - https://medium.com/@prasadpal107/saving-freezing-optimizing-for-inference-restoring-of-tensorflow-models-b4146deb21b5
#   - https://docs.nvidia.com/deeplearning/frameworks/tf-trt-user-guide/index.html
#

# Import a graph by reading it as a string, parsing this string then importing it using the tf.import_graph_def command
print('Importing graph...')
if not os.path.exists('pet_models/optimized_model_INT8'):
    os.makedirs('pet_models/optimized_model_INT8')

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    tf.import_graph_def(frozen_graph_optimized, name='')
    with tf.gfile.FastGFile('pet_models/optimized_model_INT8/optimized_graph.pb', "w") as f:
        f.write(frozen_graph_optimized.SerializeToString())
print('Importing graph completed')


PATH_TO_LABELS = 'pet_models/pet_label_map.pbtxt'
PATH_TO_TEST_IMAGES_DIR = 'pet_models/test_data' #Change the dataset and view the detections
OUT_PATH = 'pet_models/test_result_INT8'

if not os.path.exists(OUT_PATH):
    os.makedirs(OUT_PATH)

detect_frames(detection_graph, PATH_TO_LABELS, PATH_TO_TEST_IMAGES_DIR, OUT_PATH)
detect_frames(detection_graph, PATH_TO_LABELS, PATH_TO_TEST_IMAGES_DIR, OUT_PATH)
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

