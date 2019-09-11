'''
    python3 test_ball_detection_restore.py
'''

from __future__ import absolute_import


# Importing Matplotlib which is a plotting library for the Python programming
# language and its numerical mathematics extension NumPy

from matplotlib import pyplot as plt

# tensorflow libraries    
import tensorflow as tf
import tensorflow.contrib.tensorrt as trt

# tqdm timing library
import tqdm

# more common python libraries

import numpy as np
import time
import os
import glob
from os.path import join

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0" #selects a specific device


from PIL import Image

# more helper functions for detection tasks 
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

from common import detect_frames

print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
print('\tNative Frozen Model')
path_to_graph = join('ball_models/frozen_model','frozen_inference_graph.pb') 
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

PATH_TO_LABELS = 'ball_models/ball_label_map.pbtxt'
PATH_TO_TEST_IMAGES_DIR = 'ball_models/test_data' #Change the dataset and view the detections
OUT_PATH = 'ball_models/test_result'

detect_frames(detection_graph, PATH_TO_LABELS, PATH_TO_TEST_IMAGES_DIR, OUT_PATH)
detect_frames(detection_graph, PATH_TO_LABELS, PATH_TO_TEST_IMAGES_DIR, OUT_PATH)
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

path_to_graph = join('ball_models/optimized_model_FP16','optimized_graph.pb') 
# Import a graph by reading it as a string, parsing this string then importing it using the tf.import_graph_def command
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
print('\tFP32 Optimized Model')
print('Importing graph...')
detection_graph = tf.Graph()
with detection_graph.as_default():
    optimized_graph_def = tf.GraphDef()
    with tf.gfile.GFile(path_to_graph, 'rb') as fid:
        optimized_graph = fid.read()
        optimized_graph_def.ParseFromString(optimized_graph)
        tf.import_graph_def(optimized_graph_def, name='')
print('Importing graph completed')

PATH_TO_LABELS = 'ball_models/ball_label_map.pbtxt'
PATH_TO_TEST_IMAGES_DIR = 'ball_models/test_data' #Change the dataset and view the detections
OUT_PATH = 'ball_models/test_result_FP32'

detect_frames(detection_graph, PATH_TO_LABELS, PATH_TO_TEST_IMAGES_DIR, OUT_PATH)
detect_frames(detection_graph, PATH_TO_LABELS, PATH_TO_TEST_IMAGES_DIR, OUT_PATH)
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

path_to_graph = join('ball_models/optimized_model_FP16','optimized_graph.pb') 
# Import a graph by reading it as a string, parsing this string then importing it using the tf.import_graph_def command
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
print('\tFP16 Optimized Model')
print('Importing graph...')
detection_graph = tf.Graph()
with detection_graph.as_default():
    optimized_graph_def = tf.GraphDef()
    with tf.gfile.GFile(path_to_graph, 'rb') as fid:
        optimized_graph = fid.read()
        optimized_graph_def.ParseFromString(optimized_graph)
        tf.import_graph_def(optimized_graph_def, name='')
print('Importing graph completed')

PATH_TO_LABELS = 'ball_models/ball_label_map.pbtxt'
PATH_TO_TEST_IMAGES_DIR = 'ball_models/test_data' #Change the dataset and view the detections
OUT_PATH = 'ball_models/test_result_FP16'

detect_frames(detection_graph, PATH_TO_LABELS, PATH_TO_TEST_IMAGES_DIR, OUT_PATH)
detect_frames(detection_graph, PATH_TO_LABELS, PATH_TO_TEST_IMAGES_DIR, OUT_PATH)
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

path_to_graph = join('ball_models/optimized_model_INT8','optimized_graph.pb')
# Import a graph by reading it as a string, parsing this string then importing it using the tf.import_graph_def command
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
print('\tINT8 Optimized Model')
print('Importing graph...')
detection_graph = tf.Graph()
with detection_graph.as_default():
    optimized_graph_def = tf.GraphDef()
    with tf.gfile.GFile(path_to_graph, 'rb') as fid:
        optimized_graph = fid.read()
        optimized_graph_def.ParseFromString(optimized_graph)
        tf.import_graph_def(optimized_graph_def, name='')
print('Importing graph completed')

PATH_TO_LABELS = 'ball_models/ball_label_map.pbtxt'
PATH_TO_TEST_IMAGES_DIR = 'ball_models/test_data' #Change the dataset and view the detections
OUT_PATH = 'ball_models/test_result_INT8'

detect_frames(detection_graph, PATH_TO_LABELS, PATH_TO_TEST_IMAGES_DIR, OUT_PATH)
detect_frames(detection_graph, PATH_TO_LABELS, PATH_TO_TEST_IMAGES_DIR, OUT_PATH)
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
