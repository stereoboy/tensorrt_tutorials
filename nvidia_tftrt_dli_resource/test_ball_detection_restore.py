'''
    python3 test_ball_detection_restore.py
'''

#from __future__ import absolute_import


# Importing Matplotlib which is a plotting library for the Python programming
# language and its numerical mathematics extension NumPy

#from matplotlib import pyplot as plt
import matplotlib
import matplotlib.pyplot as plt

# tensorflow libraries    
import tensorflow as tf
import tensorflow.contrib.tensorrt as trt

# tqdm timing library
import tqdm

# more common python libraries

from datetime import datetime
import numpy as np
import time
import os
import glob
from os.path import join

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0" #selects a specific device

from PIL import Image

from common import detect_frames

names = ['Native', 'FP32', 'FP16', 'INT8']
label_list = []
mean_list = []
std_list = []
color_list = []

def truncate(n):
    return int(n * 10000) / 10000

def update_values(label, mean, std, color):
    label_list.append(label)
    mean_list.append(truncate(mean))
    std_list.append(truncate(std))
    color_list.append(color)

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

mean, std = detect_frames(detection_graph, PATH_TO_LABELS, PATH_TO_TEST_IMAGES_DIR, OUT_PATH)
update_values('Natv#0', mean, std, 'yellow')
mean, std = detect_frames(detection_graph, PATH_TO_LABELS, PATH_TO_TEST_IMAGES_DIR, OUT_PATH)
update_values('Natv#1', mean, std, 'yellow')
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

path_to_graph = join('ball_models/optimized_model_FP32','optimized_graph.pb') 
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

mean, std = detect_frames(detection_graph, PATH_TO_LABELS, PATH_TO_TEST_IMAGES_DIR, OUT_PATH)
update_values('FP32#0', mean, std, 'green')
mean, std = detect_frames(detection_graph, PATH_TO_LABELS, PATH_TO_TEST_IMAGES_DIR, OUT_PATH)
update_values('FP32#1', mean, std, 'green')
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

mean, std =  detect_frames(detection_graph, PATH_TO_LABELS, PATH_TO_TEST_IMAGES_DIR, OUT_PATH)
update_values('FP16#0', mean, std, 'blue')
mean, std =  detect_frames(detection_graph, PATH_TO_LABELS, PATH_TO_TEST_IMAGES_DIR, OUT_PATH)
update_values('FP16#1', mean, std, 'blue')
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

mean, std =  detect_frames(detection_graph, PATH_TO_LABELS, PATH_TO_TEST_IMAGES_DIR, OUT_PATH)
update_values('INT8#0', mean, std, 'red')
mean, std =  detect_frames(detection_graph, PATH_TO_LABELS, PATH_TO_TEST_IMAGES_DIR, OUT_PATH)
update_values('INT8#1', mean, std, 'red')
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

print("\tDraw Graph")
print_str = "\n"
print_str += "|  Name  |  Mean (ms) |  Std (ms)  |\n"
print_str += "|  ----  |  --------- |  --------  |\n"
for label, mean, std in zip(label_list, mean_list, std_list):
    print_str += "| %s | %.4f     | %.4f     |\n"%(label, mean, std)
print_str += "\n"
print(print_str)

timestamp = datetime.now().strftime("ball_%Y_%m%d_%H%M%S")
with open('result_{}.log'.format(timestamp), 'w+') as f:
    f.write(print_str)

matplotlib.use("TkAgg")
x_pos = np.arange(len(label_list))

# Build the plot
fig, ax = plt.subplots()
rect0 = ax.bar(x_pos, mean_list, width=0.25, color=color_list, edgecolor='black', yerr=std_list, align='center', alpha=0.9, ecolor='gray', capsize=10)

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width()/2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rect0)

ax.set_ylabel('Performance on single image (ms)')
ax.set_xticks(x_pos)
ax.set_xticklabels(label_list)
ax.set_title('')
ax.yaxis.grid(True)

# Save the figure and show
plt.tight_layout()
plt.savefig('plot_{}.png'.format(timestamp))
plt.show()

