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


# OpenCV library
from PIL import Image

# more helper functions for detection tasks 
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

def detect_frames(path_to_labels,
                  data_folder,
                  output_path):

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # We load the label maps and access category names and their associated indicies
    label_map = label_map_util.load_labelmap(path_to_labels)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=1, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    print('Starting session...')
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            # Define input and output Tensors for detection_graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represents the level of confidence for each of the objects.
            # Score is shown on the resulting image, together with the class label.
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            frames_path = data_folder
            xml_path = join(data_folder, 'xml')
            num_frames = len([name for name in
                              os.listdir(frames_path)
                              if os.path.isfile(join(frames_path, name))])

            number_of_tests = 100
            counter = 1
            timings = []
            total_time = 0
            print('Running Inference:')
            for fdx, file_name in \
                    enumerate(sorted(os.listdir(data_folder))):
                image = Image.open(join(frames_path, file_name))

                image_np = np.array(image)
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image_np, axis=0)
                tic = time.time()
                (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})
                toc = time.time()
                t_diff = toc - tic
                total_time = total_time + t_diff
                # Visualization of the results of a detection.
                vis_util.visualize_boxes_and_labels_on_image_array(
                    image_np,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=7,
                    min_score_thresh=0.5)
                image = Image.fromarray(image_np, 'RGB')
                image.save(join(output_path, file_name))
                prog = 'Completed current frame in: %.6f seconds. %% (Total: %.6f secconds)' % (t_diff, total_time)

                if counter > 10:
                    timings.append(1000*t_diff)
                print('{}\r'.format(prog))
                counter = counter + 1
                if counter > number_of_tests:
                    break
            print("mean = {}, std = {}".format(np.mean(timings), np.std(timings)))

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

detect_frames(PATH_TO_LABELS, PATH_TO_TEST_IMAGES_DIR, OUT_PATH)
detect_frames(PATH_TO_LABELS, PATH_TO_TEST_IMAGES_DIR, OUT_PATH)
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

detect_frames(PATH_TO_LABELS, PATH_TO_TEST_IMAGES_DIR, OUT_PATH)
detect_frames(PATH_TO_LABELS, PATH_TO_TEST_IMAGES_DIR, OUT_PATH)
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

detect_frames(PATH_TO_LABELS, PATH_TO_TEST_IMAGES_DIR, OUT_PATH)
detect_frames(PATH_TO_LABELS, PATH_TO_TEST_IMAGES_DIR, OUT_PATH)
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
