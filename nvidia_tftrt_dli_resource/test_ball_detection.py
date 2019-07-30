'''
    python3 test_ball_detection.py
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

# tqdm timing library
import tqdm
import pdb

# more common python libraries

from collections import namedtuple
from PIL import Image
import numpy as np
import time
import subprocess
import os
import glob
from os.path import join

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0" #selects a specific device

# OpenCV library
import cv2

# more helper functions for detection tasks 

from tensorrt.graph_utils import force_nms_cpu as f_force_nms_cpu
from tensorrt.graph_utils import replace_relu6 as f_replace_relu6
from tensorrt.graph_utils import remove_assert as f_remove_assert

from google.protobuf import text_format
from object_detection.protos import pipeline_pb2, image_resizer_pb2
from object_detection import exporter
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


print("Custom Operations")
print("Custom Relu6 Code Sample")

config_path = join('tensorrt', 'model_config.json')

with open(config_path, 'r') as f:
    test_config = json.load(f)
    print(json.dumps(test_config, sort_keys=True, indent=4))



Model = namedtuple('Model', ['name', 'url', 'extract_dir'])

INPUT_NAME = 'image_tensor'
BOXES_NAME = 'detection_boxes'
CLASSES_NAME = 'detection_classes'
SCORES_NAME = 'detection_scores'
MASKS_NAME = 'detection_masks'
NUM_DETECTIONS_NAME = 'num_detections'
FROZEN_GRAPH_NAME = 'frozen_inference_graph.pb'
PIPELINE_CONFIG_NAME = 'pipeline.config'
CHECKPOINT_PREFIX = 'model.ckpt'

#config_path, checkpoint_path = download_model(**test_config['source_model'])

config_path = "ball_models/frozen_model/pipeline.config"
checkpoint_path = "ball_models/frozen_model/model.ckpt"
print(config_path, checkpoint_path)

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

            reference_image = os.listdir(data_folder)[0]
            image = cv2.imread(join(data_folder, reference_image))
            height, width, channels = image.shape
            number_of_tests = 10
            counter = 1
            total_time = 0
            print('Running Inference:')
            for fdx, file_name in \
                    enumerate(sorted(os.listdir(data_folder))):
                image = cv2.imread(join(frames_path, file_name))
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
                    image,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=7,
                    min_score_thresh=0.5)
                cv2.imwrite(join(output_path, file_name), image)
                plt.figure(figsize=(6,3))
                plt.imshow(image.astype(np.uint8))
                plt.axis('off')
#                plt.show()
                prog = 'Completed current frame in: %.3f seconds. %% (Total: %.3f secconds)' % (t_diff, total_time)
                print('{}\r'.format(prog))
                counter = counter + 1
                if counter > number_of_tests:
                    break


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

if not os.path.exists(OUT_PATH):
    os.makedirs(OUT_PATH)


detect_frames(PATH_TO_LABELS, PATH_TO_TEST_IMAGES_DIR, OUT_PATH)

print("Optimization")
def _read_image(image_path, image_shape):
    #image = Image.open(image_path).convert('RGB')
    image = Image.open(image_path).convert('RGB')
    if image_shape is not None:
        image = image.resize(image_shape[::-1])
    return np.array(image)

def optimize_model(config_path,
                   checkpoint_path,
                   use_trt=True,
                   force_nms_cpu=True,
                   replace_relu6=True,
                   remove_assert=True,
                   override_nms_score_threshold=None,
                   override_resizer_shape=None,
                   max_batch_size=1,
                   precision_mode='FP32',
                   minimum_segment_size=50,
                   max_workspace_size_bytes=1 << 25,
                   calib_images_dir=None,
                   num_calib_images=None,
                   calib_image_shape=None,
                   tmp_dir='.optimize_model_tmp_dir',
                   remove_tmp_dir=True,
                   output_path=None):
    """Optimizes an object detection model using TensorRT

    Optimizes an object detection model using TensorRT.  This method also
    performs pre-tensorrt optimizations specific to the TensorFlow object
    detection API models.  Please see the list of arguments for other
    optimization parameters.

    Args
    ----
        config_path: A string representing the path of the object detection
            pipeline config file.
        checkpoint_path: A string representing the path of the object
            detection model checkpoint.
        use_trt: A boolean representing whether to optimize with TensorRT. If
            False, regular TensorFlow will be used but other optimizations
            (like NMS device placement) will still be applied.
        force_nms_cpu: A boolean indicating whether to place NMS operations on
            the CPU.
        replace_relu6: A boolean indicating whether to replace relu6(x)
            operations with relu(x) - relu(x-6).
        remove_assert: A boolean indicating whether to remove Assert
            operations from the graph.
        override_nms_score_threshold: An optional float representing
            a NMS score threshold to override that specified in the object
            detection configuration file.
        override_resizer_shape: An optional list/tuple of integers
            representing a fixed shape to override the default image resizer
            specified in the object detection configuration file.
        max_batch_size: An integer representing the max batch size to use for
            TensorRT optimization.
        precision_mode: A string representing the precision mode to use for
            TensorRT optimization.  Must be one of 'FP32', 'FP16', or 'INT8'.
        minimum_segment_size: An integer representing the minimum segment size
            to use for TensorRT graph segmentation.
        max_workspace_size_bytes: An integer representing the max workspace
            size for TensorRT optimization.
        calib_images_dir: A string representing a directory containing images to
            use for int8 calibration.
        num_calib_images: An integer representing the number of calibration
            images to use.  If None, will use all images in directory.
        calib_image_shape: A tuple of integers representing the height,
            width that images will be resized to for calibration.
        tmp_dir: A string representing a directory for temporary files.  This
            directory will be created and removed by this function and should
            not already exist.  If the directory exists, an error will be
            thrown.
        remove_tmp_dir: A boolean indicating whether we should remove the
            tmp_dir or throw error.
        output_path: An optional string representing the path to save the
            optimized GraphDef to.

    Returns
    -------
        A GraphDef representing the optimized model.
    """
    if os.path.exists(tmp_dir):
        if not remove_tmp_dir:
            raise RuntimeError(
                'Cannot create temporary directory, path exists: %s' % tmp_dir)
        subprocess.call(['rm', '-rf', tmp_dir])

    # load config from file
    config = pipeline_pb2.TrainEvalPipelineConfig()
    with open(config_path, 'r') as f:
        text_format.Merge(f.read(), config, allow_unknown_extension=True)

    # override some config parameters
    if config.model.HasField('ssd'):
        config.model.ssd.feature_extractor.override_base_feature_extractor_hyperparams = True
        if override_nms_score_threshold is not None:
            config.model.ssd.post_processing.batch_non_max_suppression.score_threshold = override_nms_score_threshold
        if override_resizer_shape is not None:
            config.model.ssd.image_resizer.fixed_shape_resizer.height = override_resizer_shape[
                0]
            config.model.ssd.image_resizer.fixed_shape_resizer.width = override_resizer_shape[
                1]
    elif config.model.HasField('faster_rcnn'):
        if override_nms_score_threshold is not None:
            config.model.faster_rcnn.second_stage_post_processing.score_threshold = override_nms_score_threshold
        if override_resizer_shape is not None:
            config.model.faster_rcnn.image_resizer.fixed_shape_resizer.height = override_resizer_shape[
                0]
            config.model.faster_rcnn.image_resizer.fixed_shape_resizer.width = override_resizer_shape[
                1]
    print("config.model.ssd.image_resizer.fixed_shape_resizer")
    print(config.model.ssd.image_resizer.fixed_shape_resizer)

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True

    # export inference graph to file (initial), this will create tmp_dir
#    with tf.Session(config=tf_config):
#        with tf.Graph().as_default():
#            exporter.export_inference_graph(
#                INPUT_NAME,
#                config,
#                checkpoint_path,
#                tmp_dir,
#                input_shape=[max_batch_size, None, None, 3])
#
#    # read frozen graph from file
#    frozen_graph_path = os.path.join(tmp_dir, FROZEN_GRAPH_NAME)
    frozen_graph_path = os.path.join('ball_models/frozen_model', FROZEN_GRAPH_NAME)
    frozen_graph = tf.GraphDef()
    with open(frozen_graph_path, 'rb') as f:
        frozen_graph.ParseFromString(f.read())

    # apply graph modifications
    if force_nms_cpu:
        frozen_graph = f_force_nms_cpu(frozen_graph)
    if replace_relu6:
        frozen_graph = f_replace_relu6(frozen_graph)
    if remove_assert:
        frozen_graph = f_remove_assert(frozen_graph)

    # get input names
    output_names = [BOXES_NAME, CLASSES_NAME, SCORES_NAME, NUM_DETECTIONS_NAME]

    # optionally perform TensorRT optimization
    if use_trt:
        with tf.Graph().as_default() as tf_graph:
            with tf.Session(config=tf_config) as tf_sess:
                frozen_graph = trt.create_inference_graph(
                    input_graph_def=frozen_graph,
                    outputs=output_names,
                    max_batch_size=max_batch_size,
                    max_workspace_size_bytes=max_workspace_size_bytes,
                    precision_mode=precision_mode,
                    minimum_segment_size=minimum_segment_size)

                # perform calibration for int8 precision
                if precision_mode == 'INT8':

                    if calib_images_dir is None:
                        raise ValueError('calib_images_dir must be provided for int8 optimization.')

                    tf.import_graph_def(frozen_graph, name='')
                    tf_input = tf_graph.get_tensor_by_name(INPUT_NAME + ':0')
                    tf_boxes = tf_graph.get_tensor_by_name(BOXES_NAME + ':0')
                    tf_classes = tf_graph.get_tensor_by_name(CLASSES_NAME + ':0')
                    tf_scores = tf_graph.get_tensor_by_name(SCORES_NAME + ':0')
                    tf_num_detections = tf_graph.get_tensor_by_name(
                        NUM_DETECTIONS_NAME + ':0')

                    image_paths = glob.glob(os.path.join(calib_images_dir, '*.jpg'))
                    image_paths = image_paths[0:num_calib_images]

                    for image_idx in tqdm.tqdm(range(0, len(image_paths), max_batch_size)):

                        # read batch of images
                        batch_images = []
                        for image_path in image_paths[image_idx:image_idx+max_batch_size]:
                            #image = _read_image(image_path, calib_image_shape)
                            image = cv2.imread(image_path)
                            batch_images.append(image)

                        # execute batch of images
                        boxes, classes, scores, num_detections = tf_sess.run(
                            [tf_boxes, tf_classes, tf_scores, tf_num_detections],
                            feed_dict={tf_input: batch_images})

                    #pdb.set_trace()
                    frozen_graph = trt.calib_graph_to_infer_graph(frozen_graph)

    # re-enable variable batch size, this was forced to max
    # batch size during export to enable TensorRT optimization
    for node in frozen_graph.node:
        if INPUT_NAME == node.name:
            node.attr['shape'].shape.dim[0].size = -1

    # write optimized model to disk
    if output_path is not None:
        with open(output_path, 'wb') as f:
            f.write(frozen_graph.SerializeToString())

    # remove temporary directory
    subprocess.call(['rm', '-rf', tmp_dir])

    return frozen_graph

print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
print("\tINT8")
# optimize model using source model
frozen_graph_optimized = optimize_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    use_trt=True,
    precision_mode="INT8",
    force_nms_cpu=True,
    replace_relu6=True,
    remove_assert=True,
    override_nms_score_threshold=0.3,
    max_batch_size=1,
    calib_images_dir='ball_models/test_data',
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
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    tf.import_graph_def(frozen_graph_optimized, name='')
    with tf.gfile.FastGFile('ball_models/optimized_model_INT8/optimized_graph.pb', "w") as f:
        f.write(frozen_graph_optimized.SerializeToString())
print('Importing graph completed')


PATH_TO_LABELS = 'ball_models/ball_label_map.pbtxt'
PATH_TO_TEST_IMAGES_DIR = 'ball_models/test_data' #Change the dataset and view the detections
OUT_PATH = 'ball_models/test_result_INT8'

if not os.path.exists(OUT_PATH):
    os.makedirs(OUT_PATH)

detect_frames(PATH_TO_LABELS, PATH_TO_TEST_IMAGES_DIR, OUT_PATH)
detect_frames(PATH_TO_LABELS, PATH_TO_TEST_IMAGES_DIR, OUT_PATH)
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
print("\tFP32")
# optimize model using source model
frozen_graph_optimized = optimize_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
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
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    tf.import_graph_def(frozen_graph_optimized, name='')
    with tf.gfile.FastGFile('ball_models/optimized_model_FP32/optimized_graph.pb', "w") as f:
        f.write(frozen_graph_optimized.SerializeToString())

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
#    with tf.Session(config=tf_config) as tf_sess:
#        init = tf.global_variables_initializer()
#        tf_sess.run(init)
#        saver = tf.train.Saver()
#        saver.save(tf_sess, 'ball_models/optimized_model_FP16')
#        tf.train.write_graph(tf_sess.graph.as_graph_def(), 'ball_models/optimized_model_FP16', 'optimized_model.pbtxt', as_text=True)
print('Importing graph completed')


PATH_TO_LABELS = 'ball_models/ball_label_map.pbtxt'
PATH_TO_TEST_IMAGES_DIR = 'ball_models/test_data' #Change the dataset and view the detections
OUT_PATH = 'ball_models/test_result_FP32'

if not os.path.exists(OUT_PATH):
    os.makedirs(OUT_PATH)

detect_frames(PATH_TO_LABELS, PATH_TO_TEST_IMAGES_DIR, OUT_PATH)
detect_frames(PATH_TO_LABELS, PATH_TO_TEST_IMAGES_DIR, OUT_PATH)
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")



print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
print("\tFP16")
# optimize model using source model
frozen_graph_optimized = optimize_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
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
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    tf.import_graph_def(frozen_graph_optimized, name='')
    with tf.gfile.FastGFile('ball_models/optimized_model_FP16/optimized_graph.pb', "w") as f:
        f.write(frozen_graph_optimized.SerializeToString())

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
print('Importing graph completed')


PATH_TO_LABELS = 'ball_models/ball_label_map.pbtxt'
PATH_TO_TEST_IMAGES_DIR = 'ball_models/test_data' #Change the dataset and view the detections
OUT_PATH = 'ball_models/test_result_FP16'

if not os.path.exists(OUT_PATH):
    os.makedirs(OUT_PATH)

detect_frames(PATH_TO_LABELS, PATH_TO_TEST_IMAGES_DIR, OUT_PATH)
detect_frames(PATH_TO_LABELS, PATH_TO_TEST_IMAGES_DIR, OUT_PATH)
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

