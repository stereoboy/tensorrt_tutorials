'''
    python3 test_ball_detection_restore_v2.py
'''

#from __future__ import absolute_import


# Importing Matplotlib which is a plotting library for the Python programming
# language and its numerical mathematics extension NumPy

#from matplotlib import pyplot as plt
import matplotlib
import matplotlib.pyplot as plt

# tensorflow libraries    
import tensorflow as tf

# tqdm timing library
import tqdm

# more common python libraries

from datetime import datetime
import numpy as np
import time
import os
import glob
from os.path import join
from multiprocessing import Process, Queue
import logging
import sys
import argparse

from common import detect_frames

os.environ["CUDA_VISIBLE_DEVICES"]="0" #selects a specific device
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

class Statics():
    def __init__(self):
        self.names = ['Native', 'FP32', 'FP16', 'INT8']
        self.label_list = []
        self.mean_list = []
        self.std_list = []
        self.color_list = []

    def truncate(self, n):
        return int(n * 10000) / 10000

    def update_values(self, label, mean, std, color):
        self.label_list.append(label)
        self.mean_list.append(self.truncate(mean))
        self.std_list.append(self.truncate(std))
        self.color_list.append(color)

    def __call__(self):

        print("\tDraw Graph")
        print_str = "\n"
        print_str += "|  Name  |  Mean (ms) |  Std (ms)  |\n"
        print_str += "|  ----  |  --------- |  --------  |\n"
        for label, mean, std in zip(self.label_list, self.mean_list, self.std_list):
            print_str += "| %s | %.4f     | %.4f     |\n"%(label, mean, std)
        print_str += "\n"
        print(print_str)

        timestamp = datetime.now().strftime("ball_%Y_%m%d_%H%M%S")
        with open('result_{}.log'.format(timestamp), 'w+') as f:
            f.write(print_str)

        #matplotlib.use("TkAgg")
        x_pos = np.arange(len(self.label_list))

        # Build the plot
        fig, ax = plt.subplots()
        rect0 = ax.bar(x_pos, self.mean_list, width=0.25, color=self.color_list, edgecolor='black', yerr=self.std_list, align='center', alpha=0.9, ecolor='gray', capsize=10)

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
        ax.set_xticklabels(self.label_list)
        ax.set_title('')
        ax.yaxis.grid(True)

        # Save the figure and show
        plt.tight_layout()
        plt.savefig('plot_{}.png'.format(timestamp))
        plt.show()

def test_native(queue):
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    print('\tNative Frozen Model (pid={})'.format(os.getpid()))
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
    queue.put((mean, std))
    mean, std = detect_frames(detection_graph, PATH_TO_LABELS, PATH_TO_TEST_IMAGES_DIR, OUT_PATH)
    queue.put((mean, std))
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

def test_model(name, queue):
    path_to_graph = join('ball_models/optimized_model_{}'.format(name),'optimized_graph.pb')
    # Import a graph by reading it as a string, parsing this string then importing it using the tf.import_graph_def command
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    print('\t {} Optimized Model (pid={})'.format(name, os.getpid()))
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
    OUT_PATH = 'ball_models/test_result_{}'.format(name)

    mean, std =  detect_frames(detection_graph, PATH_TO_LABELS, PATH_TO_TEST_IMAGES_DIR, OUT_PATH)
    queue.put((mean, std))
    mean, std =  detect_frames(detection_graph, PATH_TO_LABELS, PATH_TO_TEST_IMAGES_DIR, OUT_PATH)
    queue.put((mean, std))
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

def main():
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    print('\tMain (pid={})'.format(os.getpid()))
    parser = argparse.ArgumentParser(description='tf-trt performance test')
    parser.add_argument('--all', action="store_true", default=True, help='test all models')
    parser.add_argument('--native', action="store_true", default=False, help='test native (non-optimized)')
    parser.add_argument('--fp32', action="store_true", default=False, help='test fp32-optimized model')
    parser.add_argument('--fp16', action="store_true", default=False, help='test fp16-optimized model')
    parser.add_argument('--int8', action="store_true", default=False, help='test int8-optimized model')
    options=parser.parse_args(sys.argv[1:])
    logging.info(options)

    statics = Statics()

    do_all = True if not options.native and not options.fp32 and not options.fp16 and not options.int8 else False

    # Test Native
    if options.native or do_all:
        queue = Queue()
        p = Process(target=test_native, args=(queue,))
        p.start()
        mean, std = queue.get()
        statics.update_values('Natv#0', mean, std, 'yellow')
        mean, std = queue.get()
        statics.update_values('Natv#1', mean, std, 'yellow')
        p.join()

    # Test FP32
    if options.fp32 or do_all:
        queue = Queue()
        p = Process(target=test_model, args=('FP32', queue,))
        p.start()
        mean, std = queue.get()
        statics.update_values('FP32#0', mean, std, 'green')
        mean, std = queue.get()
        statics.update_values('FP32#1', mean, std, 'green')
        p.join()

    # Test FP16
    if options.fp16 or do_all:
        queue = Queue()
        p = Process(target=test_model, args=('FP16', queue,))
        p.start()
        mean, std = queue.get()
        statics.update_values('FP16#0', mean, std, 'blue')
        mean, std = queue.get()
        statics.update_values('FP16#1', mean, std, 'blue')
        p.join()

    # Test INT8
    if options.int8 or do_all:
        queue = Queue()
        p = Process(target=test_model, args=('INT8', queue,))
        p.start()
        mean, std = queue.get()
        statics.update_values('INT8#0', mean, std, 'red')
        mean, std = queue.get()
        statics.update_values('INT8#1', mean, std, 'red')
        p.join()

    statics()

if __name__ == '__main__':
    main()
