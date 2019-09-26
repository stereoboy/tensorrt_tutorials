'''
    python3 test_ball_detection_restore.py
'''


import sys
import argparse
import uff
import tensorrt as trt
import tensorflow as tf
import graphsurgeon as gs
import os

import pycuda.driver as cuda
import ctypes

# Importing Matplotlib which is a plotting library for the Python programming
# language and its numerical mathematics extension NumPy

from matplotlib import pyplot as plt

# tensorflow libraries    
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"]="0" #selects a specific device



INPUT_NAME = 'image_tensor'
BOXES_NAME = 'detection_boxes'
CLASSES_NAME = 'detection_classes'
SCORES_NAME = 'detection_scores'
MASKS_NAME = 'detection_masks'
NUM_DETECTIONS_NAME = 'num_detections'

def ssd_unsupported_nodes_to_plugin_nodes(ssd_graph):
    """Makes ssd_graph TensorRT comparible using graphsurgeon.

    This function takes ssd_graph, which contains graphsurgeon
    DynamicGraph data structure. This structure describes frozen Tensorflow
    graph, that can be modified using graphsurgeon (by deleting, adding,
    replacing certain nodes). The graph is modified by removing
    Tensorflow operations that are not supported by TensorRT's UffParser
    and replacing them with custom layer plugin nodes.

    Note: This specific implementation works only for
    ssd_inception_v2_coco_2017_11_17 network.

    Args:
        ssd_graph (gs.DynamicGraph): graph to convert
    Returns:
        gs.DynamicGraph: UffParser compatible SSD graph
    """
    # Create TRT plugin nodes to replace unsupported ops in Tensorflow graph
    channels = 3
    height = 300
    width = 300

#    Cast = gs.create_plugin_node(name="Cast",
#        op="Cast_TRT",

    Input = gs.create_plugin_node(name="Input",
        op="Placeholder",
        dtype=tf.float32,
        shape=[1, channels, height, width])

    PriorBox = gs.create_plugin_node(name="GridAnchor", op="GridAnchor_TRT",
        minSize=0.1,
        maxSize=0.95,
#        minSize=0.1,
#        maxSize=0.25,
        aspectRatios=[1.0, 2.0, 0.5, 3.0, 0.33],
#        aspectRatios=[1.0],
        variance=[0.1,0.1,0.2,0.2],
        featureMapShapes=[19, 10, 5, 3, 2, 1],
#        featureMapShapes=[19, 10],
        numLayers=6
#        numLayers=2
    )
    NMS = gs.create_plugin_node(
        name="NMS",
        op="NMS_TRT",
        shareLocation=1,
        varianceEncodedInTarget=0,
        backgroundLabelId=0,
        confidenceThreshold=1e-8,
        nmsThreshold=0.6,
        topK=100,
        keepTopK=100,
        #numClasses=91,
        numClasses=2,
        inputOrder=[0, 2, 1],
        confSigmoid=1,
        isNormalized=1
    )
    concat_priorbox = gs.create_node(
        "concat_priorbox",
        op="ConcatV2",
        dtype=tf.float32,
        axis=2
    )
    concat_box_loc = gs.create_plugin_node(
        "concat_box_loc",
        op="FlattenConcat_TRT",
        dtype=tf.float32,
    )
    concat_box_conf = gs.create_plugin_node(
        "concat_box_conf",
        op="FlattenConcat_TRT",
        dtype=tf.float32,
    )

    # Create a mapping of namespace names -> plugin nodes.
    namespace_plugin_map = {
        "MultipleGridAnchorGenerator": PriorBox,
        "Postprocessor": NMS,
        "Preprocessor": Input,
        "ToFloat": Input,
        "Cast": Input,
        "image_tensor": Input,
        #"MultipleGridAnchorGenerator/Concatenate": concat_priorbox,
        #"MultipleGridAnchorGenerator/Identity": concat_priorbox,
        #"concat": concat_box_loc,
        #"concat_1": concat_box_conf
    }

    # Create a new graph by collapsing namespaces
    ssd_graph.collapse_namespaces(namespace_plugin_map)
    # Remove the outputs, so we just have a single output node (NMS).
    # If remove_exclusive_dependencies is True, the whole graph will be removed!
    ssd_graph.remove(ssd_graph.graph_outputs, remove_exclusive_dependencies=False)
    return ssd_graph


def main():
    print('input arguments: {}'.format(sys.argv))
    parser = argparse.ArgumentParser(description='Converter pb to uff')
    parser.add_argument('input_model', action="store", help='input tensorflow pb model file')
    parser.add_argument('-v', '--verbose', action="store_true", default=False, help='show debug messages')
    parser.add_argument('-o', '--outfile', action="store", dest='outfile', default='./default.uff', help='out uff file')
    options = parser.parse_args(sys.argv[1:])
    print(options)

    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(options.input_model, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        found = False
        for node in od_graph_def.node:
            if node.name == "NMS":
                print(node)
                found = True
        if found:
            print("FOUND")
        else:
            print("Not FOUND")
    print("dynamic_graph = gs.DynamicGraph(options.input_model)")
    dynamic_graph = gs.DynamicGraph(options.input_model)
    found = False
#    for node in dynamic_graph.as_graph_def().node:
#        print(node)
    if found:
        print("FOUND")
    else:
        print("Not FOUND")

    dynamic_graph = ssd_unsupported_nodes_to_plugin_nodes(dynamic_graph)

    #output_names = [BOXES_NAME, CLASSES_NAME, SCORES_NAME, NUM_DETECTIONS_NAME]
    output_names = ["NMS"]
    uff.from_tensorflow(
            dynamic_graph.as_graph_def(),
            output_filename=options.outfile,
            output_nodes=output_names
            )

    ctypes.CDLL(os.path.join("./plugin/build", "libflattenconcat.so"))
#    print(ctypes.CDLL(os.path.join("./plugin/build", "libgridanchor.so")))
#    ctypes.CDLL(os.path.join("./plugin/build", "libcast.so"))
    print("##############################################################")
    print("\tbuilder, parser, engine")
    print("##############################################################")
    trt_logger = trt.Logger(trt.Logger.INFO)
    
    trt.init_libnvinfer_plugins(trt_logger, '')
    trt_runtime = trt.Runtime(trt_logger)

    builder = trt.Builder(trt_logger)
    builder.max_workspace_size = 1 << 30
    builder.fp16_mode = True
    builder.max_batch_size = 1
    
    network = builder.create_network()
    parser = trt.UffParser()

    parser.register_input('Input', (3, 300, 300))
    parser.register_output('MarkOutput_0')
    #parser.register_output('NMS')
    print("##############################################################")
    parser.parse(options.outfile, network)

#    trt_engine = builder.build_cuda_engine(network)
#
#    if trt_engine == None:
#        print("##############################################################")
#        print("[ERROR] failed to create Engine")
#        print("##############################################################")
#        sys.exit()
#
#
#    print("##############################################################")
#    print("\tengine save")
#    print("##############################################################")
#    buf = trt_engine.serialize()
#    with open('engine.buf', 'wb') as f:
#        f.write(buf)
#    
#    print("##############################################################")
#    print("\tengine load")
#    print("##############################################################")
#    with open('engine.buf', 'rb') as f:
#        engine_data = f.read()
#    trt_engine = trt_runtime.deserialize_cuda_engine(engine_data)
#    
#    stream = cuda.Stream()
#
#    for binding in engine:
#        print(binding)


#    od_graph_def = tf.GraphDef()
#    with tf.gfile.GFile(options.input_model, 'rb') as fid:
#        serialized_graph = fid.read()
#        od_graph_def.ParseFromString(serialized_graph)
#        #tf.import_graph_def(od_graph_def, name='')
#
#    #G_LOGGER = trt.infer.ConsoleLogger(trt.infer.LogSeverity.INFO)
#    output_names = [BOXES_NAME, CLASSES_NAME, SCORES_NAME, NUM_DETECTIONS_NAME]
#    input_name = INPUT_NAME
#    uff_model = uff.from_tensorflow(graphdef=od_graph_def,
#                                    output_filename=options.outfile,
#                                    output_nodes=output_names
#                                    )
#

#    parser = trt.UffParser.create_uff_parser()
#    parser.register_input(input_name, (input_channels , input_height, input_width), 0)
#    parser.register_output(output_names[0])
#
#    engine = trt.utils.uff_to_trt_engine(G_LOGGER, uff_model, parser, 1, 1 << 26)
#    trt.utils.write_engine_to_file(output_filename, engine.serialize())
#    engine.destroy()


if __name__ == '__main__':
    main()
