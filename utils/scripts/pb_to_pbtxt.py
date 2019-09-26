import os
import sys
import argparse
import tensorflow as tf
   
def main():
    print('input arguments: {}'.format(sys.argv))
    parser = argparse.ArgumentParser(description='Converter pb to uff')
    parser.add_argument('input_model', action="store", help='input tensorflow pb model file')
    parser.add_argument('-v', '--verbose', action="store_true", default=False, help='show debug messages')
    parser.add_argument('-o', '--outfile', action="store", dest='outfile', default='', help='out pbtxt file')
    parser.add_argument('-d', '--outdir', action="store", dest='outdir', default='./', help='out directory')

 
    options = parser.parse_args(sys.argv[1:])
    print(options)

    filename = os.path.splitext(os.path.basename(options.input_model))[0]
    print(filename)

    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(options.input_model, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        
        if options.outfile == '':
            out_filename = "{}.pbtxt".format(filename)
        else:
            out_filename = options.outfile
        tf.train.write_graph(od_graph_def,
                            os.path.join(options.outdir),
                            out_filename,
                            as_text=True)



if __name__ == '__main__':
    main()
