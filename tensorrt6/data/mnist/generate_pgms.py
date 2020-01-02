#!/usr/bin/env python3
from PIL import Image
import numpy as np
import argparse
import os

# Returns a numpy buffer of shape (num_images, 28, 28)
def load_mnist_data(filepath):
    with open(filepath, "rb") as f:
        raw_buf = np.fromstring(f.read(), dtype=np.uint8)
    # Make sure the magic number is what we expect
    assert raw_buf[0:4].view(">i4")[0] == 2051
    num_images = raw_buf[4:8].view(">i4")[0]
    image_h = raw_buf[8:12].view(">i4")[0]
    image_w = raw_buf[12:16].view(">i4")[0]
    # Colors in the dataset are inverted vs. what the samples expect.
    return np.ascontiguousarray(255 - raw_buf[16:].reshape(num_images, image_h, image_w))

# Returns a list of length num_images
def load_mnist_labels(filepath):
    with open(filepath, "rb") as f:
        raw_buf = np.fromstring(f.read(), dtype=np.uint8)
    # Make sure the magic number is what we expect
    assert raw_buf[0:4].view(">i4")[0] == 2049
    num_labels = raw_buf[4:8].view(">i4")[0]
    return list(raw_buf[8:].astype(np.int32).reshape(num_labels))

def main():
    parser = argparse.ArgumentParser(description="Extracts 10 PGM files from the MNIST dataset", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-d", "--dataset", help="Path to the MNIST training, testing or validation dataset, e.g. train-images-idx3-ubyte, t10k-images-idx3-ubyte, etc.", default=os.path.abspath("train-images-idx3-ubyte"))
    parser.add_argument("-l", "--labels", help="Path to the MNIST training, testing or validation labels, e.g. train-labels-idx1-ubyte, t10k-labels-idx1-ubyte, etc.", default=os.path.abspath("train-labels-idx1-ubyte"))
    parser.add_argument("-o", "--output", help="Path to the output directory.", default=os.getcwd())

    args, _ = parser.parse_known_args()

    data = load_mnist_data(args.dataset)
    labels = load_mnist_labels(args.labels)
    output_dir = args.output

    # Find one image for each digit.
    for i in range(10):
        index = labels.index(i)
        image = Image.fromarray(data[index], mode="L")
        path = os.path.join(output_dir, "{:}.pgm".format(i))
        image.save(path)

if __name__ == '__main__':
    main()
