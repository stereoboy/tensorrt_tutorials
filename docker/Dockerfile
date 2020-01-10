# for tensorrt + tensorflow-gpu=1.14.0
#FROM nvcr.io/nvidia/tensorrt:19.02-py3
#FROM tensorflow/tensorflow:1.13.1-gpu-py3-jupyter
#FROM nvcr.io/nvidia/tensorflow:19.06-py3
#FROM nvcr.io/nvidia/tensorflow:19.08-py3
FROM nvcr.io/nvidia/tensorflow:19.10-py3
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update -y
RUN apt-get install -y python3-tk
RUN apt-get remove -y cmake
RUN pip install cmake==3.11.0
RUN pip install matplotlib tqdm Pillow
RUN pip install python-opencv
RUN pip install pycuda==2019.1
RUN pip install pytest==4.6.2
RUN wget https://github.com/NVlabs/cub/archive/1.8.0.zip -O cub-1.8.0.zip \
 && unzip cub-1.8.0.zip \
 && mv cub-1.8.0/cub /usr/include/x86_64-linux-gnu/ \
 && rm -rf cub-1.8.0.zip cub-1.8.0
ENV PYTHONPATH $PYTHONPATH:/work2/tensorflow/models:/work2/tensorflow/models/research:/work2/tensorflow/models/research/slim
