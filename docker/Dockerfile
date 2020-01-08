# for tensorrt + tensorflow-gpu=1.14.0
#FROM nvcr.io/nvidia/tensorrt:19.02-py3
#FROM tensorflow/tensorflow:1.13.1-gpu-py3-jupyter
#FROM nvcr.io/nvidia/tensorflow:19.08-py3
FROM nvcr.io/nvidia/tensorflow:19.10-py3
FROM nvcr.io/nvidia/tensorflow:19.06-py3
RUN pip install matplotlib tqdm Pillow
ENV PYTHONPATH $PYTHONPATH:/work2/tensorflow/models:/work2/tensorflow/models/research:/work2/tensorflow/models/research/slim
