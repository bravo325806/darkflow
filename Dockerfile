FROM tensorflow/tensorflow:latest-gpu-py3

WORKDIR /

RUN apt-get update && apt-get install -y git libsm6 libxext6 libxrender-dev
RUN git clone https://github.com/bravo325806/darkflow

WORKDIR /darkflow

RUN pip install Cython opencv-python
RUN pip install -e .

VOLUME /dataset
VOLUME /darkflow/built_graph
VOLUME /darkflow/parameter

CMD nohup python3 train
