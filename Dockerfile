FROM tensorflow/tensorflow:1.14.0-gpu-py3

WORKDIR /

RUN apt-get update && apt-get install -y git libsm6 libxext6 libxrender-dev
RUN git clone https://github.com/bravo325806/darkflow

WORKDIR /darkflow

RUN pip install Cython opencv-python bs4
RUN pip install -e .

VOLUME /dataset
VOLUME /darkflow/built_graph
VOLUME /darkflow/parameter

CMD nohup python3 train
