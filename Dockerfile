FROM tensorflow/tensorflow:1.12.0-gpu-py3

RUN apt-get -y update
RUN apt-get -y install libglib2.0-0
RUN apt-get -y install libsm6 libxext6 libxrender-dev
RUN apt-get -y install graphviz
RUN apt-get -y install git

RUN pip install git+git://github.com/neelsj/keras.git@2.2.4_tfdimensionfix
RUN pip install scikit-learn
RUN pip install scikit-image
RUN pip install opencv-python
RUN pip install pydot
RUN pip install graphviz