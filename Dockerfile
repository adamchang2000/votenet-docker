FROM nvidia/cuda:11.0-base-ubuntu18.04

WORKDIR /usr/src/app

RUN apt-get update && apt-get install -y \
        software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update && apt-get install -y \
        python3.7 \
        python3-pip
RUN python3.7 -m pip install pip
RUN pip3 install --upgrade pip
RUN apt-get update && apt-get install -y \
        python3-distutils \
        python3-setuptools

COPY requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt

COPY doc ./doc
COPY models ./models
COPY pointnet2 ./pointnet2
COPY scannet ./scannet
COPY sunrgbd ./sunrgbd
COPY utils ./utils
COPY CODE_OF_CONDUCT.md ./CODE_OF_CONDUCT.md
COPY CONTRIBUTING.md ./CONTRIBUTING.md
COPY demo.py ./demo.py
COPY eval.py ./eval.py
COPY LICENSE ./LICENSE
COPY README.md ./README.md
COPY train.py ./train.py

CMD tail -f /dev/null