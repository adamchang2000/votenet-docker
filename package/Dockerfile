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

CMD tail -f /dev/null