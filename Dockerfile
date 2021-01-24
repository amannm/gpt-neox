FROM nvidia/cuda:11.0.3-devel-ubuntu20.04

## OS Environment
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHON_VERSION=3
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        software-properties-common curl wget sudo cmake llvm-9-dev python3 python3-dev && \
        rm -f /usr/bin/python && \
        ln -s /usr/bin/python3 /usr/bin/python && \
        curl -O https://bootstrap.pypa.io/get-pip.py && \
        python get-pip.py && \
        rm get-pip.py && \
        pip install --upgrade pip
RUN add-apt-repository ppa:git-core/ppa -y && \
    apt-get update && \
    apt-get install -y --no-install-recommends git

# PyTorch
RUN pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html

# DeepSpeed
WORKDIR /tmp
RUN git clone https://github.com/EleutherAI/DeeperSpeed.git
RUN cd DeeperSpeed && \
    ./install.sh --allow_sudo
RUN rm -rf DeeperSpeed

# GPT-NeoX
WORKDIR /app
RUN mkdir data && \
    cd data && \
    wget https://github.com/lucidrains/reformer-pytorch/raw/master/examples/enwik8_simple/data/enwik8.gz
ADD requirements.txt requirements.txt
RUN pip install -r requirements.txt
ADD configs configs
ADD gpt_neox gpt_neox
ADD train_enwik8.py train_enwik8.py
ADD scripts/train_enwik8.sh train.sh
RUN chmod +x train.sh

ENTRYPOINT /app/train.sh
