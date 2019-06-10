FROM ubuntu:16.04
# FROM nvidia/cuda:9.0-base-ubuntu16.04

ARG PYTHON_VERSION=3.6

RUN apt-get update && apt-get install -y \
    fonts-powerline \
    zip \
    unzip \
    tmux \
    libpq-dev \
    curl \
    gdb \
    gcc \
    g++ \
    python \
    pkg-config \
    clang-format \
    libgsl2 \
    wget \
    libgsl-dev \
    libarmadillo6 \
    libarmadillo-dev \
    libboost-all-dev \
    vim \
    vim-gtk \
    zsh \
    autojump \
    keychain \
    sudo \
    git \
    htop \
    build-essential \
    cmake \
    make \
    python-dev \
    python3-dev \
    python3-dev \
    zlib1g-dev \
    python-numpy \
    zlib1g-dev \
    libjpeg-dev \
    xvfb \
    libav-tools \
    xorg-dev \
    python-opengl \
    libboost-all-dev \
    libsdl2-dev \
    swig \
    libglfw3-dev \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev \
    openmpi-bin \
    openmpi-doc \
    libopenmpi-dev \
    patchelf \
    python-pip \
    python3-pip \
    net-tools \
    openmpi-bin \
    openmpi-doc \
    libopenmpi-dev

# =========       mujoco       =================
ENV MUJOCO_VERSION=1.5.0 \
    MUJOCO_PATH=/root/.mujoco
RUN MUJOCO_ZIP="mjpro$(echo ${MUJOCO_VERSION} | sed -e "s/\.//g")_linux.zip" \
    && mkdir -p ${MUJOCO_PATH} \
    && wget -P ${MUJOCO_PATH} https://www.roboti.us/download/${MUJOCO_ZIP} \
    && unzip ${MUJOCO_PATH}/${MUJOCO_ZIP} -d ${MUJOCO_PATH} \
    && rm ${MUJOCO_PATH}/${MUJOCO_ZIP}
ENV LD_LIBRARY_PATH /root/.mujoco/mjpro150/bin:${LD_LIBRARY_PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib64:${LD_LIBRARY_PATH}
ENV MUJOCO_VERSION=1.3.1 \
    MUJOCO_PATH=/root/.mujoco
# include your Mujoco license
ADD ./mjkey /root/.mujoco/mjkey.txt

# ============   anaconda    =================
WORKDIR /
RUN apt-get update --fix-missing && apt-get install -y bzip2 ca-certificates \
    libglib2.0-0 libxext6 libsm6 libxrender1 git mercurial subversion
RUN wget  --no-check-certificate --quiet https://repo.continuum.io/archive/Anaconda2-2018.12-Linux-x86_64.sh && \
    /bin/bash /Anaconda2-2018.12-Linux-x86_64.sh -b -p /opt/conda && \
    rm /Anaconda2-2018.12-Linux-x86_64.sh && \
    ln -s /opt/anaconda/etc/profile.d/conda.sh /etc/profile.d/conda.sh
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
RUN . /opt/conda/etc/profile.d/conda.sh
RUN /opt/conda/bin/conda update -n base -c defaults conda
RUN echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc
RUN echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.zshrc
ADD ./tmaml.yml /tmp/conda.yml
RUN /opt/conda/bin/conda env create -f /tmp/conda.yml
RUN rm /tmp/conda.yml

WORKDIR /export/home
RUN chsh -s /usr/bin/zsh