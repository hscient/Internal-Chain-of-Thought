# Base Image: CUDA 11.8 & Ubuntu 22.04
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# 시스템 설정
ENV DEBIAN_FRONTEND=noninteractive
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# 필수 패키지 설치 (git, opencv dependency 등)
RUN apt-get update && apt-get install -y \
    wget bzip2 git libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Miniconda 설치
ENV PATH="/root/miniconda3/bin:${PATH}"
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash Miniconda3-latest-Linux-x86_64.sh -b -p /root/miniconda3 \
    && rm -f Miniconda3-latest-Linux-x86_64.sh \
    && conda init bash

# Conda 환경(ICoT) 생성
RUN conda create -n ICoT python=3.11 -y

# requirements.txt 설치 (캐시 없이 설치하여 용량 절약)
COPY requirements.txt /tmp/requirements.txt
RUN /root/miniconda3/envs/ICoT/bin/pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu118 -r /tmp/requirements.txt

# 작업 경로 및 실행 설정
WORKDIR /workspace
RUN echo "conda activate ICoT" >> ~/.bashrc
CMD ["/bin/bash"]