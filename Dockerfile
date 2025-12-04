FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# 기본 패키지 + PPA 추가를 위한 도구 설치
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    bzip2 \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ca-certificates \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Python 3.11 설치 (deadsnakes PPA 사용)
RUN add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && apt-get install -y \
        python3.11 \
        python3.11-venv \
        python3.11-distutils \
    && rm -rf /var/lib/apt/lists/*

# Python 3.11용 pip 설치
RUN curl -sS https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3.11 get-pip.py && \
    rm get-pip.py

# venv 생성 (원하면 conda env 비슷하게 쓰기 위해)
ENV VENV_PATH=/opt/venv
RUN python3.11 -m venv $VENV_PATH

# venv 를 기본 python/pip 으로 사용
ENV PATH="$VENV_PATH/bin:$PATH"

# pip 최신화
RUN pip install --no-cache-dir --upgrade pip

# requirements 복사 및 설치
COPY requirements.txt /tmp/requirements.txt

# requirements.txt 안에 이미 PyTorch CUDA 11.8 extra-index-url 설정 있음
# (--extra-index-url https://download.pytorch.org/whl/cu118) :contentReference[oaicite:0]{index=0}
RUN pip install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /workspace

# bash 들어갔을 때 venv 자동 활성화되게 (선택)
RUN echo "source /opt/venv/bin/activate" >> ~/.bashrc

CMD ["/bin/bash"]
