# 1. Base Image: CUDA 11.8 & Ubuntu 22.04
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# 2. 시스템 환경 설정
ENV DEBIAN_FRONTEND=noninteractive
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# 3. 필수 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    wget \
    bzip2 \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 4. Miniconda 설치 (안전한 방식)
# 경로를 명확히 잡고, 불필요한 폴더 생성 단계 제거
ENV PATH="/root/miniconda3/bin:${PATH}"

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh \
    && bash miniconda.sh -b -p /root/miniconda3 \
    && rm miniconda.sh \
    && /root/miniconda3/bin/conda init bash

# 5. [중요] Shell을 /bin/bash로 변경
# conda 명령어는 bash 환경에서 가장 안정적입니다.
SHELL ["/bin/bash", "-c"]

# 6. Conda 가상환경(ICoT) 생성
# 전체 경로 사용 + conda-forge 채널 추가 (Python 3.11 호환성 확보) + solver 업데이트
RUN /root/miniconda3/bin/conda create -n ICoT python=3.11 -y

# 7. requirements.txt 복사 및 라이브러리 설치
COPY requirements.txt /tmp/requirements.txt

# Conda 환경의 pip를 직접 지정하여 실행 (활성화 문제 원천 차단)
RUN /root/miniconda3/envs/ICoT/bin/pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cu118 \
    -r /tmp/requirements.txt

# 8. 작업 디렉토리 설정
WORKDIR /workspace

# 9. 컨테이너 접속 시 환경 자동 활성화
RUN echo "source /root/miniconda3/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate ICoT" >> ~/.bashrc

# 10. 실행 명령
CMD ["/bin/bash"]