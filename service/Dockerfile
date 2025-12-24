FROM nvidia/cuda:12.8.0-runtime-ubuntu24.04

WORKDIR /content

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV HF_HOME=/content/cache
ENV PATH="/home/camenduru/.local/bin:${PATH}"

RUN apt update && apt install -y --no-install-recommends software-properties-common build-essential cmake \
    python3 python3-pip python-is-python3 python3-dev git git-lfs wget curl aria2 ffmpeg ca-certificates libgl1 libglib2.0-0 libegl1 && \
    apt clean && rm -rf /var/lib/apt/lists/* && \
    adduser --disabled-password --gecos '' camenduru && \
    adduser camenduru sudo && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers && \
    chown -R camenduru:camenduru /content /home && \
    chmod -R 777 /content /home

USER camenduru

RUN GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 --branch dev https://github.com/camenduru/TRELLIS.2-hf /content/TRELLIS.2 && \
    pip install --upgrade pip --break-system-packages && \
    pip install --no-cache-dir torch==2.8.0+cu128 torchvision==0.23.0+cu128 torchaudio==2.8.0+cu128 --index-url https://download.pytorch.org/whl/cu128 --break-system-packages && \
    pip install --no-cache-dir triton==3.4.0 pillow==12.0.0 imageio==2.37.2 imageio-ffmpeg==0.6.0 tqdm==4.67.1 easydict==1.13 opencv-python-headless==4.12.0.88 --break-system-packages && \
    pip install --no-cache-dir trimesh==4.10.1 transformers==4.57.3 zstandard==0.25.0 kornia==0.8.2 timm==1.0.22 --break-system-packages && \
    pip install --no-cache-dir https://github.com/camenduru/wheels/releases/download/trellis2/cumesh-0.0.1-cp312-cp312-linux_x86_64.whl --break-system-packages && \
    pip install --no-cache-dir https://github.com/camenduru/wheels/releases/download/trellis2/flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl --break-system-packages && \
    pip install --no-cache-dir https://github.com/camenduru/wheels/releases/download/trellis2/flex_gemm-0.0.1-cp312-cp312-linux_x86_64.whl --break-system-packages && \
    pip install --no-cache-dir https://github.com/camenduru/wheels/releases/download/trellis2/nvdiffrast-0.4.0-cp312-cp312-linux_x86_64.whl --break-system-packages && \
    pip install --no-cache-dir https://github.com/camenduru/wheels/releases/download/trellis2/nvdiffrec_render-0.0.0-cp312-cp312-linux_x86_64.whl --break-system-packages && \
    pip install --no-cache-dir https://github.com/camenduru/wheels/releases/download/trellis2/o_voxel-0.0.1-cp312-cp312-linux_x86_64.whl --break-system-packages && \
    pip install --no-cache-dir hf_transfer plyfile runpod boto3 --break-system-packages && \
    hf download camenduru/TRELLIS.2-4B && \
    hf download camenduru/RMBG-2.0 && \
    hf download camenduru/dinov3-vitl16-pretrain-lvd1689m

COPY --chown=camenduru:camenduru worker_runpod.py /content/TRELLIS.2/worker_runpod.py

WORKDIR /content/TRELLIS.2
CMD ["python3", "worker_runpod.py"]
