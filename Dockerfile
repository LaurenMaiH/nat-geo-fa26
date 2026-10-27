FROM python:3.12-slim

# ffmpeg: video proxy generation
# libglib2.0-0, libgl1: required by opencv-python
# git: required to pip-install OpenAI CLIP (git+https://github.com/openai/CLIP.git),
#      which ultralytics YOLO-World needs at runtime for set_classes() text encoding
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libglib2.0-0 \
    libgl1 \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# torch/torchvision are installed BEFORE requirements.txt so that ultralytics / open-clip
# find them already satisfied. Otherwise pip pulls the default CUDA build from PyPI
# (several GB of NVIDIA libraries) even though the container has no GPU.
#   TORCH_VARIANT=cpu    (default) small CPU-only build
#   TORCH_VARIANT=cu130  CUDA build, set by docker-compose.gpu.yml
# Any index under https://download.pytorch.org/whl/ works (cpu, cu126, cu130, ...).
ARG TORCH_VARIANT=cpu
RUN pip install --no-cache-dir torch torchvision \
    --index-url https://download.pytorch.org/whl/${TORCH_VARIANT}

# Dependencies are installed before the code is copied, so editing application/ or the
# pipeline only rebuilds the last few layers below (seconds), not the pip installs above.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY application/ ./application/
COPY overnight_pipeline_optimized.py .
COPY bytetrack_custom.yaml .

EXPOSE 5000

CMD ["python", "application/app.py"]
