FROM python:3.12-slim

# ffmpeg: video proxy generation
# libglib2.0-0, libgl1: required by opencv-python
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libglib2.0-0 \
    libgl1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY application/ ./application/
COPY overnight_pipeline_optimized.py .
COPY bytetrack_custom.yaml .

EXPOSE 5000

CMD ["python", "application/app.py"]
