# Kani TTS Backend
#
# Option 1: Build from scratch (slower but more control)
# Option 2: Use HuggingFace Space image as base (faster)
#
# This Dockerfile uses Option 1 for reliability

FROM nvidia/cuda:12.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    git \
    git-lfs \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

WORKDIR /app

# Clone the HuggingFace Space (contains model code and speakers)
RUN git lfs install \
    && git clone https://huggingface.co/spaces/nineninesix/kani-tts-2-pt /app/kani-space \
    && cd /app/kani-space && git lfs pull

# Install Python dependencies
# First the space's own requirements
RUN pip install --no-cache-dir -r /app/kani-space/requirements.txt || true

# Then our API server requirements
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy our API server
COPY server.py /app/
COPY util_wrapper.py /app/

# Link the space's modules so we can import them
RUN ln -s /app/kani-space/speakers /app/speakers \
    && ln -s /app/kani-space/util.py /app/util.py

# Expose port
EXPOSE 7860

# Environment
ENV PORT=7860
ENV SPEAKER_MAP=/app/speakers/speaker_map.json

# Pre-download models during build (optional, makes first run faster)
# RUN python -c "from kani_tts import KaniTTS; KaniTTS('nineninesix/kani-tts-2-pt-450m', device_map='cpu')"

CMD ["python", "server.py"]
