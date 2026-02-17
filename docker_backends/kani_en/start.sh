#!/bin/bash
# Start Kani TTS 2 - English Accents (kani-en)
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed or not in PATH"; exit 1
fi

docker stop kani-en-proxy 2>/dev/null || true
docker stop kani-en-hf 2>/dev/null || true

echo "Starting Kani TTS English Accents (kani-en)..."
if docker run -d --rm \
    --name kani-en-hf \
    --platform linux/amd64 \
    -p 7864:7860 \
    --gpus all \
    registry.hf.space/nineninesix-kanitts-2-en:latest \
    python app.py; then
    echo "GPU mode enabled."
else
    echo "GPU failed, falling back to CPU."
    docker run -d --rm \
        --name kani-en-hf \
        --platform linux/amd64 \
        -p 7864:7860 \
        registry.hf.space/nineninesix-kanitts-2-en:latest \
        python app.py
fi

echo "Waiting for HF container..."
sleep 5

echo "Building and starting proxy..."
docker build -t audiobook-kani-en-proxy:latest -f Dockerfile.proxy .

docker run -d --rm \
    --name kani-en-proxy \
    --link kani-en-hf:kani-en-hf \
    -p 7866:7862 \
    -e GRADIO_URL=http://kani-en-hf:7860 \
    audiobook-kani-en-proxy:latest

echo ""
echo "Kani TTS English Accents started!"
echo "  Gradio UI: http://localhost:7864"
echo "  REST API:  http://localhost:7866"
