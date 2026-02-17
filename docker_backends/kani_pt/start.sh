#!/bin/bash
# Start Kani TTS 2 - Multilingual (kani-pt)
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed or not in PATH"; exit 1
fi

docker stop kani-proxy 2>/dev/null || true
docker stop kani-hf 2>/dev/null || true

echo "Starting Kani TTS Multilingual (kani-pt)..."
if docker run -d --rm \
    --name kani-hf \
    --platform linux/amd64 \
    -p 7860:7860 \
    --gpus all \
    registry.hf.space/nineninesix-kani-tts-2-pt:latest \
    python app.py; then
    echo "GPU mode enabled."
else
    echo "GPU failed, falling back to CPU."
    docker run -d --rm \
        --name kani-hf \
        --platform linux/amd64 \
        -p 7860:7860 \
        registry.hf.space/nineninesix-kani-tts-2-pt:latest \
        python app.py
fi

echo "Waiting for HF container..."
sleep 5

echo "Building and starting proxy..."
docker build -t audiobook-kani-pt-proxy:latest -f Dockerfile.proxy .

docker run -d --rm \
    --name kani-proxy \
    --link kani-hf:kani-hf \
    -p 7862:7862 \
    -e GRADIO_URL=http://kani-hf:7860 \
    audiobook-kani-pt-proxy:latest

echo ""
echo "Kani TTS Multilingual started!"
echo "  Gradio UI: http://localhost:7860"
echo "  REST API:  http://localhost:7862"
