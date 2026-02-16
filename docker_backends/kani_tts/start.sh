#!/bin/bash
# Start Kani TTS Docker backend
#
# Option 1: Use the HuggingFace Space image directly (recommended)
# Option 2: Build and use our custom API server
#
# This script uses Option 1 for simplicity.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Check for Docker
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed or not in PATH"
    exit 1
fi

# Stop any existing containers
echo "Stopping any existing Kani containers..."
docker stop kani-hf 2>/dev/null || true
docker stop kani-proxy 2>/dev/null || true

# Start the HuggingFace Space container
echo "Starting Kani TTS (HuggingFace Space)..."
# Try GPU first; if it fails, automatically fall back to CPU.
if docker run -d --rm \
    --name kani-hf \
    --platform linux/amd64 \
    -p 7860:7860 \
    --gpus all \
    registry.hf.space/nineninesix-kani-tts-2-pt:latest \
    python app.py; then
    echo "NVIDIA GPU mode enabled (--gpus all)."
else
    echo "Warning: GPU launch failed, falling back to CPU mode."
    docker run -d --rm \
        --name kani-hf \
        --platform linux/amd64 \
        -p 7860:7860 \
        registry.hf.space/nineninesix-kani-tts-2-pt:latest \
        python app.py
fi

echo "Waiting for container to start..."
sleep 5

# Build and start the proxy (converts Gradio API to REST API)
echo "Building and starting REST API proxy..."
docker build -t audiobook-kani-proxy:latest -f Dockerfile.proxy .

docker run -d --rm \
    --name kani-proxy \
    --link kani-hf:kani-hf \
    -p 7862:7862 \
    -e GRADIO_URL=http://kani-hf:7860 \
    audiobook-kani-proxy:latest

echo ""
echo "Kani TTS started!"
echo "  Gradio UI: http://localhost:7860"
echo "  REST API:  http://localhost:7862"
echo ""
echo "To use in the audiobook generator, the app will auto-detect the backend."
echo "Or manually register via: POST /api/docker/register {backend_id: 'kani', port: 7862}"
