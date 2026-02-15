#!/bin/bash
# Stop Kani TTS Docker backend

echo "Stopping Kani TTS containers..."
docker stop kani-proxy 2>/dev/null || true
docker stop kani-hf 2>/dev/null || true
echo "Kani TTS stopped."
