#!/bin/bash
# Stop Kani TTS 2 - English Accents (kani-en)
echo "Stopping kani-en containers..."
docker stop kani-en-proxy 2>/dev/null || true
docker stop kani-en-hf 2>/dev/null || true
echo "Stopped."
