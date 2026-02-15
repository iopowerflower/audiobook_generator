# Docker-based TTS Backends

This folder contains Docker-based TTS backends that integrate with the audiobook generator.

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  Flask App (app.py)                                              │
│  ├── Local backends (Kokoro, Piper, Orpheus)                     │
│  └── Docker backends (via HTTP API)                              │
│        │                                                         │
│        ▼                                                         │
│  DockerBackendManager (manager.py)                               │
│  ├── Common interface (interface.py)                             │
│  └── Per-backend configuration                                   │
└──────────────────────────────────────────────────────────────────┘
                           │
                           │ HTTP (localhost:PORT)
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│  Docker Container (e.g., Kani TTS)                               │
│  ├── Model files                                                 │
│  ├── REST API server                                             │
│  └── GPU/CPU inference                                           │
└──────────────────────────────────────────────────────────────────┘
```

## Benefits

1. **Isolation**: Each model runs in its own container with its own dependencies
2. **No conflicts**: Different Python versions, CUDA versions, etc. don't conflict
3. **Easy updates**: Just pull new container images
4. **Portability**: Works the same on any system with Docker
5. **GPU sharing**: Containers can share GPU resources via NVIDIA Docker

## Available Backends

### Kani TTS (`kani_tts/`)

A fast, multilingual TTS model with voice cloning support.

**Features:**
- 🎤 Voice cloning from audio samples
- 🌐 Multilingual (English, Kyrgyz, Spanish)
- 🎚️ Adjustable temperature, top_p, repetition penalty
- 👥 Pre-built speaker embeddings

**Quick Start:**
```bash
cd docker_backends/kani_tts
./start.sh
```

**Manual Start:**
```bash
# Start HuggingFace Space container (includes model)
docker run -d --name kani-hf -p 7860:7860 --gpus all \
    registry.hf.space/nineninesix-kani-tts-2-pt:latest

# Start REST API proxy
docker build -t kani-proxy -f Dockerfile.proxy .
docker run -d --name kani-proxy -p 7862:7862 \
    --link kani-hf:kani-hf \
    -e GRADIO_URL=http://kani-hf:7860 \
    kani-proxy
```

**Ports:**
- `7860`: Gradio UI (HuggingFace Space)
- `7862`: REST API (for audiobook generator)

## Common Interface

All Docker backends expose the same REST API:

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/capabilities` | GET | Backend capabilities |
| `/voices` | GET | Available voices |
| `/synthesize` | POST | Synthesize speech |
| `/clone_voice` | POST | Clone voice from audio |

### Synthesis Request

```json
POST /synthesize
{
    "text": "Hello, world!",
    "voice_id": "speaker:Kore (en)",
    "speed": 1.0,
    "language": "en",
    "speaker_embedding": [0.1, 0.2, ...],  // optional, for cloned voices
    "custom_params": {                      // optional, backend-specific
        "temperature": 1.0,
        "top_p": 0.95
    }
}
```

### Voice Cloning

```bash
curl -X POST http://localhost:7862/clone_voice \
    -F "audio=@sample.wav" \
    -F "name=My Voice"
```

Returns embedding that can be reused in synthesis requests.

## Adding New Backends

1. Create a folder: `docker_backends/my_tts/`
2. Add files:
   - `Dockerfile` or `Dockerfile.proxy`
   - `server.py` (FastAPI REST API)
   - `requirements.txt`
   - `start.sh` and `stop.sh`
3. Register in `app.py`:
   ```python
   # In _init_docker_backends()
   backends_to_check = [
       ("kani", 7862, "Kani TTS"),
       ("mytts", 7863, "My TTS"),  # Add your backend
   ]
   ```

## Troubleshooting

### Container won't start
```bash
docker logs kani-hf
```

### No GPU support
Ensure NVIDIA Container Toolkit is installed:
```bash
nvidia-smi  # Should show GPU
docker info | grep nvidia  # Should show nvidia runtime
```

### Port conflicts
Change ports in `start.sh` and update the registration in `app.py`.

### Voice not appearing in UI
1. Check backend is running: `curl http://localhost:7862/health`
2. Refresh voices: `POST /api/docker/backends/kani/refresh`
