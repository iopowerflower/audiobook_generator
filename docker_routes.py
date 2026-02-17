"""
Flask Blueprint for Docker TTS backends.

Handles registration, start/stop, voice refresh, and synthesis.
"""

import io
import os
import subprocess
import requests
from pathlib import Path
from flask import Blueprint, request, jsonify

from docker_backends.manager import DockerBackendManager, DockerBackendConfig
from docker_backends.interface import SynthesisRequest
from docker_backends.registry import KNOWN_BACKENDS

docker_bp = Blueprint("docker", __name__)

# Module state
DOCKER_TTS_VOICES = {}
_DOCKER_BACKENDS = {}
_DOCKER_BACKEND_MANAGER = DockerBackendManager()

# Base directory for per-backend proxy folders
_BASE_DIR = Path(__file__).resolve().parent
_BACKENDS_DIR = _BASE_DIR / "docker_backends"


def is_docker_tts_voice(voice_id: str) -> bool:
    """Check if a voice ID is from a Docker-based backend."""
    return isinstance(voice_id, str) and voice_id.startswith("docker:")


def register_docker_backend(backend_id: str, port: int, name: str = None) -> bool:
    """Register a Docker TTS backend by port."""
    global DOCKER_TTS_VOICES

    config = DockerBackendConfig(
        backend_id=backend_id,
        name=name or backend_id,
        image=f"audiobook-{backend_id}:latest",
        port=port,
        gpu=False,
        health_endpoint="/health",
        startup_timeout=60,
    )

    backend = _DOCKER_BACKEND_MANAGER.register(config)
    _DOCKER_BACKENDS[backend_id] = backend

    if backend.is_available():
        _refresh_docker_backend_voices(backend_id)
        return True
    return False


def _refresh_docker_backend_voices(backend_id: str) -> None:
    """Refresh voices and capabilities from a Docker backend."""
    global DOCKER_TTS_VOICES

    backend = _DOCKER_BACKENDS.get(backend_id)
    if not backend or not backend.is_available():
        return

    # Clear cached data so we get fresh capabilities/voices from the proxy
    backend.invalidate_cache()

    try:
        voices = backend.get_voices()
        caps = backend.get_capabilities()

        for voice in voices:
            app_voice_id = f"docker:{backend_id}:{voice.id}"
            DOCKER_TTS_VOICES[app_voice_id] = {
                "name": voice.name,
                "code": "docker",
                "gender": voice.gender,
                "type": f"{backend.name} (Docker)",
                "desc": voice.description or f"From {backend.name}",
                "region": f"🐳 {backend.name}",
                "_backend_id": backend_id,
                "_backend_voice_id": voice.id,
                "_is_cloned": voice.is_cloned,
                "_embedding": voice.embedding,
                "_capabilities": caps,
            }
    except Exception as e:
        print(f"[docker] Failed to load voices from {backend_id}: {e}")


def synthesize_docker_tts_wav_bytes(
    text: str,
    voice_id: str,
    speaking_rate: float = 1.0,
    custom_params: dict = None,
) -> bytes:
    """Synthesize speech using a Docker-based TTS backend."""
    if not is_docker_tts_voice(voice_id):
        raise Exception("Invalid Docker TTS voice id")

    parts = voice_id.split(":", 2)
    if len(parts) != 3:
        raise Exception(f"Invalid voice id format: {voice_id}")

    _, backend_id, backend_voice_id = parts
    backend = _DOCKER_BACKENDS.get(backend_id)
    if not backend:
        raise Exception(f"Backend {backend_id} not registered")
    if not backend.is_available():
        raise Exception(f"Backend {backend_id} is not running")

    voice_info = DOCKER_TTS_VOICES.get(voice_id, {})
    embedding = voice_info.get("_embedding")

    request = SynthesisRequest(
        text=text,
        voice_id=backend_voice_id,
        speed=speaking_rate,
        speaker_embedding=embedding,
        custom_params=custom_params or {},
    )

    response = backend.synthesize(request)

    if response.format == "wav":
        return response.audio_bytes
    import soundfile as sf
    audio, sr = sf.read(io.BytesIO(response.audio_bytes))
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV")
    return buf.getvalue()


def docker_clone_voice(
    backend_id: str, audio_bytes: bytes, name: str, sample_rate: int = 16000
) -> dict:
    """
    Clone a voice by extracting a speaker embedding from audio.

    Runs the extraction script inside the HF container via docker exec,
    then stores the 128-dim embedding for use with mode="json" synthesis.
    """
    import json
    import tempfile

    cfg = next((b for b in KNOWN_BACKENDS if b["id"] == backend_id), None)
    if not cfg:
        raise Exception(f"Unknown backend: {backend_id}")

    backend = _DOCKER_BACKENDS.get(backend_id)
    if not backend:
        raise Exception(f"Backend {backend_id} not registered")

    hf_container = cfg["hf_container"]
    extract_script = _BACKENDS_DIR / "extract_embedding.py"

    # Write audio to temp file and copy into HF container
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    # Use /home/user/ inside the container (avoids Windows path issues)
    container_audio = f"{hf_container}:/home/user/clone_audio.wav"
    container_script = f"{hf_container}:/home/user/extract_embedding.py"

    try:
        # Copy audio file into container
        r = subprocess.run(
            ["docker", "cp", tmp_path, container_audio],
            capture_output=True, text=True, timeout=30,
        )
        if r.returncode != 0:
            raise Exception(f"Failed to copy audio: {r.stderr}")

        # Copy extraction script into container
        r = subprocess.run(
            ["docker", "cp", str(extract_script), container_script],
            capture_output=True, text=True, timeout=30,
        )
        if r.returncode != 0:
            raise Exception(f"Failed to copy script: {r.stderr}")

        # Run extraction inside HF container (has torch, SpeakerEmbedder, etc.)
        r = subprocess.run(
            ["docker", "exec", hf_container, "python",
             "/home/user/extract_embedding.py", "/home/user/clone_audio.wav"],
            capture_output=True, text=True, timeout=120,
        )
        if r.returncode != 0:
            raise Exception(f"Embedding extraction failed: {r.stderr}")

        # Parse the embedding vector from stdout
        embedding = json.loads(r.stdout.strip())
        if not isinstance(embedding, list) or len(embedding) != 128:
            raise Exception(f"Bad embedding: expected 128-dim list, got {type(embedding)} len={len(embedding) if isinstance(embedding, list) else '?'}")

    finally:
        os.unlink(tmp_path)

    # Register the cloned voice with its embedding
    voice_id = f"cloned:{name}"
    app_voice_id = f"docker:{backend_id}:{voice_id}"
    backend_name = cfg.get("name", backend_id)

    DOCKER_TTS_VOICES[app_voice_id] = {
        "name": f"🎤 {name}",
        "code": "docker",
        "gender": "UNKNOWN",
        "type": f"{backend_name} (Cloned)",
        "desc": "Cloned voice",
        "region": f"🐳 {backend_name}",
        "_backend_id": backend_id,
        "_backend_voice_id": voice_id,
        "_is_cloned": True,
        "_embedding": embedding,
    }

    return {"id": app_voice_id, "name": name, "embedding": embedding}


def _init_docker_backends() -> None:
    """Check for running Docker backends and register them."""
    for cfg in KNOWN_BACKENDS:
        try:
            resp = requests.get(
                f"http://localhost:{cfg['proxy_port']}/health", timeout=2
            )
            if resp.status_code == 200:
                register_docker_backend(
                    cfg["id"], cfg["proxy_port"], cfg["name"]
                )
        except Exception:
            pass


def _start_backend(backend_id: str) -> tuple[bool, str]:
    """Start HF container + proxy for a registered backend. Returns (success, message)."""
    cfg = next((b for b in KNOWN_BACKENDS if b["id"] == backend_id), None)
    if not cfg:
        return False, f"Unknown backend: {backend_id}"

    # Stop existing
    subprocess.run(
        ["docker", "stop", cfg["proxy_container"]],
        capture_output=True,
        timeout=15,
    )
    subprocess.run(
        ["docker", "stop", cfg["hf_container"]],
        capture_output=True,
        timeout=15,
    )

    # Start HF container (try GPU first, fallback to CPU)
    try:
        r = subprocess.run(
            ["docker", "run", "-d", "--rm", "--name", cfg["hf_container"],
             "--platform", "linux/amd64", "-p", f"{cfg['hf_port']}:7860",
             "--gpus", "all", cfg["image"], "python", "app.py"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if r.returncode != 0:
            r = subprocess.run(
                ["docker", "run", "-d", "--rm", "--name", cfg["hf_container"],
                 "--platform", "linux/amd64", "-p", f"{cfg['hf_port']}:7860",
                 cfg["image"], "python", "app.py"],
                capture_output=True,
                text=True,
                timeout=60,
            )
    except Exception as e:
        return False, str(e)

    if r.returncode != 0:
        return False, r.stderr or "Failed to start HF container"

    # Wait for HF to be ready (Gradio can take 30-60s to load models)
    import time
    time.sleep(5)
    for _ in range(60):
        try:
            resp = requests.get(f"http://localhost:{cfg['hf_port']}/", timeout=5)
            if resp.status_code == 200:
                break
        except Exception:
            pass
        time.sleep(2)
    else:
        return False, "HF container did not become ready (timeout)"

    # Build and start proxy from per-backend directory
    proxy_dir = _BACKENDS_DIR / cfg.get("proxy_dir", cfg["id"])
    if not proxy_dir.exists():
        return False, f"Proxy dir not found: {proxy_dir}"

    proxy_image = cfg.get("proxy_image", f"audiobook-{cfg['id']}-proxy:latest")

    build = subprocess.run(
        ["docker", "build", "-t", proxy_image,
         "-f", "Dockerfile.proxy", "."],
        cwd=str(proxy_dir),
        capture_output=True,
        text=True,
        timeout=120,
    )
    if build.returncode != 0:
        return False, build.stderr or "Proxy build failed"

    hf_host = cfg["hf_container"]
    proxy_cmd = [
        "docker", "run", "-d", "--rm",
        "--name", cfg["proxy_container"],
        "--link", f"{hf_host}:{hf_host}",
        "-p", f"{cfg['proxy_port']}:7862",
        "-e", f"GRADIO_URL=http://{hf_host}:7860",
        proxy_image,
    ]
    r2 = subprocess.run(proxy_cmd, capture_output=True, text=True, timeout=30)
    if r2.returncode != 0:
        return False, r2.stderr or "Proxy start failed"

    # Register and load voices
    register_docker_backend(cfg["id"], cfg["proxy_port"], cfg["name"])
    return True, "Started"


def _stop_backend(backend_id: str) -> tuple[bool, str]:
    """Stop HF + proxy containers for a backend."""
    cfg = next((b for b in KNOWN_BACKENDS if b["id"] == backend_id), None)
    if not cfg:
        return False, f"Unknown backend: {backend_id}"

    # Remove from our registry
    keys_to_remove = [k for k in DOCKER_TTS_VOICES if k.startswith(f"docker:{backend_id}:")]
    for k in keys_to_remove:
        del DOCKER_TTS_VOICES[k]
    if backend_id in _DOCKER_BACKENDS:
        del _DOCKER_BACKENDS[backend_id]

    for c in [cfg["proxy_container"], cfg["hf_container"]]:
        subprocess.run(["docker", "stop", c], capture_output=True, timeout=15)

    return True, "Stopped"


# Initialize on import
try:
    _init_docker_backends()
except Exception as e:
    print(f"[docker] Backend initialization failed: {e}")


# --- Routes ---

@docker_bp.route("/api/docker/backends")
def list_docker_backends():
    """List all known backends with status (from registry + running state)."""
    result = []
    for cfg in KNOWN_BACKENDS:
        backend = _DOCKER_BACKENDS.get(cfg["id"])
        available = backend.is_available() if backend else False
        voice_count = len([v for v in DOCKER_TTS_VOICES if v.startswith(f"docker:{cfg['id']}:")])
        voice_cloning = False
        if available and backend:
            try:
                voice_cloning = backend.get_capabilities().voice_cloning
            except Exception:
                pass
        result.append({
            "id": cfg["id"],
            "name": cfg["name"],
            "port": cfg["proxy_port"],
            "available": available,
            "voice_count": voice_count,
            "voice_cloning": voice_cloning,
        })
    return jsonify({"backends": result})


@docker_bp.route("/api/docker/backends/<backend_id>/start", methods=["POST"])
def start_docker_backend(backend_id):
    """Start a Docker backend (HF + proxy)."""
    success, msg = _start_backend(backend_id)
    if success:
        return jsonify({"status": "ok", "message": msg})
    return jsonify({"error": msg}), 500


@docker_bp.route("/api/docker/backends/<backend_id>/stop", methods=["POST"])
def stop_docker_backend(backend_id):
    """Stop a Docker backend."""
    success, msg = _stop_backend(backend_id)
    if success:
        return jsonify({"status": "ok", "message": msg})
    return jsonify({"error": msg}), 500


@docker_bp.route("/api/docker/backends/<backend_id>/capabilities")
def get_docker_backend_capabilities(backend_id):
    backend = _DOCKER_BACKENDS.get(backend_id)
    if not backend:
        return jsonify({"error": f"Backend {backend_id} not found"}), 404
    if not backend.is_available():
        return jsonify({"error": f"Backend {backend_id} is not running"}), 503
    caps = backend.get_capabilities()
    return jsonify({
        "has_predefined_voices": caps.has_predefined_voices,
        "voice_cloning": caps.voice_cloning,
        "voice_cloning_formats": caps.voice_cloning_formats,
        "emotion_tags": caps.emotion_tags,
        "emotion_options": caps.emotion_options,
        "streaming": caps.streaming,
        "sample_rate": caps.sample_rate,
        "multilingual": caps.multilingual,
        "languages": caps.languages,
        "speaker_embedding": caps.speaker_embedding,
        "embedding_dim": caps.embedding_dim,
        "custom_params": caps.custom_params,
    })


@docker_bp.route("/api/docker/backends/<backend_id>/refresh", methods=["POST"])
def refresh_docker_backend_voices(backend_id):
    backend = _DOCKER_BACKENDS.get(backend_id)
    if not backend:
        return jsonify({"error": f"Backend {backend_id} not found"}), 404
    if not backend.is_available():
        return jsonify({"error": f"Backend {backend_id} is not running"}), 503
    _refresh_docker_backend_voices(backend_id)
    voice_count = len([v for v in DOCKER_TTS_VOICES if v.startswith(f"docker:{backend_id}:")])
    return jsonify({"status": "ok", "voice_count": voice_count})


@docker_bp.route("/api/docker/clone", methods=["POST"])
def clone_voice_endpoint():
    backend_id = request.form.get("backend_id")
    name = request.form.get("name")
    audio_file = request.files.get("audio")
    if not backend_id or not name or not audio_file:
        return jsonify({"error": "Missing: backend_id, name, audio"}), 400
    cfg = next((b for b in KNOWN_BACKENDS if b["id"] == backend_id), None)
    if not cfg:
        return jsonify({"error": f"Backend {backend_id} not found"}), 404
    backend = _DOCKER_BACKENDS.get(backend_id)
    if not backend or not backend.is_available():
        return jsonify({"error": f"Backend {backend_id} is not running"}), 503
    try:
        result = docker_clone_voice(backend_id, audio_file.read(), name)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@docker_bp.route("/api/docker/register", methods=["POST"])
def register_docker_backend_endpoint():
    data = request.get_json()
    if not data:
        return jsonify({"error": "Missing JSON body"}), 400
    backend_id = data.get("backend_id")
    port = data.get("port")
    name = data.get("name")
    if not backend_id or not port:
        return jsonify({"error": "Missing: backend_id, port"}), 400
    try:
        success = register_docker_backend(backend_id, int(port), name)
        if success:
            voice_count = len([v for v in DOCKER_TTS_VOICES if v.startswith(f"docker:{backend_id}:")])
            return jsonify({"status": "ok", "voice_count": voice_count})
        return jsonify({"error": "Backend not available on port"}), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 500
