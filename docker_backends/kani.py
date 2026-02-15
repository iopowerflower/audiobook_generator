"""
Kani TTS Backend Configuration

This configures the Kani TTS Docker backend for integration with the audiobook generator.
"""

from .manager import DockerBackendConfig, DockerTTSBackend


def create_kani_backend(port: int = 7862) -> DockerTTSBackend:
    """
    Create a Kani TTS backend instance.
    
    Two deployment options:
    
    1. Use the HuggingFace Space Docker image directly:
       docker run -d -p 7860:7860 --gpus all registry.hf.space/nineninesix-kani-tts-2-pt:latest
       Then run our proxy to expose REST API on port 7862
    
    2. Use docker-compose (handles both):
       cd docker_backends/kani_tts && docker-compose --profile proxy up -d
    
    The backend expects the proxy API on the specified port.
    """
    config = DockerBackendConfig(
        backend_id="kani",
        name="Kani TTS",
        image="audiobook-kani-proxy:latest",  # Our proxy image
        port=port,
        gpu=False,  # Proxy doesn't need GPU, the HF container does
        health_endpoint="/health",
        startup_timeout=120,
    )
    return DockerTTSBackend(config)


# Voice ID prefix for Kani voices
KANI_VOICE_PREFIX = "kani:"


def is_kani_voice(voice_id: str) -> bool:
    """Check if a voice ID is a Kani TTS voice."""
    return voice_id.startswith(KANI_VOICE_PREFIX)


def kani_voice_id(backend_voice_id: str) -> str:
    """Convert a backend voice ID to an app voice ID."""
    return f"{KANI_VOICE_PREFIX}{backend_voice_id}"


def parse_kani_voice_id(voice_id: str) -> str:
    """Extract the backend voice ID from an app voice ID."""
    return voice_id.replace(KANI_VOICE_PREFIX, "", 1)
