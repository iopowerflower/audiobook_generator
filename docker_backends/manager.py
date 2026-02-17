"""
Docker Backend Manager

Handles starting, stopping, and communicating with Docker-based TTS backends.
"""

import subprocess
import time
import requests
import json
from typing import Optional
from dataclasses import dataclass

from .interface import (
    TTSBackend, TTSCapabilities, TTSVoice, 
    SynthesisRequest, SynthesisResponse
)


@dataclass
class DockerBackendConfig:
    """Configuration for a Docker-based TTS backend."""
    
    backend_id: str                         # Unique ID (e.g., "kani")
    name: str                               # Display name (e.g., "Kani TTS")
    image: str                              # Docker image name
    port: int                               # Port to expose
    gpu: bool = True                        # Whether to use GPU
    env_vars: dict = None                   # Environment variables
    health_endpoint: str = "/health"        # Health check endpoint
    startup_timeout: int = 120              # Seconds to wait for startup


class DockerTTSBackend(TTSBackend):
    """
    TTS backend that communicates with a Docker container via HTTP.
    """
    
    def __init__(self, config: DockerBackendConfig):
        self.config = config
        self._container_id: Optional[str] = None
        self._base_url = f"http://localhost:{config.port}"
        self._capabilities: Optional[TTSCapabilities] = None
        self._voices: Optional[list[TTSVoice]] = None
    
    @property
    def name(self) -> str:
        return self.config.name
    
    @property
    def backend_id(self) -> str:
        return self.config.backend_id
    
    def is_available(self) -> bool:
        """Check if the backend is running and healthy."""
        try:
            resp = requests.get(
                f"{self._base_url}{self.config.health_endpoint}",
                timeout=5
            )
            return resp.status_code == 200
        except Exception:
            return False
    
    def invalidate_cache(self) -> None:
        """Clear cached capabilities and voices so the next call re-fetches."""
        self._capabilities = None
        self._voices = None

    def get_capabilities(self) -> TTSCapabilities:
        """Get capabilities from the backend."""
        if self._capabilities is not None:
            return self._capabilities
        
        try:
            resp = requests.get(f"{self._base_url}/capabilities", timeout=10)
            resp.raise_for_status()
            data = resp.json()
            
            self._capabilities = TTSCapabilities(
                has_predefined_voices=data.get("has_predefined_voices", True),
                voice_cloning=data.get("voice_cloning", False),
                voice_cloning_formats=data.get("voice_cloning_formats", []),
                emotion_tags=data.get("emotion_tags", False),
                emotion_options=data.get("emotion_options", []),
                streaming=data.get("streaming", False),
                sample_rate=data.get("sample_rate", 22050),
                output_format=data.get("output_format", "wav"),
                multilingual=data.get("multilingual", False),
                languages=data.get("languages", ["en"]),
                speaker_embedding=data.get("speaker_embedding", False),
                embedding_dim=data.get("embedding_dim", 0),
                custom_params=data.get("custom_params", {}),
            )
            return self._capabilities
        except Exception as e:
            print(f"[{self.backend_id}] Failed to get capabilities: {e}")
            return TTSCapabilities()
    
    def get_voices(self) -> list[TTSVoice]:
        """Get available voices from the backend."""
        try:
            resp = requests.get(f"{self._base_url}/voices", timeout=30)
            resp.raise_for_status()
            data = resp.json()
            
            voices = []
            for v in data.get("voices", []):
                voices.append(TTSVoice(
                    id=v["id"],
                    name=v.get("name", v["id"]),
                    gender=v.get("gender", "UNKNOWN"),
                    language=v.get("language", "en"),
                    description=v.get("description", ""),
                    preview_url=v.get("preview_url"),
                    is_cloned=v.get("is_cloned", False),
                    embedding=v.get("embedding"),
                ))
            
            self._voices = voices
            return voices
        except Exception as e:
            print(f"[{self.backend_id}] Failed to get voices: {e}")
            return []
    
    def synthesize(self, request: SynthesisRequest) -> SynthesisResponse:
        """Synthesize speech."""
        payload = {
            "text": request.text,
            "voice_id": request.voice_id,
            "speed": request.speed,
            "language": request.language,
        }
        
        if request.emotion:
            payload["emotion"] = request.emotion
        if request.speaker_embedding:
            payload["speaker_embedding"] = request.speaker_embedding
        if request.custom_params:
            payload["custom_params"] = request.custom_params
        
        resp = requests.post(
            f"{self._base_url}/synthesize",
            json=payload,
            timeout=300,  # TTS can be slow
        )
        resp.raise_for_status()
        
        # Response is audio bytes with headers for metadata
        sample_rate = int(resp.headers.get("X-Sample-Rate", 22050))
        audio_format = resp.headers.get("X-Audio-Format", "wav")
        duration = resp.headers.get("X-Duration-Seconds")
        
        return SynthesisResponse(
            audio_bytes=resp.content,
            sample_rate=sample_rate,
            format=audio_format,
            duration_seconds=float(duration) if duration else None,
        )
    
    def clone_voice(self, audio_bytes: bytes, name: str, sample_rate: int = 16000) -> TTSVoice:
        """Clone a voice from audio."""
        caps = self.get_capabilities()
        if not caps.voice_cloning:
            raise NotImplementedError("Voice cloning not supported")
        
        resp = requests.post(
            f"{self._base_url}/clone_voice",
            files={"audio": ("audio.wav", audio_bytes, "audio/wav")},
            data={"name": name, "sample_rate": sample_rate},
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        
        return TTSVoice(
            id=data["id"],
            name=data.get("name", name),
            gender=data.get("gender", "UNKNOWN"),
            language=data.get("language", "en"),
            description=data.get("description", "Cloned voice"),
            is_cloned=True,
            embedding=data.get("embedding"),
        )
    
    def start(self) -> bool:
        """Start the Docker container."""
        if self.is_available():
            print(f"[{self.backend_id}] Already running")
            return True
        
        # Build docker run command
        cmd = ["docker", "run", "-d", "--rm"]
        cmd.extend(["-p", f"{self.config.port}:{self.config.port}"])
        
        if self.config.gpu:
            cmd.extend(["--gpus", "all"])
        
        if self.config.env_vars:
            for key, value in self.config.env_vars.items():
                cmd.extend(["-e", f"{key}={value}"])
        
        cmd.append(self.config.image)
        
        print(f"[{self.backend_id}] Starting container: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                print(f"[{self.backend_id}] Failed to start: {result.stderr}")
                return False
            
            self._container_id = result.stdout.strip()
            print(f"[{self.backend_id}] Container started: {self._container_id[:12]}")
            
            # Wait for health check
            return self._wait_for_health()
        except Exception as e:
            print(f"[{self.backend_id}] Error starting container: {e}")
            return False
    
    def stop(self) -> bool:
        """Stop the Docker container."""
        if not self._container_id:
            return True
        
        try:
            subprocess.run(
                ["docker", "stop", self._container_id],
                capture_output=True,
                timeout=30
            )
            self._container_id = None
            return True
        except Exception as e:
            print(f"[{self.backend_id}] Error stopping container: {e}")
            return False
    
    def _wait_for_health(self) -> bool:
        """Wait for the container to become healthy."""
        start = time.time()
        while time.time() - start < self.config.startup_timeout:
            if self.is_available():
                print(f"[{self.backend_id}] Container is healthy")
                return True
            time.sleep(2)
        
        print(f"[{self.backend_id}] Timeout waiting for container")
        return False


class DockerBackendManager:
    """
    Manages multiple Docker-based TTS backends.
    """
    
    def __init__(self):
        self._backends: dict[str, DockerTTSBackend] = {}
    
    def register(self, config: DockerBackendConfig) -> DockerTTSBackend:
        """Register a new backend."""
        backend = DockerTTSBackend(config)
        self._backends[config.backend_id] = backend
        return backend
    
    def get(self, backend_id: str) -> Optional[DockerTTSBackend]:
        """Get a backend by ID."""
        return self._backends.get(backend_id)
    
    def list_available(self) -> list[DockerTTSBackend]:
        """List all backends that are currently running."""
        return [b for b in self._backends.values() if b.is_available()]
    
    def list_all(self) -> list[DockerTTSBackend]:
        """List all registered backends."""
        return list(self._backends.values())
    
    def start_all(self) -> dict[str, bool]:
        """Start all registered backends."""
        results = {}
        for backend_id, backend in self._backends.items():
            results[backend_id] = backend.start()
        return results
    
    def stop_all(self) -> dict[str, bool]:
        """Stop all registered backends."""
        results = {}
        for backend_id, backend in self._backends.items():
            results[backend_id] = backend.stop()
        return results
