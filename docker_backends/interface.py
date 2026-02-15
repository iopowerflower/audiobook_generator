"""
Common interface for Docker-based TTS backends.

Each TTS model (Kani, F5-TTS, future models) implements this interface
via a REST API running inside a Docker container.
"""

from dataclasses import dataclass, field
from typing import Optional, Any
from abc import ABC, abstractmethod


@dataclass
class TTSCapabilities:
    """Describes what features a TTS backend supports."""
    
    # Voice selection
    has_predefined_voices: bool = True      # Has built-in voice options
    
    # Voice cloning
    voice_cloning: bool = False             # Can clone voice from audio sample
    voice_cloning_formats: list[str] = field(default_factory=list)  # ["wav", "mp3"]
    
    # Emotional/expressive speech
    emotion_tags: bool = False              # Supports emotion markup like <happy>
    emotion_options: list[str] = field(default_factory=list)  # ["happy", "sad", "angry"]
    
    # Audio output
    streaming: bool = False                 # Can stream audio in chunks
    sample_rate: int = 22050                # Output sample rate
    output_format: str = "wav"              # Primary output format
    
    # Language support
    multilingual: bool = False
    languages: list[str] = field(default_factory=lambda: ["en"])
    
    # Advanced features
    speaker_embedding: bool = False         # Accepts embedding vectors directly
    embedding_dim: int = 0                  # Embedding dimension if supported
    
    # Model-specific parameters (exposed in UI as sliders/inputs)
    custom_params: dict[str, dict] = field(default_factory=dict)
    # Example: {"temperature": {"min": 0.1, "max": 2.0, "default": 1.0, "step": 0.1}}


@dataclass
class TTSVoice:
    """Represents a voice option from a TTS backend."""
    
    id: str                                 # Unique voice ID
    name: str                               # Display name
    gender: str = "UNKNOWN"                 # MALE, FEMALE, UNKNOWN
    language: str = "en"                    # Language code
    description: str = ""                   # Short description
    preview_url: Optional[str] = None       # URL to preview audio (if available)
    
    # For cloned voices
    is_cloned: bool = False                 # Whether this is a user-cloned voice
    embedding: Optional[list[float]] = None # Embedding if cloned


@dataclass
class SynthesisRequest:
    """Request for speech synthesis."""
    
    text: str
    voice_id: str
    speed: float = 1.0
    language: str = "en"
    
    # Optional: for emotion support
    emotion: Optional[str] = None
    
    # Optional: for voice cloning / custom embedding
    speaker_embedding: Optional[list[float]] = None
    
    # Model-specific params (temperature, top_p, etc.)
    custom_params: dict[str, Any] = field(default_factory=dict)


@dataclass
class SynthesisResponse:
    """Response from speech synthesis."""
    
    audio_bytes: bytes
    sample_rate: int
    format: str = "wav"                     # "wav", "mp3", etc.
    duration_seconds: Optional[float] = None


class TTSBackend(ABC):
    """
    Abstract base class for TTS backends.
    
    Each Docker-based model implements this interface via HTTP.
    The DockerBackendManager handles container lifecycle and HTTP communication.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of this backend."""
        pass
    
    @property
    @abstractmethod
    def backend_id(self) -> str:
        """Unique identifier for this backend (used in voice IDs, etc.)."""
        pass
    
    @abstractmethod
    def get_capabilities(self) -> TTSCapabilities:
        """Get the capabilities of this backend."""
        pass
    
    @abstractmethod
    def get_voices(self) -> list[TTSVoice]:
        """Get available voices."""
        pass
    
    @abstractmethod
    def synthesize(self, request: SynthesisRequest) -> SynthesisResponse:
        """Synthesize speech from text."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the backend is running and healthy."""
        pass
    
    # Optional extended methods (override if supported)
    
    def clone_voice(self, audio_bytes: bytes, name: str, sample_rate: int = 16000) -> TTSVoice:
        """
        Clone a voice from an audio sample.
        
        Returns a TTSVoice with the embedding populated.
        Raises NotImplementedError if voice cloning is not supported.
        """
        raise NotImplementedError("Voice cloning not supported by this backend")
    
    def delete_cloned_voice(self, voice_id: str) -> bool:
        """Delete a previously cloned voice."""
        raise NotImplementedError("Voice cloning not supported by this backend")
