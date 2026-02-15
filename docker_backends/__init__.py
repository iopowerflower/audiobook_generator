"""
Docker-based TTS Backends

This module provides a common interface for TTS models running in Docker containers.
Each backend exposes a REST API, and this module handles communication.
"""

from .interface import TTSBackend, TTSCapabilities, TTSVoice
from .manager import DockerBackendManager

__all__ = ['TTSBackend', 'TTSCapabilities', 'TTSVoice', 'DockerBackendManager']
