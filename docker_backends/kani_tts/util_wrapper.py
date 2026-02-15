"""
Wrapper to import util from the cloned space.

This handles the path setup so server.py can import SpeakerManager etc.
"""

import sys
import os

# Add the kani-space directory to path if not already there
space_path = "/app/kani-space"
if space_path not in sys.path and os.path.exists(space_path):
    sys.path.insert(0, space_path)

# Now import from the space's util.py
try:
    from util import SpeakerManager, InitModels, load_config, Examples
except ImportError:
    # Fallback: minimal implementation
    import json
    import torch
    
    class SpeakerManager:
        def __init__(self, speaker_map_path="./speakers/speaker_map.json"):
            self.speaker_map = {}
            self._embedder = None
            if os.path.exists(speaker_map_path):
                with open(speaker_map_path) as f:
                    self.speaker_map = json.load(f)
        
        def get_speaker_names(self):
            return list(self.speaker_map.keys())
        
        def get_speaker_emb(self, mode, speaker_name=None, json_emb=None):
            if mode == "select" and speaker_name in self.speaker_map:
                return self.speaker_map[speaker_name]
            return None
        
        def generate_embedding(self, audio_data):
            """Generate speaker embedding from audio."""
            if self._embedder is None:
                from kani_tts import SpeakerEmbedder
                self._embedder = SpeakerEmbedder()
            
            if isinstance(audio_data, tuple):
                sr, audio = audio_data
            else:
                sr, audio = 16000, audio_data
            
            return self._embedder.embed_audio(audio, sample_rate=sr)
