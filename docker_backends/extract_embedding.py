"""
Extract speaker embedding from audio file inside a Kani HF container.

Usage (via docker exec):
    docker exec <hf-container> python /home/user/extract_embedding.py /home/user/audio.wav

Outputs the 128-dim embedding as a JSON array on stdout.
All other output goes to stderr so it doesn't corrupt the JSON.
"""

import sys
import os
import json
import numpy as np

audio_path = sys.argv[1]

# Redirect stdout to stderr during imports/init so library prints don't
# contaminate our JSON output.
_real_stdout = sys.stdout
sys.stdout = sys.stderr

sys.path.insert(0, "/home/user/app")

import soundfile as sf
from util import SpeakerEmbedder

audio, sr = sf.read(audio_path)

if audio.ndim > 1:
    audio = audio.mean(axis=1)

audio = audio.astype(np.float32)

embedder = SpeakerEmbedder()
embedding = embedder.embed_audio(audio, sample_rate=sr)

# Restore real stdout for the JSON output
sys.stdout = _real_stdout

emb_list = embedding[0].cpu().tolist()
print(json.dumps(emb_list))
